import time
from contextlib import nullcontext

import torch
from mtorl.utils.initial import ENV
from mtorl.utils.log import logger
from torch.cuda.amp import GradScaler, autocast
from mtorl.utils.checkpoint import Checkpoint


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = ENV.model
        self.data_loaders = ENV.data_loaders

        self.optimizer = ENV.optimizer
        self.criterion = ENV.criterion
        self.scheduler = ENV.scheduler

        self.inference = cfg.inference
        self._max_epoch = cfg.epoch
        self.save_dir = cfg.save_dir
        self.scheduler_mode = cfg.scheduler_mode
        assert self.scheduler_mode == 'epoch' or self.scheduler_mode == 'iter'

        # amp
        self.amp_context = nullcontext
        if self.cfg.amp:
            self.amp_context = autocast
            self.scaler = GradScaler()

        self.iter = 0
        self.epoch = 0
        self.data_loader = None
        self.data_batch = None
        self.output = None
        self.vars_record = {'b_losses': [], 'o_loss': 0, 'count': 1}
        self.checkpoint = Checkpoint(self)

    def run(self):
        workflow = ['train', 'test']
        if self.inference:
            workflow = ['test']
            self._max_epoch = 1
        data_loaders = ENV.data_loaders

        for self.epoch in range(self._max_epoch):
            logger.info('*' * 80)
            logger.info(f'epoch_{self.epoch}')
            logger.info('*' * 80)
            for mode in workflow:
                logger.info(f'=> {mode}')
                self.epoch_start_time = time.time()
                epoch_runner = getattr(self, str(mode))
                data_loader = data_loaders[mode]
                self._max_iter = len(data_loader)

                epoch_runner(data_loader)

    def train(self, data_loader):
        self.data_loader = data_loader
        self.model.train()
        for self.iter, self.data_batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()

            with self.amp_context():
                self.output = self.batch_processor()

            loss = sum(self.output['b_losses']) + self.output['o_loss']
            if self.cfg.amp:
                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self._iter_record()
            if self.scheduler_mode == 'iter':
                self.scheduler.step()
        if self.scheduler_mode == 'epoch':
            self.scheduler.step()

        self._iter_record(force=True)
        self.checkpoint.save_checkpoint(self)

    def test(self, data_loader):
        self.data_loader = data_loader
        self.model.eval()
        for self.iter, self.data_batch in enumerate(self.data_loader):

            with torch.no_grad(), self.amp_context():
                self.output = self.batch_processor()

            self.checkpoint.save_test_results(self)
            self._iter_record(interval_iters=1000)

        self._iter_record(force=True)
        self.checkpoint.save_test_results(self, epoch_end=True)

    def batch_processor(self):
        images, labels = self.data_batch['image'], self.data_batch['labels']
        b_x, o_x = self.model(images)

        boundary_losses, orientation_loss = None, None
        if labels is not None and self.criterion is not None:
            boundary_losses, orientation_loss = self.criterion(b_x, o_x, labels)

        return {
            'image_name': self.data_batch['image_name'],
            'b_losses': boundary_losses,
            'o_loss': orientation_loss,
            'b_x': b_x,
            'o_x': o_x
        }

    def _iter_record(self, force=False, interval_iters=100):
        if self.output is not None and self.output['o_loss'] is not None:
            if len(self.vars_record['b_losses']) == 0:
                self.vars_record['b_losses'] = self.output['b_losses']
            else:
                self.vars_record['b_losses'] = [b1+b2.item() for b1, b2
                                                in zip(self.vars_record['b_losses'], self.output['b_losses'])]
            self.vars_record['o_loss'] = self.output['o_loss'].item()
            self.vars_record['count'] += 1

        if self.iter % interval_iters == 0 or force:
            lr = self.optimizer.param_groups[0]['lr']
            count = self.vars_record['count']
            b_losses_str = [f'{b_loss / count:.3e}' for b_loss in self.vars_record['b_losses']]
            log_str = f' Epoch [{self.epoch}/{self._max_epoch}][{self.iter}/{self._max_iter}] | '
            log_str += f'lr: {lr:.3e} | '
            log_str += f"b_losses: {b_losses_str} | "
            log_str += f"o_loss: {self.vars_record['o_loss'] / count:.3e} | "
            log_str += f'time: {(time.time() - self.epoch_start_time):.1f}s'
            logger.info(log_str)
            self.vars_record = {'b_losses': [], 'o_loss': 0, 'count': 1}
