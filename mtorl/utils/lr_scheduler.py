import math
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler


class WarmUpLRScheduler(_LRScheduler):

    def __init__(self, optimizer, T_warmup, after_scheduler_name, **kwargs):
        self.T_warmup = T_warmup
        self.after_scheduler = getattr(lr_scheduler, after_scheduler_name)(optimizer=optimizer, **kwargs)
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.T_warmup:
            return self.after_scheduler.get_lr()

        return [base_lr * (self.last_epoch + 1.0) / (self.T_warmup + 1.0) for base_lr in self.base_lrs]

    def get_last_lr(self):
        if self.last_epoch >= self.T_warmup:
            self.after_scheduler.get_last_lr()

        return super(WarmUpLRScheduler, self).get_last_lr()

    def print_lr(self, is_verbose, group, lr, epoch=None):
        if self.last_epoch >= self.T_warmup:
            self.after_scheduler.print_lr(is_verbose, group, lr, epoch)

        return super(WarmUpLRScheduler, self).print_lr(is_verbose, group, lr, epoch)

    def step(self, epoch=None):
        if self.last_epoch >= self.T_warmup:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.T_warmup)
        else:
            return super(WarmUpLRScheduler, self).step(epoch)
