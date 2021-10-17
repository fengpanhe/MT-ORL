import os

import torch
import torchvision
from addict import Dict as adict

from mtorl.utils.initial import (ENV, init_criterion, init_dataloader,
                                 init_model, init_optimizer, init_resume,
                                 init_scheduler)
from mtorl.utils.log import logger, set_logger
from mtorl.utils.runner import Runner
from parse_args import parse_args

_CURR_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    cfg = parse_args()
    cfg = adict(vars(cfg))
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    log_file = os.path.join(cfg.save_dir, 'log.txt')
    set_logger(log_file=log_file)
    cfg.mtorl_root_dir = './'

    logger.info(f"pytorch vision: {torch.__version__}")
    logger.info(f"torchvision vision: {torchvision.__version__}")
    logger.info(f"log_file: {log_file}")
    logger.info('*' * 80)
    logger.info('the args are the below')
    logger.info('*' * 80)
    for key, value in cfg.items():
        logger.info('{:<20}:{}'.format(key, str(value)))
    logger.info('*' * 80 + '\n')

    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


    ENV.checkpoint = init_resume(cfg)
    ENV.model = init_model(cfg, ENV.checkpoint)

    ENV.data_loaders = init_dataloader(cfg)

    ENV.optimizer = init_optimizer(cfg)
    ENV.criterion = init_criterion(cfg)

    if not cfg.inference:
        ENV.scheduler = init_scheduler(cfg)

    runner = Runner(cfg)
    runner.run()


if __name__ == '__main__':
    main()
