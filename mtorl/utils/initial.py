import os

import torch
from addict import Dict as adict
from mtorl.datasets.piod_dataset import get_piod_dataloader
from mtorl.datasets.bsdsown_dataset import get_bsdsown_dataloader
from mtorl.models import OPNet
from mtorl.models.losses.occlusion_loss import OcclousionLoss
from mtorl.utils.log import logger
from mtorl.utils.lr_scheduler import WarmUpLRScheduler

__all__ = ['ENV', 'init_resume', 'init_model']

ENV = adict()
ENV.device = torch.device('cpu')


def _load_checkpoint(src_path: str):
    r"""
    Load checkpoint from local of hdfs
    """

    if not isinstance(src_path, str):
        return None

    if os.path.exists(src_path):
        return torch.load(src_path)
    else:
        logger.warning(f'file {src_path} is not found.')


def init_resume(cfg):
    checkpoint = None
    if cfg.resume is not None:
        checkpoint = _load_checkpoint(cfg.resume)
    if checkpoint is not None:
        logger.info(f'=> Model resume: loaded from {cfg.resume}\n')
    return checkpoint


def init_model(cfg, resume_checkpoint=None):
    logger.info(f"=> Model: {cfg.model_name}")
    if cfg.model_name == 'opnet':
        model = OPNet(backbone_name='resnet50',
                      num_ori_out=2,
                      use_pixel_shuffle=True,
                      orientation_path=True,
                      lite=False)
    if cfg.bankbone_pretrain is not None:
        model.load_backbone_pretrained(cfg.bankbone_pretrain)

    # Resume model if necessary
    if resume_checkpoint is not None:
        state_dict = resume_checkpoint.get('state_dict', resume_checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f'  Model resume')

    if cfg.cuda:
        model = model.cuda()
    return model


def init_dataloader(cfg):
    if cfg.dataset == 'piod':
        dataloader = get_piod_dataloader(cfg)
    elif cfg.dataset == 'bsdsown':
        dataloader = get_bsdsown_dataloader(cfg)
    else:
        raise ValueError(f'cfg.dataset={cfg.dataset} is invalid')

    return dataloader


def init_criterion(cfg):
    logger.info('=> Criterion')
    boundary_weights = cfg.boundary_weights.split(',')
    boundary_weights = list(map(float, boundary_weights))
    logger.info(f'   boundary_weights: {boundary_weights}')
    logger.info(f'   boundary_lambda: {cfg.boundary_lambda}')
    logger.info(f'   orientation_weight: {cfg.orientation_weight}')

    criterion = OcclousionLoss(
        boundary_weights=boundary_weights,
        boundary_lambda=cfg.boundary_lambda,
        orientation_weight=cfg.orientation_weight
    )
    if cfg.cuda:
        criterion = criterion.cuda()
    return criterion


def init_optimizer(cfg):
    logger.info('=> optimizer')
    logger.info(f'  optim_name: {cfg.optim}')
    logger.info(f'  lr: {cfg.base_lr}')
    logger.info(f'  weight_decay: {cfg.weight_decay}')
    module_name_scale = eval(cfg.module_name_scale)
    logger.info(f'  module_name_scale: {cfg.module_name_scale}')

    params_group = []
    for name, m in ENV.model.named_children():
        scale = module_name_scale.get(name, None)
        if scale is not None:
            params_group.append({'params': m.parameters(), 'lr': cfg.base_lr * scale})
        else:
            params_group.insert(0, {'params': m.parameters(), 'lr': cfg.base_lr})

    if cfg.optim == 'adamw':
        optim = torch.optim.AdamW(params_group, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'sgd':
        optim = torch.optim.SGD(params_group, momentum=cfg.momentum, lr=cfg.base_lr,
                                weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'cfg.optim={cfg.optim} is invalid')
    return optim


def init_scheduler(cfg):
    logger.info(' => scheduler')
    logger.info(f'  name: {cfg.scheduler_name}')
    logger.info(f'  scheduler_mode: {cfg.scheduler_mode}')
    logger.info(f'  warmup_epochs: {cfg.warmup_epochs}')

    scheduler_param = eval(cfg.scheduler_param)
    logger.info(f'  scheduler_param: {scheduler_param}')

    T_scale = 1
    if cfg.scheduler_mode == 'epoch':
        T_scale = 1
    elif cfg.scheduler_mode == 'iter':
        T_scale = len(ENV.data_loaders['train'])
    else:
        raise ValueError(f'cfg.scheduler_mode={cfg.scheduler_mode} is not supported')

    T_warmup = cfg.warmup_epochs * T_scale
    if cfg.scheduler_name == 'CosineAnnealingLR':
        if 'T_max' not in scheduler_param.keys():
            scheduler_param['T_max'] = cfg.epoch * T_scale

    scheduler = WarmUpLRScheduler(ENV.optimizer, T_warmup,
                                  after_scheduler_name=cfg.scheduler_name, **scheduler_param)
    return scheduler
