import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train for different args')
    parser.add_argument('--epoch', type=int, default=20, help='max epoch to train network, default is 20')
    parser.add_argument('--inference', action='store_true', help='amp')
    parser.add_argument('--seed', type=float, default=1, help='seed for torch.cuda.manual_seed()')
    parser.add_argument('--amp', action='store_true', help='amp')
    parser.add_argument('--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('--resume', type=str, default=None, help='whether resume from some, default is None')
    parser.add_argument('--save_dir', type=str, default='result', help='')

    # data
    parser.add_argument('--dataset', type=str, default='piod',  help='piod or bsdsown')
    parser.add_argument('--dataset_dir', type=str, default='data/PIOD',  help='data/PIOD or data/BSDSownership')
    parser.add_argument('--random_corp_size', type=int, default=320)
    parser.add_argument('--random_rotation_degrees', type=str, default='None', help='(0,360)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)

    # model
    parser.add_argument('--model_name', type=str, default='opnet')
    parser.add_argument('--bankbone_pretrain', type=str, default='data/resnet50-25c4b509.pth',
                        help='init net from pretrained model default is None')

    # train

    # loss
    parser.add_argument('--boundary_weights', type=str, default='0.5,0.5,0.5,0.5,0.5,1.1', help='')
    parser.add_argument('--orientation_weight', type=float, default=0.1, help='')
    parser.add_argument('--boundary_lambda', type=float, default=1.7, help='')

    # optim
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'radam', 'sgd'], help='self.optimizer')
    parser.add_argument('--base_lr', type=float, default=3e-5, help='the base learning rate of model')
    parser.add_argument('--module_name_scale', type=str, default="{}",
                        help='module_name= backbone, ori_convolution, ori_decoder, \
                            boundary_convolution, boundary_decoder, osm, encoder_sides, fuse')
    parser.add_argument('--weight_decay', type=float, default=0.002, help='the weight_decay of net')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum')

    parser.add_argument('--scheduler_name', type=str, default='MultiStepLR',
                        help='learning rate scheduler (default: MultiStepLR)')
    parser.add_argument('--scheduler_param', type=str, default="{'milestones':[5,10,15]}", help='')
    parser.add_argument('--scheduler_mode', type=str, default='epoch', help='epoch or iter')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup_epochs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(dir(args))
