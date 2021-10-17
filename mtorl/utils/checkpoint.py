import math
import os
import tarfile
import time

import imageio
import torch
from mtorl.utils.log import logger
from torchvision.io import write_png


def make_tar(output_path, source_path, include_prefix=None):
    assert os.path.exists(source_path), 'source path {source_path} does not exists'
    # arcname = os.path.basename(os.path.realpath(source_path))

    with tarfile.open(output_path, "w") as tar:
        for prefix in include_prefix:
            include_path = os.path.join(source_path, prefix)
            if not os.path.exists(include_path):
                logger.warning(f'include_path {include_path} not found')
            tar.add(include_path, arcname=prefix)


class Checkpoint(object):
    def __init__(self, runner):
        super().__init__()
        self.model_dir = os.path.join(runner.save_dir, 'model')
        self.test_results_dir = os.path.join(runner.save_dir, 'test_result')

        make_tar(output_path=os.path.join(runner.save_dir, 'mtorl_code.tar'),
                 source_path=runner.cfg.mtorl_root_dir,
                 include_prefix=['mtorl', 'main.py', 'parse_args.py'])
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)

    def save_checkpoint(self, runner):
        meta = dict(time=time.asctime(),
                    torch_version=torch.__version__,
                    epoch=runner.epoch,
                    iter=runner.iter)

        state = dict(state_dict=runner.model.state_dict(),
                     meta=meta,
                     cfg=runner.cfg)
        save_path = os.path.join(self.model_dir, f'checkpoint_{runner.epoch}.pth')
        torch.save(state, save_path)

    def save_test_results(self, runner, epoch_end=False):
        save_root_path = self.test_results_dir
        save_dir_name = f'epoch_{runner.epoch}_test_result'
        save_path = os.path.join(save_root_path, save_dir_name)
        if not os.path.exists(save_path):
            logger.info(f'save_test_path is {save_path}')
            os.makedirs(save_path)

        if epoch_end:
            save_tar_path = os.path.join(save_root_path, f'{save_dir_name}.tar')
            make_tar(save_tar_path, save_path, os.listdir(save_path))
            logger.info(f'tar {save_path} to {save_tar_path}')
            os.system(f'rm -r {save_path}')
        else:
            b_x, o_x = runner.output['b_x'], runner.output['o_x']
            image_name = runner.output['image_name'][0]
            boundary_result = runner.model.getBoundary(b_x)[0, 0]
            orientation_result = runner.model.getOrientation(o_x)[0, 0]

            boundary_result = (1 - boundary_result) * 255
            boundary_result = boundary_result.cpu().type(torch.uint8)

            orientation_result = orientation_result % (2 * math.pi)
            orientation_result = orientation_result * 255 / (2 * math.pi)
            orientation_result = orientation_result.cpu().type(torch.uint8)

            imageio.imwrite(os.path.join(save_path, f'{image_name}_boundary.png'), boundary_result)
            imageio.imwrite(os.path.join(save_path, f'{image_name}_orientation.png'), orientation_result)
