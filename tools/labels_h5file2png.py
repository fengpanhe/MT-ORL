import h5py
import os
import numpy as np
from skimage import io
import imageio

dataset_dir = 'data/PIOD'
train_lst = f'{dataset_dir}/Augmentation/train_pari_320x320.lst'
test_lst = f'{dataset_dir}/Augmentation/test.lst'
h5_label_dir = f'{dataset_dir}/Augmentation/Aug_HDF5EdgeOriLabel'
image_dir = f'{dataset_dir}/Augmentation/Aug_JPEGImages'

png_edge_label_dir = f'{dataset_dir}/Aug_PngEdgeLabel'
png_ori_label_dir = f'{dataset_dir}/Aug_PngOriLabel'
train_ids_lst = f'{dataset_dir}/Augmentation/train_ids.lst'
test_ids_lst = f'{dataset_dir}/Augmentation/test_ids.lst'


def labels_h5file2png():

    if not os.path.exists(png_edge_label_dir):
        os.makedirs(png_edge_label_dir)

    if not os.path.exists(png_ori_label_dir):
        os.makedirs(png_ori_label_dir)

    with open(train_lst, 'r') as f:
        img_path_list = f.readlines()
    train_img_name_list = [os.path.basename(img_path.split()[0]).split('.')[0] for img_path in img_path_list]
    with open(train_ids_lst, 'r') as f:
        for img_name in train_img_name_list:
            f.write(img_name + '\n')

    with open(test_lst, 'r') as f:
        img_path_list = f.readlines()
    test_img_name_list = [os.path.basename(img_path.split()[0]).split('.')[0] for img_path in img_path_list]
    with open(test_ids_lst, 'r') as f:
        for img_name in test_img_name_list:
            f.write(img_name + '\n')

    name_list = train_img_name_list + test_img_name_list
    for name in name_list:
        f = h5py.File(os.path.join(h5_label_dir, f'{name}.h5'), 'r')
        edge_label = f['label'][0][0] * 255
        ori_label = (f['label'][0][1] + np.pi) / (2 * np.pi) * 255
        imageio.imwrite(os.path.join(png_edge_label_dir, f'{name}.png'), edge_label.astype(np.uint8))
        imageio.imwrite(os.path.join(png_ori_label_dir, f'{name}.png'), ori_label.astype(np.uint8))
        f.close()


if __name__ == '__main__':
    labels_h5file2png()
