# MT-ORL: Multi-Task Occlusion Relationship Learning

Official implementation of paper "MT-ORL: Multi-Task Occlusion Relationship Learning" (ICCV 2021)

---

Paper:
[[ICCV2021]](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_MT-ORL_Multi-Task_Occlusion_Relationship_Learning_ICCV_2021_paper.html),
[[arXiv]](https://arxiv.org/abs/2108.05722)

Author:
Panhe Feng<sup>1,2</sup>,
[Qi She](http://scholar.google.com/citations?user=iHoGTt4AAAAJ&hl=en)<sup>2</sup>,
Lei Zhu<sup>1</sup>,
[Jiaxin Li](https://www.jiaxinli.me/)<sup>2</sup>,
Lin ZHANG<sup>2</sup>,
[Zijian Feng](https://vincentfung13.github.io/)<sup>2</sup>,
[Changhu Wang](https://scholar.google.com.sg/citations?user=DsVZkjAAAAAJ&hl=en)<sup>2</sup>,
Chunpeng Li<sup>1</sup>,
Xuejing Kang<sup>1</sup>,
Anlong Ming<sup>1</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications,
<sup>2</sup>ByteDance Inc.

## Introduction

Retrieving occlusion relation among objects in a single image is challenging due to sparsity of boundaries in image. We observe two key issues in existing works: firstly, lack of an architecture which can exploit the limited amount of coupling in the decoder stage between the two subtasks, namely occlusion boundary extraction and occlusion orientation prediction, and secondly, improper representation of occlusion orientation. In this paper, we propose a novel architecture called Occlusion-shared and Path-separated Network (OPNet), which solves the first issue by exploiting rich occlusion cues in shared high-level features and structured spatial information in task-specific low-level features. We then design a simple but effective orthogonal occlusion representation (OOR) to tackle the second issue. Our method surpasses the state-of-the-art methods by 6.1%/8.3% Boundary-AP and 6.5%/10% Orientation-AP on standard PIOD/BSDS ownership datasets.

## Citation

If you find our work helpful to your research, please cite our paper:

```
@InProceedings{Feng_2021_ICCV,
    author    = {Feng, Panhe and She, Qi and Zhu, Lei and Li, Jiaxin and Zhang, Lin and Feng, Zijian and Wang, Changhu and Li, Chunpeng and Kang, Xuejing and Ming, Anlong},
    title     = {MT-ORL: Multi-Task Occlusion Relationship Learning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9364-9373}
}
```

---

## Environmental Setup

Quick start full script:

```bash
conda create -n mtorl python=3.7 -y
conda activate mtorl
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install imageio h5py

# clone code
git clone https://github.com/fengpanhe/MT-ORL
cd MT-ORL
```

## Data Preparation

You can download two datasets we have processed from [here](https://1drv.ms/u/s!AlBUVia9fuD_nbBdTMMGiglxhRu8pg?e=dc23tx) (PIOD.zip and BSDSownership.zip), or follow the documentation of the [DOOBNet](https://github.com/GuoxiaWang/DOOBNet) to prepare two datasets.

Unzip PIOD.zip and BSDSownership.zip to `./data/`, the file structure is as followed:

```
data
├── BSDSownership
│   ├── Augmentation
│   ├── BSDS300
│   ├── testfg
│   ├── test.lst
│   ├── trainfg
│   └── train.lst
├── PIOD
│   ├── Aug_JPEGImages
│   ├── Aug_PngEdgeLabel
│   ├── Aug_PngOriLabel
│   ├── Data
│   ├── test_ids.lst
│   ├── train_ids.lst
│   └── val_doc_2010.txt
```

## Training

Download the Res50 weight file [resnet50s-a75c83cf.zip](https://s3.us-west-1.wasabisys.com/encoding/models/resnet50s-a75c83cf.zip) form [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), and unzip to `./data/`

### PASCAL Instance Occlusion Dataset (PIOD)

For training OPNet on PIOD dataset, you can run:

```bash
python3 main.py --cuda --amp --epoch 20  --base_lr 0.00003 \
    --dataset piod --dataset_dir data/PIOD \
    --bankbone_pretrain data/resnet50s-a75c83cf.pth \
    --save_dir result/piod_saved
```

### BSDS ownership

For training OPNet on BSDS ownership, you can run:

```bash
python3 main.py --cuda --amp --epoch 20 --boundary_lambda 1.1 \
    --dataset bsdsown --dataset_dir data/BSDSownership \
    --base_lr 0.0003 --module_name_scale "{'backbone': 0.1}" \
    --bankbone_pretrain data/resnet50s-a75c83cf.pth \
    --save_dir result/bsdsown_saved
```

## Evaluation

Here we provide the PIOD and the BSDS ownership dataset's evaluation and visualization code in `tools/doobscripts` folder (this code is modified from [DOOBNet/doobscripts](https://github.com/GuoxiaWang/DOOBNet)).

Matlab is required for evaluation. We have a python script (`tools/evaluate/evaluate_occ.py`) that calls the matlab evaluation program. you can follow
[Calling MATLAB from Python
](https://ww2.mathworks.cn/help/matlab/matlab-engine-for-python.html?lang=en)
to configure matlab for python.

To evaluate PIOD, you can run:

```bash
# Evaluate multiple
python tools/evaluate/evaluate_occ.py --dataset PIOD --occ 1 --epochs "5:20:2" --zip-dir result/piod_saved/test_result

# Evaluate one
python tools/evaluate/evaluate_occ.py --dataset PIOD --occ 1 --zipfile result/piod_saved/test_result/epoch_19_test_result.tar

```

To evaluate BSDSownership, you can run:

```bash
# Evaluate multiple
python tools/evaluate/evaluate_occ.py  --dataset BSDSownership --occ 1 --epochs "5:20:2" --zip-dir result/bsdsown_saved/test_result

# Evaluate one
python tools/evaluate/evaluate_occ.py --dataset BSDSownership --occ 1 --zipfile result/bsdsown_saved/test_result/epoch_19_test_result.tar
```

## Trained Models

Here we obtain better performance than those reported in the paper.

|    Dataset     | B-ODS | B-OIS | B-AP | O-ODS | O-OIS | O-AP |                                      model                                       |                                   test result                                   |
| :------------: | :---: | :---: | :--: | :---: | :---: | :--: | :------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------: |
|      PIOD      | 80.0  | 80.5  | 84.3 | 77.5  | 77.9  | 80.8 |  [PIOD_model.pth](https://1drv.ms/u/s!AlBUVia9fuD_nbBdTMMGiglxhRu8pg?e=dc23tx)   |  [PIOD_test.tar](https://1drv.ms/u/s!AlBUVia9fuD_nbBdTMMGiglxhRu8pg?e=dc23tx)   |
| BSDS ownership | 68.3  | 71.4  | 69.0 | 62.2  | 65.0  | 60.9 | [BSDSown_model.pth](https://1drv.ms/u/s!AlBUVia9fuD_nbBdTMMGiglxhRu8pg?e=dc23tx) | [BSDSown_test.tar](https://1drv.ms/u/s!AlBUVia9fuD_nbBdTMMGiglxhRu8pg?e=dc23tx) |

## Acknowledgement

The evaluation code `tools/doobscripts` is based on [DOOBNet/doobscripts](https://github.com/GuoxiaWang/DOOBNet). Thanks to the contributors of DOOBNet.

We use the ResNet50 with pretrained from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). Thanks to the contributors of PyTorch-Encoding.
