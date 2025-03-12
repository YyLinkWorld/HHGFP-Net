# Hierarchical Heterogeneous Geometric Foreground Perception Network for Remote Sensing Object Detection
![image](./resources/HHGFP-Net.png)

## Abstract
Recently, deep learning-based remote sensing object detection has been widely explored and obtained remarkable performance. However, most existing multi-scale feature extraction methods neglect exploring the interfering representation of different hierarchical features in the backbone, which is crucial for learning more discriminative features. Moreover, FPN and its variants have difficulty in effectively perceiving the pose and salient information of remote sensing objects, leading to reduced detection accuracy. To address these issues, we propose a Hierarchical Heterogeneous Geometric Foreground Perception Network (HHGFP-Net) for remote sensing object detection. Specifically, a Hierarchical Heterogeneous Receptive-ﬁled Module (HHRM) is proposed to reward and penalize the feature information of the corresponding levels according to the differences between the shallow and deep feature layers in the backbone, improving discriminative feature ability. Furthermore, a Geometric Foreground Perception Feature Pyramid Network (GFP-FPN) is developed to refine geometric shapes and enhance foreground contents, providing more precise feature representations for objects, particularly small objects. Experimental results on four challenging remote sensing object detection datasets demonstrate that our HHGFP-Net achieves state-of-the-art performance.

## News!
- 2024-9-25 The code from the paper has been published on this page.
- 2025-3-7 This work has been accepted by IEEE TGRS.

## HHRM
![image](./resources/HHRM.png)

## GFP-FPN
![image](./resources/GFP-FPN.png)

## Installation

HHGFP-Net depends on [MMRotate](https://github.com/open-mmlab/mmrotate),[PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
Openmmlab provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)

## Results
[HHFEN pre-trained 300 epoch weights on ImageNet-1k](https://pan.baidu.com/s/1edUqy3Enz6cbbwdRjE3MnA) extraction code: aaa0

| Dataset | Train | Val | Test | Crop|epoch|mAP |
| ------------ | ------- | ------| ------- | ------ | ------- | ------ |
|DOTA-v1.0 | train+val | val | test | [0.5,1.0,1.5] | 12 | 80.68 |
|DOTA-v1.5 | train+val | val | test | [0.5,1.0,1.5]  | 12 | 77.54 |
|DIOR | train+val | test |test| [1.0]  | 12 | 72.34 |
|STAR|train+val | val |test | [0.5,1.0,1.5] |12| 39.90 |

## Visualization

### Heatmap Comparisons of LSKNet and HHGFP-Net in shallow layer at the backone.
- (a) input RGB image
- (b) LSKNet
- (c) HHGFP-Net

![image](./resources/attentionmap.png)


### Heatmap Comparisons of FPN and GFP-FPN before the final detection.
- (a) input RGB image
- (b) FPN
- (c) GFP-FPN
![image](./resources/AnaysisFPN.png)

### Visualization of GFRB
![image](./resources/visual_geometric.png)

### Results on  DOTA-v1.0
![image](./resources/analysis_dota.png)

### Results on  DOTA-v1.5
![image](./resources/analysis_dota15.png)

### Results on  DIOR
![image](./resources/analysis_dior.png)

### Results on  STAR
![image](./resources/analysis_star.png)

## Abaout the code and weights

The code that has been published so far has been reorganized according to the naming convention of the modules in the paper, and all the weights used in the experiments will be released after we reorganize the experiments.

### Copyright

The project has been licensed by Apache-2.0. Please refer to for details. [LICENSE.txt](./LICENSE)

## Thanks
[MMRotate](https://github.com/open-mmlab/mmrotate)
[MMYOLO](https://github.com/open-mmlab/mmrotate)
[MMPretrain](https://github.com/open-mmlab/mmrotate)
Center for Advanced Computing, School of Computer Science, China Three Gorges University
