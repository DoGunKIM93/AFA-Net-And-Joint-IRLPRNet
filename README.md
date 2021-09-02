# AFA-Net
AFA-Net: Adaptive Feature Attention Network in Image Deblurring and Super-Resolution for Improving License Plate Recognition

## Abstract
Although a number of license plate recognition systems have become significantly advanced, they are still far from producing ideal images. They are still fiddled with low-resolution and motion blur, which are two of the most common problems in images extracted from automobile driving environment. In order to address this issue, we present a novel license plate recognition method that processes low-resolution and motion blurred images from dash cams. We propose a unique framework, AFA-Net (Adaptive Feature Attention Network), organized by image pre-restoration, feature composition and image reconstruction modules. The experimental results with 2,876 low-resolution and motion blurred images indicate that AFA-Net achieves 15.28\% improvement in recognition accuracy, 6.47\% in sequence similarity, and 3.89\% in character similarity, compared to the traditional license plate recognition models.


## Overview of our framework
![initial](https://user-images.githubusercontent.com/16958744/105068951-c7605080-5ac4-11eb-96f3-fab38861ce82.PNG)

## 
![initial](https://user-images.githubusercontent.com/16958744/131770038-dfc5f5a7-9edf-46f3-a0e0-25e356f7d048.PNG)

## PREREQUISITES
Prerequisites for AFA-Net.

## OS
AIR Research Framework is supported on Ubuntu 16.04 LTS or above.

## Python
It is recommended that use Python 3.7 or greater, which can be installed either through the Anaconda package manager or the Python website.

## Pytorch
Recommended that use Pytorch 1.5.0 or above version.
Important: EDVR or some models that have dependency on Deformable Convolution Networks feature only works in Pytorch 1.5.0a0+8f84ded.

## Pull container image
At the first, pull docker container image.
docker pull nvcr.io/nvidia/pytorch:20.03-py3

## Clone
```
git clone https://github.com/DoGunKIM93/AFA-Net.git
```

## Install some required packages
```
pip install fast_slic munch IQA_pytorch pillow
```

## Dataset
LBLP Dataset download : Due to the anonymity issue, the full LBLP dataset will be released after the conference acceptance. For now, we only provide the test of the dataset through the png file. (LBLP_samples)
```
datasetPath: 'dataset directory path' (in Param.yaml)
```

## Pre-trained
LBLP Pre-trained download : Due to the anoyomity issue, the pre-trained model will be released after the conference acceptacne. (Each file exceeds 100 MB)
```
pretrainedPath: 'Pre-trained directory path' (in Param.yaml)
```

## Train 
At AFA-Net folder, type following command:
```
python main.py
```
## Test
At AFA-Net folder, type following command:
```
python main.py -it
```
