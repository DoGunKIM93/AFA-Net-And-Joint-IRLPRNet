# AFA-Net: Adaptive Feature Attention Network in Image Deblurring and Super-Resolution for Improving License Plate Recognition

## Abstract
Although a number of license plate recognition (LPR) systems have become significantly advanced, they are still far from producing ideal images. They are still fiddled with low-resolution (LR) and motion blur, which are two of the most common problems in images extracted from automobile driving environment. In order to address this issue, we present a novel LPR method that processes LR and motion blurred images from dash cams. We propose a unique framework, AFA-Net (Adaptive Feature Attention Network), which synthesizes the characteristics of Super Resolution (SR) and deblurring sub-networks at the pixel and feature levels. The proposed AFA-Net, organized by image pre-restoration, feature composition and image reconstruction modules, can generate a clear restoration image for robust LPR performance with the images obtained from dash cams in an unconstrained environment. Furthermore, we explore the novel problem, Joint-IRLPRNet (Joint-Image Restoration and License Plate Recognition Network), that simultaneously address image restoration (i.e. SR and deblurring) and LPR in an end-to end trainable manner. Moreover, we introduce a dataset called LBLP (LR and blurred license plate (LP)). The dataset is composed of 2,779 LR and motion blurred cropped LP images, extracted from unconstrained dash cams. The experimental results on LBLP dataset indicate that AFA-Net achieves 15.28\% improvement in recognition accuracy, 6.47\% in sequence similarity, and 3.89\% in character similarity, compared to the traditional LPR model with image restoration model. Moreover, Joint-IRLPRNet can be more effective results than AFA-Net.

## Overview of our framework
![initial](https://user-images.githubusercontent.com/16958744/140759452-45664911-9c55-44e8-ba0a-d25f695c7817.png)
![initial](https://user-images.githubusercontent.com/16958744/131770465-3a0e0788-448a-4758-a715-7f6438ea08a1.PNG)

## Results
![initial](https://user-images.githubusercontent.com/16958744/131770166-e6a8f02d-65f1-4212-9e37-af0015772954.PNG)

## Joint-IRLPRNet
![initial](https://user-images.githubusercontent.com/16958744/140759723-c446fb51-3623-4110-a74a-3835dbac94ed.png)

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
LBLP Dataset download : google drive link (https://drive.google.com/file/d/1e95003bFPG30soCQan_opC6l3wHaIaU2/view?usp=drive_link)
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

## Evaluation Metrics (Supplementary material)
![initial](https://user-images.githubusercontent.com/16958744/140759646-3e0bd657-dd7c-466a-aeff-0701944804ec.png)
