'''
augmentation.py
'''
version = '1.0.200722'


#FROM Python LIBRARY
import os
import random
import math
import numpy as np

from PIL import Image
from PIL import PngImagePlugin


#FROM PyTorch
import torch

from torchvision import transforms
from torchvision import datasets







######################################################################################################################################################################## 

# Tensor Augmentation "FUNCTIONs"   *NOT Classes



# Public Tensor Augmentation Functions Should be Made of Private Basis Functions 

######################################################################################################################################################################## 






######################################################################################################################################################################## 

# Public Tensor Augmentation Functions

######################################################################################################################################################################## 


def adasd(x):
    
    mode = 'cpu' if x.get_device() is 'cpu' else 'gpu'

    x = _avvvxcv(x, mode)

    return x






######################################################################################################################################################################## 

# Private Basis Functions 

######################################################################################################################################################################## 


def _avvvxcv(x, mode):

    assert mode in ['cpu','gpu']

    if mode == 'cpu':
        pass
    elif mode == 'gpu':
        pass

    return










class PairedRandomCrop(object):

    def __init__(self, outputSize, samplingCount, ResizeMinMax = None, ResizeBlurFactor = None):

        assert isinstance(outputSize, (int, tuple, list))
        self.samplingCount = samplingCount
        self.ResizeMinMax = ResizeMinMax
        self.ResizeBlurFactor = ResizeBlurFactor

        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
            self.outputSize = outputSize


    def __call__(self, sample):

        resizeFactor = random.uniform(self.ResizeMinMax[0], self.ResizeMinMax[1])
        
        cropSize = [self.outputSize[0], self.outputSize[1]] 
        if self.ResizeMinMax is not None:
            cropSize[0] = math.ceil(cropSize[0] * resizeFactor)
            cropSize[1] = math.ceil(cropSize[1] * resizeFactor)
            

        if isinstance(sample[0], (tuple, list)):
            LRImage = sample[0][0]
            HRImage = sample[0][1]

            ratio = LRImage.size[0] / HRImage.size[0]

            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]
             
            newHRHeight, newHRWidth = cropSize
            newLRHeight = math.ceil(cropSize[0] * ratio)
            newLRWidth = math.ceil(cropSize[1] * ratio)

 
            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'

            rst = []

            for i in range(self.samplingCount):
                
                if HRHeight == newHRHeight: 
                    topHR = 0
                else:
                    topHR = np.random.randint(0, HRHeight - newHRHeight)
                if HRWidth == newHRWidth: 
                    leftHR = 0
                else:
                    leftHR = np.random.randint(0, HRWidth - newHRWidth)

                topLR = math.ceil(topHR * ratio)
                leftLR = math.ceil(leftHR * ratio)

                frames = []
                for imagePair in sample:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]

                    LRImageSampled = LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight))
                    HRImageSampled = HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))

                    if self.ResizeMinMax is not None:
                        LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio), math.ceil(self.outputSize[1]  * ratio)), resample = Image.BICUBIC)
                        HRImageSampled = HRImageSampled.resize(self.outputSize, resample = Image.BICUBIC)

                        if self.ResizeBlurFactor is not None and resizeFactor > 1:
                            blurFactor = 1 / math.sqrt(resizeFactor)

                            LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio * blurFactor), math.ceil(self.outputSize[1] * ratio * blurFactor)), resample = Image.BICUBIC)
                            HRImageSampled = HRImageSampled.resize((math.ceil(self.outputSize[0] * blurFactor), math.ceil(self.outputSize[1] * blurFactor)), resample = Image.BICUBIC)   

                            LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio), math.ceil(self.outputSize[1]  * ratio)), resample = Image.BICUBIC)
                            HRImageSampled = HRImageSampled.resize(self.outputSize, resample = Image.BICUBIC)    

                    frames.append([LRImageSampled, HRImageSampled])
                
                rst.append(frames)

            return rst
        else:
            LRImages = []
            HRImages = []

            LRImage = sample[0]
            HRImage = sample[1]

            ratio = LRImage.size[0] / HRImage.size[0]

            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]
            
            newHRHeight, newHRWidth = cropSize
            newLRHeight = math.ceil(cropSize[0] * ratio)
            newLRWidth = math.ceil(cropSize[1] * ratio)

            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'

            rst = []

            for i in range(self.samplingCount):
                
                if HRHeight == newHRHeight: 
                    topHR = 0
                else:
                    topHR = np.random.randint(0, HRHeight - newHRHeight)
                if HRWidth == newHRWidth: 
                    leftHR = 0
                else:
                    leftHR = np.random.randint(0, HRWidth - newHRWidth)

                topLR = math.ceil(topHR * ratio)
                leftLR = math.ceil(leftHR * ratio)

                LRImageSampled = LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight))
                HRImageSampled = HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))

                if self.ResizeMinMax is not None:
                    LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio), math.ceil(self.outputSize[1]  * ratio)), resample = Image.BICUBIC)
                    HRImageSampled = HRImageSampled.resize(self.outputSize, resample = Image.BICUBIC)

                rst.append([LRImageSampled, HRImageSampled])

            return rst


class PairedCenterCrop(object):

    def __init__(self, outputSize):
        assert isinstance(outputSize, (int, tuple, list))
        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
            self.outputSize = outputSize

    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            LRImage = sample[0][0]
            HRImage = sample[0][1]

            ratio = LRImage.size[0] / HRImage.size[0]

            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]
            
            newHRHeight, newHRWidth = self.outputSize
            newLRHeight = math.ceil(self.outputSize[0] * ratio)
            newLRWidth = math.ceil(self.outputSize[1] * ratio)

            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'
            
            if HRHeight == newHRHeight: 
                topHR = 0
            else:
                topHR = HRHeight//2 - newHRHeight//2
            if HRWidth == newHRWidth: 
                leftHR = 0
            else:
                leftHR = HRWidth//2 - newHRWidth//2


            topLR = math.ceil(topHR * ratio)
            leftLR = math.ceil(leftHR * ratio)

            rst = []
            for imagePair in sample:
                LRImage = imagePair[0]
                HRImage = imagePair[1]

                rst.append([LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight)), 
                            HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))])
            

            return rst
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            ratio = LRImage.size[0] / HRImage.size[0]


            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]

            newHRHeight, newHRWidth = self.outputSize
            newLRHeight = math.ceil(self.outputSize[0] * ratio)
            newLRWidth = math.ceil(self.outputSize[1] * ratio)

            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'
            
            if HRHeight == newHRHeight: 
                topHR = 0
            else:
                topHR = HRHeight//2 - newHRHeight//2
            if HRWidth == newHRWidth: 
                leftHR = 0
            else:
                leftHR = HRWidth//2 - newHRWidth//2


            topLR = math.ceil(topHR * ratio)
            leftLR = math.ceil(leftHR * ratio)

            LRImage = LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight))
            
            HRImage = HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))

            return [LRImage, HRImage]


class PairedRandomFlip(object):

    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            if random.random() > 0.5:
                frames = []
                for imagePair in sample:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]
                    frames.append([LRImage.transpose(Image.FLIP_LEFT_RIGHT), HRImage.transpose(Image.FLIP_LEFT_RIGHT)])
            else:
                frames = sample

            if random.random() > 0.5:
                rst = []
                for imagePair in frames:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]
                    rst.append([LRImage.transpose(Image.FLIP_TOP_BOTTOM), HRImage.transpose(Image.FLIP_TOP_BOTTOM)])
            else:
                rst = frames

            return rst
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            if random.random() > 0.5:
                LRImage = LRImage.transpose(Image.FLIP_LEFT_RIGHT)
                HRImage = HRImage.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:
                LRImage = LRImage.transpose(Image.FLIP_TOP_BOTTOM)
                HRImage = HRImage.transpose(Image.FLIP_TOP_BOTTOM)

            return [LRImage, HRImage]


class PairedRandomRotate(object):

    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            if random.random() > 0.5:
                frames = []
                for imagePair in sample:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]
                    frames.append([LRImage.rotate(90, expand=True), HRImage.rotate(90, expand=True)])
            else:
                frames = sample

            return frames
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            if random.random() > 0.5:
                LRImage = LRImage.rotate(90, expand=True)
                HRImage = HRImage.rotate(90, expand=True)

            return [LRImage, HRImage]


class ResizeByScaleFactor(object):

    def __init__(self, scale_factor, same_outputSize=False):
        self.scale_factor = scale_factor
        self.same_outputSize = same_outputSize

    def __call__(self, sample):
        if isinstance(sample, (tuple, list)):
            W = sample[0].size[0]
            H = sample[0].size[1]
            rW = math.ceil(sample[0].size[0] * self.scale_factor)
            rH = math.ceil(sample[0].size[1] * self.scale_factor)

            rst = []

            for img in sample:
            
                rszimg = img.resize((rW,rH), resample = Image.BICUBIC)

                if(self.same_outputSize == True):
                    rszimg = rszimg.resize((W,H), resample = Image.BICUBIC)

                rst.append(rszimg)

            return rst
        else:
            W = sample.size[0]
            H = sample.size[1]
            rW = math.ceil(sample.size[0] * self.scale_factor)
            rH = math.ceil(sample.size[1] * self.scale_factor)
            
            sample = sample.resize((rW,rH), resample = Image.BICUBIC)

            if(self.same_outputSize == True):
                sample = sample.resize((W,H), resample = Image.BICUBIC)

            return sample


class testTransform(object):

    def __init__(self, txt):
        self.txt = txt

    def __call__(self, sample):
        print(self.txt, sample)

        return sample


class PairedSizeMatch(object):
    '''
    sample[0] 을 [1] 사이즈로 리사이즈 합니다 
    '''
    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            rst = []
            for imagePair in sample:
                LRImage = imagePair[0]
                HRImage = imagePair[1]

                rst.append([LRImage.resize(HRImage.size[:2]), HRImage])

            return rst
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            LRImage = LRImage.resize(HRImage.size[:2])

            return [LRImage, HRImage]

