'''
augmentation.py
'''
version = '1.10.200805'


#FROM Python LIBRARY
import os
import random
import math
import numpy as np

from PIL import Image as PILImage
from PIL import PngImagePlugin
from PIL.PngImagePlugin import PngImageFile
from typing import List, Dict, Tuple, Union, Optional


#FROM PyTorch
import torch

from torchvision import transforms
from torchvision import datasets


#from THIS Project
from backbone.torchvision_injected import functional as vF






######################################################################################################################################################################## 

# Tensor Augmentation "FUNCTIONs"   *NOT Classes



# Public Tensor Augmentation Functions Should be Made of Private Basis Functions 

######################################################################################################################################################################## 






######################################################################################################################################################################## 

# Public Tensor Augmentation Functions

# *** Input Must be 1st arg

# *** Input Must be [ data , label ]  -> List

######################################################################################################################################################################## 

def toTensor(xList: list):
    '''
    Paired toTensor Function
    '''
    return [_toTensor(xList[0]), _toTensor(xList[1])]


def sizeMatch(xList: list, interpolation = 2, matchIndex = 1):
    '''
    match data size to LABEL
    '''
    _, h, w = _getSize(xList[matchIndex])
    return [ _resize(x, h, w, interpolation) if i != matchIndex else x for i, x in enumerate(xList) ]



def centerCrop(xList: list, outputLabelHeight, outputLabelWidth):
    '''
    Paired Center Crop Function

    - Args
    1. xList : list ( data, label ) -> image, ndarray, tensor
    2. outputLabelHeight(Width) : output height(width) of label data

    - Behavior
    레이블을 주어진 인자대로 센터 크롭
    데이터를 레이블과 동일한 비율로 센터 크롭
    ex) Label(500x500) -> 100x100 Center Cropped
    Data (250x250) -> 50x50 Center Cropped  

    * 레이블 없을 시 데이터만 주어진 인자대로 센터 크롭
    '''
    
    xData = xList[0]
    xLabel = xList[1]

    if xLabel is not None:

        _, dH, dW = _getSize(xData)
        _, lH, lW = _getSize(xLabel)

        ratio = dH / lH

        outputDataHeight = math.ceil(ratio * outputLabelHeight)
        outputDataWidth = math.ceil(ratio * outputLabelWidth)

        xData = _centerCrop(xData, outputDataHeight, outputDataWidth)
        xLabel = _centerCrop(xLabel, outputLabelHeight, outputLabelWidth)

        return [xData, xLabel]

    else:

        return [_centerCrop(xData, outputLabelHeight, outputLabelWidth), None]






######################################################################################################################################################################## 

# Private Basis Functions 

######################################################################################################################################################################## 



def _getType(x) -> str:

    TYPEDICT = {
                PngImageFile : 'PIL',
                PILImage.Image : 'PIL',
                np.memmap : 'NPARRAY',
                np.ndarray : 'NPARRAY',
                torch.Tensor : 'TENSOR',
                }
    
    return TYPEDICT[type(x)]



def _getSize(x) -> List[int]:  # C H W

    if _getType(x) == 'PIL': #PIL Implemenataion
        sz = x.size
        sz = [len(x.getbands()), sz[1], sz[0]] 

    elif _getType(x) == 'TENSOR': #Tensor Implementation
        sz = list(x.size())[-3:]

    elif _getType(x) == 'NPARRAY':
        sz = list(x.shape)

    return sz 




def _toTensor(x) -> torch.Tensor:

    if _getType(x) == 'PIL': #PIL Implemenataion
        x = vF.to_tensor(x)

    elif _getType(x) == 'NPARRAY':
        x = torch.tensor(x)

    elif _getType(x) == 'TENSOR': #Tensor Implementation
        pass

    return x 





def _crop(x, top: int, left: int, height: int, width: int):

    if _getType(x) in ['PIL']: #PIL & Tensor Implemenataion
        x = vF.crop(x, top, left, height, width)

    elif _getType(x) in ['TENSOR']: #PIL & Tensor Implemenataion
        x = vF.crop(x, top, left, height, width)

    elif _getType(x) is 'NPARRAY': #Tensor Implementation
        x = x[..., top:top+height, left:left+width]

    return x



def _centerCrop(x, height, width):
    _, cH, cW = _getSize(x)
    x = _crop(x, (cH - height) // 2, (cW - width) // 2, height, width)
    return x


def _resize(x, height, width, interpolation=2):
    '''
    interpolation
    0 : Nearest neighbour
    2 : Bilinear
    3 : Bicubic

    숫자가 커질수록 품질이 좋고 속도가 느려짐
    '''
    if _getType(x) in ['PIL', 'TENSOR']: #PIL & Tensor Implemenataion
        x = vF.resize(x, [height, width], interpolation=interpolation)

    elif _getType(x) is 'NPARRAY': #Tensor Implementation
        x = torch.tensor(x)
        x = vF.resize(x, [height, width], interpolation=interpolation)
        x = x.numpy()

    return x

    










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

