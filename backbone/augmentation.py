'''
augmentation.py
'''
version = '1.16.201230'

#FROM Python LIBRARY
import os
import random
import math
import numpy as np

from PIL import Image as PILImage
from PIL import ImageOps
from PIL import PngImagePlugin
from PIL.PngImagePlugin import PngImageFile
from PIL.JpegImagePlugin import JpegImageFile
from typing import List, Dict, Tuple, Union, Optional

#FROM PyTorch
import torch

from torchvision import transforms
from torchvision import datasets

#from THIS Project
import backbone.vision as vision

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
    return [_toTensor(x) for x in xList]


def sizeMatch(xList: list, interpolation = 3, matchIndex = 1):
    '''
    match data size to LABEL
    '''
    _, h, w = _getSize(xList[matchIndex])
    return [ _resize(x, h, w, interpolation) if i != matchIndex else x for i, x in enumerate(xList) ]


def resize(xList: list, outputLabelHeight, outputLabelWidth, interpolation = 3, labelIndex = -1):
    '''
    resize data with same ratio
    '''

    _, cH, cW = _getSize(xList[labelIndex])

    rst = []
    for x in xList:

        if x is not None:
            _, h, w = _getSize(x)
            ratioH = h / cH  
            ratioW = w / cW

            rst.append(_resize(x, int(outputLabelHeight * ratioH), int(outputLabelWidth * ratioW), interpolation))
        else:
            rst.append(None)

    return rst


def resizeToMultipleOf(xList: list, multiple, interpolation = 3):
    '''
    resize (shrink) data to multiple of X
    '''
    '''
    mnH = 10000000000
    mnW = 10000000000
    for x in xList:
        _, h, w = _getSize(x)
        if h < mn: 
            mnH = h
            mnW = w
    '''

    rst = []
    for x in xList:

        if x is not None:
            _, h, w = _getSize(x)

            rst.append(_resize(x, h // multiple * multiple, w // multiple * multiple, interpolation))
        else:
            rst.append(None)

    return rst


def centerCropToMultipleOf(xList: list, multiple):
    '''
    resize (shrink) data to multiple of X
    '''
    '''
    mnH = 10000000000
    mnW = 10000000000
    for x in xList:
        _, h, w = _getSize(x)
        if h < mn: 
            mnH = h
            mnW = w
    '''

    rst = []
    for x in xList:

        if x is not None:
            _, h, w = _getSize(x)

            rst.append(centerCrop([x, None], h // multiple * multiple, w // multiple * multiple)[0])
        else:
            rst.append(None)

    return rst



def shrink(xList: list, outputLabelHeight, outputLabelWidth, interpolation = 3, labelIndex = -1):

    _, cH, cW = _getSize(xList[labelIndex])

    # add resize blur ( 최솟값 기준 최댓값과의 비율의 제곱근 만큼 축소 했다가 확대 )
    minMaxRatioH = math.sqrt(outputLabelHeight / cH)
    minMaxRatioW = math.sqrt(outputLabelWidth / cW)
    
    
    rst = resize(xList, int(outputLabelHeight * minMaxRatioH), int(outputLabelWidth * minMaxRatioW), interpolation)
    rst = resize(xList, outputLabelHeight, outputLabelWidth, interpolation)

    return rst

def shrinkWithRandomMethod(xList: list, outputLabelHeight, outputLabelWidth, labelIndex = -1):

    _, cH, cW = _getSize(xList[labelIndex])

    # add resize blur ( 최솟값 기준 최댓값과의 비율의 제곱근 만큼 축소 했다가 확대 )
    minMaxRatioH = math.sqrt(outputLabelHeight / cH)
    minMaxRatioW = math.sqrt(outputLabelWidth / cW)
    
    
    if random.randint(0,1) == 0:
        rst = resize(xList, int(outputLabelHeight * minMaxRatioH), int(outputLabelWidth * minMaxRatioW), random.sample([0,2,3],1)[0])
        rst = resize(xList, outputLabelHeight, outputLabelWidth, random.sample([0,2,3],1)[0])
    else:
        rst = resize(xList, outputLabelHeight, outputLabelWidth, random.sample([0,2,3],1)[0])

    return rst


def resizeWithTextLabel(xList: list, outputLabelHeight, outputLabelWidth, interpolation = 3):

    assert len(xList) == 2

    return [ _resize(xList[0], outputLabelHeight, outputLabelWidth, interpolation), _resizeLabel(xList, outputLabelHeight, outputLabelWidth)]






def virtualScaling(xList: list, scale:int, interpolation = 3):
    '''
    make virturally downscaled image

    recommended input : [ GT, GT ]
    '''
    assert len(xList) == 2

    xData = xList[0]
    xLabel = xList[1]
    _, h, w = _getSize(xData)

    return [ _resize(xData, h//scale, w//scale, interpolation), xLabel ]


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

    #print("xD: ", len(xData))
    #print("xL: ", len(xLabel))


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



def randomCrop(xList: list, outputLabelHeight, outputLabelWidth, labelIndex = -1):

    
    _, cH, cW = _getSize(xList[labelIndex])
    rY = random.randint(0,cH - outputLabelHeight)
    rX = random.randint(0,cW - outputLabelWidth)

    rst = []
    for x in xList:
        if x is not None:
            _, h, w = _getSize(x)
            ratioH = h / cH  
            ratioW = w / cW

            rst.append(_crop(x, int(rY * ratioH), int(rX * ratioW), int(outputLabelHeight * ratioH), int(outputLabelWidth * ratioW)))
        else:
            rst.append(None)
            
    return rst


def randomCropWithRandomSize(xList: list, outputLabelHeightMin, outputLabelWidthMin, outputLabelHeightMax, outputLabelWidthMax, multipleOf=1, fixedRatio = 1, labelIndex = -1):

    if fixedRatio == 1:
        assert outputLabelHeightMax / outputLabelHeightMin == outputLabelWidthMax / outputLabelWidthMin

    outputLabelHeight = random.randint(outputLabelHeightMin, outputLabelHeightMax) // multipleOf * multipleOf

    if fixedRatio == 1:
        outputLabelWidth = outputLabelHeight * outputLabelWidthMax / outputLabelHeightMax
    else:
        outputLabelWidth = random.randint(outputLabelWidthMin, outputLabelWidthMax)

    rst = randomCrop(xList, outputLabelHeight, outputLabelWidth, labelIndex)

    return rst




def randomFlip(xList: list):
    xData = xList[0]
    xLabel = xList[1]

    h = True if random.randint(0,1) is 1 else False
    v = True if random.randint(0,1) is 1 else False

    rst = []
    for x in xList:
        if x is not None:
            rst.append(_flip(x, h, v))
        else:
            rst.append(None)

    return rst

    '''
    if xLabel is not None:

        xData = _flip(xData, h, v)
        xLabel = _flip(xLabel, h, v)

        return [xData, xLabel]

    else:

        return [_flip(xData, h, v), None]
    '''



def randomRotate(xList: list):
    xData = xList[0]
    xLabel = xList[1]

    a = random.randint(0,3)

    if xLabel is not None:

        xData = _rotate(xData, a)
        xLabel = _rotate(xLabel, a)

        return [xData, xLabel]

    else:

        return [_rotate(xData, a), None]


def randomGaussianBlur(xList: list, kernelSizeMin, kernelSizeMax, sigmaMin, sigmaMax):
    assert len(xList) == 2, "this function only implemented for [image, label]"
    assert kernelSizeMin % 2 == 1
    assert kernelSizeMax % 2 == 1
    assert kernelSizeMin <= kernelSizeMax

    kernelSize = random.sample(range(kernelSizeMin, kernelSizeMax, 2), 1)[0]
    sigma = random.uniform(sigmaMin, sigmaMax)
    return [_gaussianBlur(xList[0], kernelSize, sigma), xList[1]]


def gaussianNoise(xList: list):
    pass



######################################################################################################################################################################## 

# Private Basis Functions 

######################################################################################################################################################################## 


def _getType(x) -> str:

    TYPEDICT = {
                PngImageFile : 'PIL',
                JpegImageFile : 'PIL',
                PILImage.Image : 'PIL',
                np.memmap : 'NPARRAY',
                np.ndarray : 'NPARRAY',
                torch.Tensor : 'TENSOR',
                type(None) : 'NONE'
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




# Crop
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


def _randomCrop(x, height, width):
    _, cH, cW = _getSize(x[0])
    randH = random.randint(0,cH - height)
    randW = random.randint(0,cW - width)
    x[0] = _crop(x[0], randH, randW, height, width)

    if len(x) is 2:
        _, cH1, cW1 = _getSize(x[1])
        scale = math.ceil(cH1 / cH)
        x[1] = _crop(x[1], randH*scale, randW*scale, height*scale, width*scale)

    return x



def _flip(x, horiz, verti):

    if _getType(x) in ['PIL']: #PIL & Tensor Implemenataion

        if horiz is True:
            x = vF.hflip(x)
        if verti is True:
            x = vF.vflip(x)

    elif _getType(x) in ['TENSOR']: #PIL & Tensor Implemenataion

        if horiz is True:
            x = vF.hflip(x)
        if verti is True:
            x = vF.vflip(x)

    elif _getType(x) is 'NPARRAY': #Tensor Implementation

        if horiz is True: 
            x2 = np.flip(x,1)
            x = x2.copy()
        if verti is True:
            x2 = np.flip(x,0)
            x = x2.copy()

    return x


def _rotate(x, angle):

    if _getType(x) in ['PIL']: #PIL & Tensor Implemenataion
        x = vF.rotate(x, angle*90)

    elif _getType(x) in ['TENSOR']: #PIL & Tensor Implemenataion
        x = torch.rot90(x,angle,(2,3))

    elif _getType(x) is 'NPARRAY': #Tensor Implementation
        x2 = np.rot90(x,angle,(1,2))
        x = x2.copy()

    return x

def _resize(x, height, width, interpolation=3):
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


    
def _resizeLabel(xList: list, outputLabelHeight, outputLabelWidth):
    xData = xList[0] 
    xLabel = xList[1]

    _, h, w = _getSize(xData)
    h_ratio, w_ratio = outputLabelHeight/h, outputLabelWidth/w

    boxes = xLabel[:, :4].clone()
    labels = xLabel[:, -1].clone()
    landm = xLabel[:, 4:-1].clone()

    boxes[:, 0::2] *= h_ratio
    boxes[:, 1::2] *= w_ratio
    landm[:, 0::2] *= h_ratio
    landm[:, 1::2] *= w_ratio
    labels = np.expand_dims(labels, 1)

    xLabel = np.hstack((boxes, landm, labels))
    
    return xLabel


def _gaussianBlur(x, kernelSize, sigma):

    if _getType(x) in ['PIL']: #PIL & Tensor Implemenataion
        raise NotImplementedError('augmentation.py :: gaussianBlur Augmentation has not implemented for PIL Image due to speed issue. please use it for Tensor after toTensor() augmentation.')

    elif _getType(x) in ['TENSOR']: #PIL & Tensor Implemenataion
        x = vision.Gaussian(x, kernelSize, sigma)

    elif _getType(x) is 'NPARRAY': #Tensor Implementation
        raise NotImplementedError('augmentation.py :: gaussianBlur Augmentation has not implemented for NP array due to speed issue. please use it for Tensor after toTensor() augmentation.')

    return x







