"""
augmentation.py
"""

# FROM Python LIBRARY
import random
import math
import numpy as np
import time

import cv2

import imgaug

from PIL import Image as PILImage
from PIL import PngImagePlugin
from PIL.PngImagePlugin import PngImageFile
from PIL.JpegImagePlugin import JpegImageFile
from PIL.BmpImagePlugin import BmpImageFile
from PIL.MpoImagePlugin import MpoImageFile
from PIL.GifImagePlugin import GifImageFile
from PIL.TiffImagePlugin import TiffImageFile
from typing import List, Dict, Tuple, Union, Optional

# FROM PyTorch
import torch

# from THIS Project
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




########################################################################################################################################################################

# Type Transform

########################################################################################################################################################################

def toTensor(xList: list):
    """
    Paired toTensor Function
    """
    return [_toTensor(x) for x in xList]




########################################################################################################################################################################

# Channel Operation

########################################################################################################################################################################


def toRGB(xList: list):
    return [_toRGB(x) for x in xList]

def toGrayscale(xList: list):
    return [_toGrayscale(x) for x in xList]




########################################################################################################################################################################

# Resize

########################################################################################################################################################################

def sizeMatch(xList: list, interpolation='bicubic', labelIndex=-1):
    """
    match data size to LABEL
    """
    if labelIndex < 0: labelIndex = len(xList) + labelIndex
    _, h, w = _getSize(xList[labelIndex])
    return [_resize(x, h, w, interpolation) if i != labelIndex else x for i, x in enumerate(xList)]


def resize(xList: list, outputLabelHeight=None, outputLabelWidth=None, interpolation='bicubic', labelIndex=-1, applyIndices=None, scaleFactor=None):
    """
    resize data with same ratio
    """

    assert (outputLabelHeight is not None and outputLabelWidth is not None) or scaleFactor is not None
    assert not (outputLabelHeight is not None and outputLabelWidth is not None and scaleFactor is not None)


    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    if applyIndices is None:
        applyIndices = range((len(xList)))
    applyIndices = [x if x >= 0 else len(xList) + x for x in applyIndices]

    _, cH, cW = _getSize(xList[labelIndex])

    if scaleFactor is not None:
        outputLabelHeight = int(cH * scaleFactor)
        outputLabelWidth = int(cW * scaleFactor)


    rst = []
    for i, x in enumerate(xList):

        if i not in applyIndices:
            rst.append(x)
        elif x is not None:
            _, h, w = _getSize(x)
            ratioH = h / cH
            ratioW = w / cW

            rst.append(_resize(x, int(outputLabelHeight * ratioH), int(outputLabelWidth * ratioW), interpolation))
        else:
            rst.append(None)

    return rst



def randomResize(xList: list, outputLabelHeightMin, outputLabelWidthMin, outputLabelHeightMax, outputLabelWidthMax, interpolation='bicubic', fixedRatio=True, labelIndex=-1, applyIndices=None):
    """
    random resize data with same ratio
    """
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    if applyIndices is None:
        applyIndices = range((len(xList)))
    applyIndices = [x if x >= 0 else len(xList) + x for x in applyIndices]

    _, cH, cW = _getSize(xList[labelIndex])

    rst = []
    for i, x in enumerate(xList):

        if i not in applyIndices:
            rst.append(x)
        elif x is not None:
            _, h, w = _getSize(x)
            ratioH = h / cH
            ratioW = w / cW

            outputLabelHeight = random.randint(outputLabelHeightMin, outputLabelHeightMax)
            if fixedRatio in [1, True]:
                _r = outputLabelHeight / h
                outputLabelWidth = _r * w
            else:
                outputLabelWidth = random.randint(outputLabelWidthMin, outputLabelWidthMax)

            rst.append(_resize(x, int(outputLabelHeight * ratioH), int(outputLabelWidth * ratioW), interpolation))
        else:
            rst.append(None)

    return rst


def resizeToMultipleOf(xList: list, multiple, interpolation='bicubic'):
    """
    resize (shrink) data to multiple of X
    """

    rst = []
    for x in xList:

        if x is not None:
            _, h, w = _getSize(x)

            rst.append(_resize(x, h // multiple * multiple, w // multiple * multiple, interpolation))
        else:
            rst.append(None)

    return rst


def shrinkAndExpand(xList: list, outputLabelHeight, outputLabelWidth, shrinkInterpolation='bicubic', expandInterpolation='bicubic', labelIndex=-1):
    """
    shrink then expand to Add Resize Blur
    interpolation methods can be 'random'
    """
    # add resize blur ( 최솟값 기준 최댓값과의 비율의 제곱근 만큼 축소 했다가 확대 )

    _, cH, cW = _getSize(xList[labelIndex])

    # add resize blur (  )
    minMaxRatioH = math.sqrt(outputLabelHeight / cH)
    minMaxRatioW = math.sqrt(outputLabelWidth / cW)

    shrinkHeight = int(outputLabelHeight * minMaxRatioH)
    shrinkWidth = int(outputLabelWidth * minMaxRatioW)

    rst = resize(xList, shrinkHeight, shrinkWidth, shrinkInterpolation, labelIndex)
    rst = resize(xList, outputLabelHeight, outputLabelWidth, expandInterpolation, labelIndex)

    return rst


def resizeWithTextLabel(xList: list, outputLabelHeight, outputLabelWidth, interpolation='bicubic'):

    assert len(xList) == 2

    print("augmentation.py :: WARNING : Function 'resizeWithTextLabel' has old behaviour. It is recommand to update code before using this function.")

    return [
        _resize(xList[0], outputLabelHeight, outputLabelWidth, interpolation),
        _resizeLabel(xList, outputLabelHeight, outputLabelWidth),
    ]


def virtualScaling(xList: list, scale: int, interpolation='bicubic', labelIndex=-1):
    """
    make virturally downscaled image

    recommended input : [ GT, GT ]
    """
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    _, h, w = _getSize(xList[labelIndex])

    rst = []
    for i, x in enumerate(xList):

        if i is labelIndex:
            rst.append(x)
        elif x is not None:
            rst.append(_resize(x, h // scale, w // scale, interpolation))
        else:
            rst.append(None)

    return rst

def virtualResizing(xList: list, outputHeight: int, outputWidth: int, interpolation='bicubic', labelIndex=-1):
    """
    make virturally downscaled image

    recommended input : [ GT, GT ]
    """
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    _, h, w = _getSize(xList[labelIndex])

    rst = []
    for i, x in enumerate(xList):

        if i is labelIndex:
            rst.append(x)
        elif x is not None:
            rst.append(_resize(x, outputHeight, outputWidth, interpolation))
        else:
            rst.append(None)

    return rst



########################################################################################################################################################################

# Crop

########################################################################################################################################################################


def centerCrop(xList: list, outputLabelHeight, outputLabelWidth, labelIndex=-1):
    """
    Paired Center Crop Function

    - Args
    1. xList : list -> image, tensor
    2. outputLabelHeight(Width) : output height(width) of label data

    - Behavior
    레이블을 주어진 인자대로 센터 크롭
    데이터를 레이블과 동일한 비율로 센터 크롭
    ex) Label(500x500) -> 100x100 Center Cropped
    Data (250x250) -> 50x50 Center Cropped
    """
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    _, lH, lW = _getSize(xList[labelIndex])

    rst = []
    for i, x in enumerate(xList):

        if x is not None:
            _, dH, dW = _getSize(x)
            ratio = dH / lH
            outputDataHeight = math.ceil(ratio * outputLabelHeight)
            outputDataWidth = math.ceil(ratio * outputLabelWidth)
            rst.append(_centerCrop(x, outputDataHeight, outputDataWidth))
        else:
            rst.append(None)

    return rst


def randomCrop(xList: list, outputLabelHeight, outputLabelWidth, labelIndex=-1):
    '''
    Crop at random location, fixed size
    '''
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    _, cH, cW = _getSize(xList[labelIndex])
    rY = random.randint(0, cH - outputLabelHeight)
    rX = random.randint(0, cW - outputLabelWidth)

    rst = []
    for i, x in enumerate(xList):

        if x is not None:
            _, h, w = _getSize(x)
            ratioH = h / cH
            ratioW = w / cW

            rst.append(
                _crop(x, int(rY * ratioH), int(rX * ratioW), int(outputLabelHeight * ratioH), int(outputLabelWidth * ratioW))
            )
        else:
            rst.append(None)

    return rst


def centerCropToMultipleOf(xList: list, multiple):
    """
    resize (shrink) data to multiple of X
    """

    rst = []
    for x in xList:

        if x is not None:
            _, h, w = _getSize(x)

            rst.append(centerCrop([x, None], h // multiple * multiple, w // multiple * multiple)[0])
        else:
            rst.append(None)

    return rst


def randomCropWithRandomSize(
    xList: list,
    outputLabelHeightMin,
    outputLabelWidthMin,
    outputLabelHeightMax,
    outputLabelWidthMax,
    multipleOf=1,
    fixedRatio=True,
    labelIndex=-1,
):
    '''
    Crop at random location, random size
    '''
    if fixedRatio in [1, True]:
        assert outputLabelHeightMax / outputLabelHeightMin == outputLabelWidthMax / outputLabelWidthMin

    outputLabelHeight = random.randint(outputLabelHeightMin, outputLabelHeightMax) // multipleOf * multipleOf

    if fixedRatio in [1, True]:
        outputLabelWidth = outputLabelHeight * outputLabelWidthMax / outputLabelHeightMax
    else:
        outputLabelWidth = random.randint(outputLabelWidthMin, outputLabelWidthMax)

    rst = randomCrop(xList, outputLabelHeight, outputLabelWidth, labelIndex)

    return rst



########################################################################################################################################################################

# Spatial Operation

########################################################################################################################################################################


def horizontalFlip(xList: list):
    '''
    Horizontal Flip
    '''
    rst = []
    for x in xList:
        if x is not None:
            rst.append(_flip(x, horiz=True, verti=False))
        else:
            rst.append(None)

    return rst


def verticalFlip(xList: list):
    '''
    Vertical Flip
    '''
    rst = []
    for x in xList:
        if x is not None:
            rst.append(_flip(x, horiz=False, verti=True))
        else:
            rst.append(None)
    return rst


def random90Rotate(xList: list):
    '''
    Rotating Image to (0, 90, 180, 270) Degrees
    '''
    deg = random.randint(0, 3)

    rst = []
    for x in xList:
        if x is not None:
            rst.append(_rotate90(x, deg))
        else:
            rst.append(None)
    return rst




########################################################################################################################################################################

# Degradation

########################################################################################################################################################################


def gaussianBlur(xList: list, strengthMin, strengthMax, labelIndex=-1):
    '''
    Gaussian Blur with Random Sigma Range.
    1 >= strength >= 0, (2*strength+1 = Sigma)
    '''
    #assert kernelSizeMin % 2 == 1
    #assert kernelSizeMax % 2 == 1
    #assert kernelSizeMin <= kernelSizeMax

    #kernelSize = random.sample(range(kernelSizeMin, kernelSizeMax, 2), 1)[0]
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    sigmaMin = strengthMin * 2+1
    sigmaMax = strengthMax * 2+1
    sigma = random.uniform(sigmaMin, sigmaMax)

    rst = []
    for i, x in enumerate(xList):
        if i == labelIndex:
            rst.append(x)
        elif x is not None:
            rst.append(_gaussianBlur(x, sigma))
        else:
            rst.append(None)
    return rst


def gaussianNoise(xList: list, strengthMin, strengthMax, labelIndex=-1):
    '''
    Gaussian Noise with Random Strength Range.
    1 >= Strength >= 0
    '''
    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    strengthMin = strengthMin * 15
    strengthMax = strengthMax * 15
    strength = random.uniform(strengthMin, strengthMax)

    rst = []
    for i, x in enumerate(xList):
        if i == labelIndex:
            rst.append(x)
        elif x is not None:
            rst.append(_gaussianNoise(x, strength))
        else:
            rst.append(None)
    return rst


def motionBlur(xList: list, kernelSizeMin, kernelSizeMax, angleMin, angleMax, directionMin, directionMax, labelIndex=-1):
    '''
    Uniform Motion Blur
    360 >= angle >= 0
    1 >= direction >= -1
    '''
    assert kernelSizeMin <= kernelSizeMax
    assert angleMin <= angleMax
    assert directionMin <= directionMax

    kernelSize = random.sample(range(kernelSizeMin, kernelSizeMax, 1), 1)[0]
    angle = random.uniform(angleMin, angleMax)
    direction = random.uniform(directionMin, directionMax)

    if labelIndex < 0: labelIndex = len(xList) + labelIndex

    rst = []
    for i, x in enumerate(xList):
        if i == labelIndex:
            rst.append(x)
        elif x is not None:
            rst.append(_motionBlur(x, kernelSize, angle, direction))
        else:
            rst.append(None)
    return rst


def JPEGCompress(xList: list, strengthMin, strengthMax, labelIndex=-1):
    '''
    JPEG Compress
    1(quality 0) >= strength >= 0(quality 100)
    '''
    assert strengthMin <= strengthMax
    assert 1 >= strengthMax
    assert strengthMin >= 0

    strength = random.uniform(strengthMin, strengthMax)
    quality = round((1 - strength) * 100)

    if labelIndex is not None and labelIndex < 0: labelIndex = len(xList) + labelIndex

    rst = []
    for i, x in enumerate(xList):
        if i == labelIndex:
            rst.append(x)
        elif x is not None:
            rst.append(_JPEGCompress(x, quality))
        else:
            rst.append(None)
    return rst




########################################################################################################################################################################

# Color Adjustment

########################################################################################################################################################################


def adjustHue(xList, factorMin, factorMax, labelIndex=-1):
    """
    Adjust Hue (0.5 >= factor >= -0.5)
    """
    if labelIndex is not None and labelIndex < 0: labelIndex = len(xList) + labelIndex

    factor = random.uniform(factorMin, factorMax)

    rst = []
    for i, x in enumerate(xList):

        if labelIndex is not None and i is labelIndex:
            rst.append(x)

        elif x is not None:
            rst.append(_adjustHue(x, factor))

        else:
            rst.append(None)

    return rst
    
def adjustBrightness(xList, factorMin, factorMax, labelIndex=-1):
    """
    factor 0 : Black
    factor 1 : Original
    factor N : Brightness*=N
    """
    if labelIndex is not None and labelIndex < 0: labelIndex = len(xList) + labelIndex


    factor = random.uniform(factorMin, factorMax)

    rst = []
    for i, x in enumerate(xList):

        if labelIndex is not None and i is labelIndex:
            rst.append(x)

        elif x is not None:
            rst.append(_adjustBrightness(x, factor))

        else:
            rst.append(None)

    return rst

def adjustContrast(xList, factorMin, factorMax, labelIndex=-1):
    """
    factor 0 : Gray
    factor 1 : Original
    factor N : Contrast*=N
    """
    if labelIndex is not None and labelIndex < 0: labelIndex = len(xList) + labelIndex

    factor = random.uniform(factorMin, factorMax)

    rst = []
    for i, x in enumerate(xList):

        if labelIndex is not None and i is labelIndex:
            rst.append(x)

        elif x is not None:
            rst.append(_adjustContrast(x, factor))

        else:
            rst.append(None)

    return rst

def adjustSaturation(xList, factorMin, factorMax, labelIndex=-1):
    """
    factor 0 : Grayscale
    factor 1 : Original
    factor N : Saturation*=N
    """
    if labelIndex is not None and labelIndex < 0: labelIndex = len(xList) + labelIndex

    factor = random.uniform(factorMin, factorMax)

    rst = []
    for i, x in enumerate(xList):

        if labelIndex is not None and i is labelIndex:
            rst.append(x)

        elif x is not None:
            rst.append(_adjustSaturation(x, factor))

        else:
            rst.append(None)

    return rst

def adjustGamma(xList, gammaMin, gammaMax, gainMin, gainMax, labelIndex=-1):
    if labelIndex is not None and labelIndex < 0: labelIndex = len(xList) + labelIndex

    gamma = random.uniform(gammaMin, gammaMax)
    gain = random.uniform(gainMin, gainMax)

    rst = []
    for i, x in enumerate(xList):

        if labelIndex is not None and i is labelIndex:
            rst.append(x)

        elif x is not None:
            rst.append(_adjustGamma(x, gamma,gain))

        else:
            rst.append(None)

    return rst





########################################################################################################################################################################

# Normalize

########################################################################################################################################################################


def normalize3Ch(xList: list, meanC1, meanC2, meanC3, stdC1, stdC2, stdC3):
    return [_normalize(x, [meanC1, meanC2, meanC3], [stdC1, stdC2, stdC3]) for x in xList]











########################################################################################################################################################################

# Private Basis Functions

########################################################################################################################################################################


def _getType(x) -> str:

    TYPEDICT = {
        PngImageFile: "PIL",
        JpegImageFile: "PIL",
        PILImage.Image: "PIL",
        BmpImageFile: "PIL",
        MpoImageFile: "PIL",
        GifImageFile: "PIL",
        TiffImageFile: "PIL",
        torch.Tensor: "TENSOR",
        type(None): "NONE",
    }

    return TYPEDICT[type(x)]


def _getSize(x) -> List[int]:  # C H W

    if _getType(x) == "PIL":  # PIL Implemenataion
        sz = x.size
        sz = [len(x.getbands()), sz[1], sz[0]]

    elif _getType(x) == "TENSOR":  # Tensor Implementation
        sz = list(x.size())[-3:]

    return sz


def _toTensor(x) -> torch.Tensor: #0~255 INT    C H W

    if _getType(x) == "PIL":  # PIL Implemenataion
        x = vF.to_tensor(x)

    elif _getType(x) == "TENSOR":  # Tensor Implementation
        pass

    return x

'''
def _toNPArray(x): #float32 0~1   H W C

    if _getType(x) == "PIL":  # PIL Implemenataion
        x = np.array(x, dtype=np.float32).transpose(2,0,1) / 255

    elif _getType(x) == "TENSOR":  # Tensor Implementation
        x = x.numpy()

    return x
'''


def _toPIL(x): #float32 0~1   W H C

    if _getType(x) == "PIL":  # PIL Implemenataion
        pass

    elif _getType(x) == "TENSOR":  # Tensor Implementation
        x = vF.to_pil_image(x)

    return x


# Crop
def _crop(x, top: int, left: int, height: int, width: int):

    if _getType(x) in ["PIL"]:  # PIL & Tensor Implemenataion
        x = vF.crop(x, top, left, height, width)

    elif _getType(x) in ["TENSOR"]:  # PIL & Tensor Implemenataion
        x = vF.crop(x, top, left, height, width)

    return x


def _centerCrop(x, height, width):
    _, cH, cW = _getSize(x)
    x = _crop(x, (cH - height) // 2, (cW - width) // 2, height, width)
    return x


def _randomCrop(x, height, width):
    _, cH, cW = _getSize(x[0])
    randH = random.randint(0, cH - height)
    randW = random.randint(0, cW - width)
    x[0] = _crop(x[0], randH, randW, height, width)

    if len(x) == 2:
        _, cH1, cW1 = _getSize(x[1])
        scale = math.ceil(cH1 / cH)
        x[1] = _crop(x[1], randH * scale, randW * scale, height * scale, width * scale)

    return x


def _flip(x, horiz, verti):

    if _getType(x) in ["PIL"]:  # PIL & Tensor Implemenataion

        if horiz is True:
            x = vF.hflip(x)
        if verti is True:
            x = vF.vflip(x)

    elif _getType(x) in ["TENSOR"]:  # PIL & Tensor Implemenataion

        if horiz is True:
            x = vF.hflip(x)
        if verti is True:
            x = vF.vflip(x)

    return x


def _rotate90(x, angle):

    if _getType(x) in ["PIL"]:  # PIL & Tensor Implemenataion
        x = vF.rotate(x, angle * 90)

    elif _getType(x) in ["TENSOR"]:  # PIL & Tensor Implemenataion
        x = torch.rot90(x, angle, (-2, -1))

    return x


def _resize(x, height, width, interpolation='bicubic'):
    """
    숫자가 커질수록 품질이 좋고 속도가 느려짐

    torchvision의 tensor interpolation이 PIL과 달라 부득이하게 PIL로 변환 후 리사이즈 후 원래대로 변환함 (속도 저하됨)

    """
    interpolation = _interpolationMethodToInt(interpolation)

    if _getType(x) == "PIL":  # PIL & Tensor Implemenataion
        x = vF.resize(x, [height, width], interpolation=interpolation)

    elif _getType(x) == "TENSOR":  # Tensor Implementation
        x = _toPIL(x)
        x = vF.resize(x, [height, width], interpolation=interpolation)
        x = _toTensor(x)

    return x


def _resizeLabel(xList: list, outputLabelHeight, outputLabelWidth):
    xData = xList[0]
    xLabel = xList[1]

    _, h, w = _getSize(xData)
    h_ratio, w_ratio = outputLabelHeight / h, outputLabelWidth / w

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


def _gaussianBlur(x, sigma):

    '''
    if _getType(x) in ["PIL"]:  # PIL & Tensor Implemenataion
        raise NotImplementedError(
            "augmentation.py :: gaussianBlur Augmentation has not implemented for PIL Image due to speed issue. please use it for Tensor after toTensor() augmentation."
        )

    elif _getType(x) in ["TENSOR"]:  # PIL & Tensor Implemenataion
        x = vision.Gaussian(x, kernelSize, sigma)
    '''
    assert 1.0 <= sigma <= 3.0
    x = _applyImgAug(x, imgaug.augmenters.blur.GaussianBlur(sigma))

    return x


def _gaussianNoise(x, strength):

    assert 0.0 <= strength <= 15.0
    x = _applyImgAug(x, imgaug.augmenters.arithmetic.AdditiveGaussianNoise(0, strength))

    return x


def _motionBlur(x, kernelSize, angle, direction):

    x = _applyImgAug(x, imgaug.augmenters.blur.MotionBlur(k=kernelSize, angle=angle, direction=direction))
    return x


def _nonUniformMotionBlur(x): 
    pass

def _JPEGCompress(x, quality):

    if _getType(x) in ["PIL"]:  # PIL & Tensor Implemenataion
        x = np.asarray(x) #RGB
        x = x[:, :, ::-1].copy() #BGR

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, x = cv2.imencode('.jpg', x, encode_param)
        x = cv2.imdecode(x,1)

        x = x[:, :, ::-1].copy() #RGB
        x = PILImage.fromarray(x)


    elif _getType(x) in ["TENSOR"]:  # PIL & Tensor Implemenataion 
        x = x.numpy() # CHW
        x = np.moveaxis(x, 0, -1) #WHC
        x = np.round(x[:, :, ::-1].copy()*255) #BGR)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, x = cv2.imencode('.jpg', x, encode_param)
        x = cv2.imdecode(x,1)
        
        x = x[:, :, ::-1].copy() / 255 #RGB
        x = np.moveaxis(x, -1, 0) #WHC
        x = torch.tensor(x)

    return x

def _normalize(x, mean, std):

    if _getType(x) in ["PIL"]:  # PIL Implemenataion
        x = _toTensor(x)
        x = vF.normalize(x, mean, std)
        x = _toPIL(x)

    elif _getType(x) in ["TENSOR"]:  # Tensor Implemenataion
        x = vF.normalize(x, mean, std)

    return x 

def _toRGB(x):

    if _getType(x) in ["PIL"]:  # PIL Implemenataion
        c, _, _ = _getSize(x)
        assert c in [1,3,4]

        if c == 1:
            x = x.convert(mode="RGB")
        elif c == 3:
            pass
        elif c == 4:
            x = x.convert(mode="RGB")
        
    elif _getType(x) in ["TENSOR"]:  # Tensor Implemenataion
        c, _, _ = _getSize(x)
        assert c in [1,3,4]

        if c == 1:
            x = torch.cat([x, x, x], 0)
        elif c == 3:
            pass
        elif c == 4:
            x = x[0:3,:,:]
        
    return x 

def _toGrayscale(x):

    if _getType(x) in ["PIL"]:  # PIL Implemenataion
        c, _, _ = _getSize(x)
        assert c in [1,3,4]

        if c == 1:
            pass
        elif c == 3:
            x = x.convert(mode="L")
        elif c == 4:
            x = x.convert(mode="L")
        
    elif _getType(x) in ["TENSOR"]:  # Tensor Implemenataion
        c, _, _ = _getSize(x)
        assert c in [1,3,4]

        #ITU-R 601-2 luma transform
        if c == 1:
            pass
        elif c == 3:
            x = x[0:1, :, :] * 0.299 + x[1:2, :, :] * 0.587 + x[2:3, :, :] * 0.114
        elif c == 4:
            x = x[0:3,:,:]
            x = x[0:1, :, :] * 0.299 + x[1:2, :, :] * 0.587 + x[2:3, :, :] * 0.114
        
    return x 


def _applyImgAug(x, imgAugAugmenter):

    if _getType(x) in ["PIL"]:  # PIL & Tensor Implemenataion
        x = np.asarray(x) #HWC
        x = np.moveaxis(x, 0, 1) #WHC
        x = np.expand_dims(x, 0) #0WHC
        x = imgAugAugmenter(images=x)
        x = np.squeeze(x, 0) #WHC
        x = np.moveaxis(x, 1, 0) #HWC
        x = PILImage.fromarray(x)


    elif _getType(x) in ["TENSOR"]:  # PIL & Tensor Implemenataion 
        x = x.numpy() # CHW
        x = np.moveaxis(x, 0, -1) #WHC
        x = np.expand_dims(x, 0) #0WHC
        x = imgAugAugmenter(images=x)
        x = np.squeeze(x, 0) #WHC
        x = np.moveaxis(x, -1, 0) #CHW
        x = torch.tensor(x)

    return x


def _interpolationMethodToInt(method):

    METHOD_DICT = {
        'NN': 0,
        'NEAREST': 0,
        '0': 0,
        0: 0,

        'BL': 2,
        'BILINEAR': 2,
        '2': 2,
        2: 2,

        'BC': 3,
        'BICUBIC': 3,
        '3': 3,
        3: 3,

        'LANCZOS': 1,
        '1': 1,
        1: 1,

        'RANDOM': random.sample([0, 2, 3], 1)[0],
        'RAND': random.sample([0, 2, 3], 1)[0],
    }

    if isinstance(method, str):
        method = method.upper()
    assert method in METHOD_DICT.keys()

    return METHOD_DICT[method]


def _adjustHue(x, factor):
    assert 0.5 >= factor >= -0.5
    return vF.adjust_hue(x, factor)
    
def _adjustBrightness(x, factor):
    """
    factor 0 : Black
    factor 1 : Original
    factor N : Brightness*=N
    """
    assert factor >= 0
    return vF.adjust_brightness(x, factor)

def _adjustContrast(x, factor):
    """
    factor 0 : Gray
    factor 1 : Original
    factor N : Contrast*=N
    """
    assert factor >= 0
    return vF.adjust_contrast(x, factor)

def _adjustSaturation(x, factor):
    """
    factor 0 : Grayscale
    factor 1 : Original
    factor N : Saturation*=N
    """
    assert factor >= 0
    return vF.adjust_saturation(x, factor)

def _adjustGamma(x, gamma, gain):
    assert gamma >= 0
    assert gain >= 0
    return vF.adjust_gamma(x, gamma, gain)

