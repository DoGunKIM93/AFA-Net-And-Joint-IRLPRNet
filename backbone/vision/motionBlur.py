
import numpy as np
import random
import time
import gc

import torch
import torch.nn.functional as F

from . import vision

from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve


EPS = 1e-12
PI = pi


def _softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _norm(lst: list) -> float:
    """[summary]
    L^2 norm of a list
    [description]
    Used for internals
    Arguments:
        lst {list} -- vector
    """
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def _polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]
    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle
    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)

def _polar2zTensor(r: torch.tensor, θ: torch.tensor) -> torch.tensor:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]
    Arguments:
        r {torch.tensor} -- radius
        θ {torch.tensor} -- angle
    Returns:
        [torch.tensor] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * torch.exp(1j * θ)

def _diagonal(y, x):
    return ( x **2 + y **2) **0.5

def _getSteps(kW, kH, intensity, numSteps = None):

    diagonal = _diagonal(kW, kH)
    maxPathLen = 0.75 * diagonal * (uniform() + uniform(0, intensity**2))
    maxStep = int(diagonal)

    # getting step
    steps = []

    if numSteps is None:
        while sum(steps) < maxPathLen and len(steps) < diagonal * 2:

            # sample next step
            step = beta(1, 30) * (1 - intensity + EPS) * diagonal
            if step < maxPathLen:
                steps.append(step)

        numSteps = len(steps)

    else:
        while len(steps) < numSteps:

            # sample next step
            step = beta(1, 30) * (1 - intensity + EPS) * diagonal
            if step < maxPathLen:
                steps.append(step)

        assert len(steps) == numSteps


    # note the steps and the total number of steps

    
    steps = np.asarray(steps)
    
    return numSteps, steps    


def _getAngles(intensity, numSteps):
    # same as with the steps
    
    # first we get the max angle in radians
    maxAngle = uniform(0, intensity * PI)

    # now we sample "jitter" which is the probability that the
    # next angle has a different sign than the previous one
    jitter = beta(2, 20)

    # initialising angles (and sign of angle)
    angles = [uniform(low=-maxAngle, high=maxAngle)]

    while len(angles) < numSteps:
        np.random.seed(int(time.time()%100*10000))
        # sample next angle (absolute value)
        angle = triangular(0, intensity *
                            maxAngle, maxAngle + EPS)

        # with jitter probability change sign wrt previous angle
        if uniform() < jitter:
            angle *= - np.sign(angles[-1])
        else:
            angle *= np.sign(angles[-1])

        angles.append(angle)

    # save angles
    angles = np.asarray(angles)

    return angles






def motionBlurKernel(kernelSize, intensityRange, superSampleScale = 1, device='cuda') -> np.ndarray:
    
    return fourPointsNonUniformMotionBlurKernel((1,1), kernelSize, intensityRange, superSampleScale, device).squeeze(3).squeeze(3)





def fourPointsNonUniformMotionBlurKernel(mapSize, kernelSize, intensityRange, superSampleScale = 1, device='cuda') -> torch.Tensor:
    
    assert isinstance(kernelSize, tuple)
    assert isinstance(mapSize, tuple)
    assert isinstance(intensityRange, tuple)
    assert 1 >= intensityRange[0] >= 0 
    assert 1 >= intensityRange[1] >= 0 
    assert intensityRange[1] >= intensityRange[0]


    DEVICE = device

    NUM_POINT = 4
    mH, mW = mapSize
    kH, kW = kernelSize
    kWSS = kernelSize[0] * superSampleScale
    kHSS = kernelSize[1] * superSampleScale

    stepsList = []
    anglesList = []
    numSteps = None

    for i in range(NUM_POINT):
        intensity = random.uniform(intensityRange[0], intensityRange[1])
        # Get steps and angles
        numSteps, steps = _getSteps(kWSS, kHSS, intensity, numSteps)
        angles = _getAngles(intensity, numSteps)
        stepsList.append(steps)
        anglesList.append(angles)
        #print(angles)

    #make interpolated 2-d kernel map
    stepTnsr = torch.tensor(stepsList, dtype=torch.float32, device=DEVICE).permute(1,0).unsqueeze(1)
    angleTnsr = torch.tensor(anglesList, dtype=torch.float32, device=DEVICE).permute(1,0).unsqueeze(1)
    #print(angleTnsr.size())
    #stepTnsr = torch.cat([torch.tensor(steps).view(-1,1,1) for steps in stepsList], -1) # STEPS, POINTS
    stepTnsr = stepTnsr.view(numSteps, 1, 2, 2) #STEPS, 2, 2
    #angleTnsr = torch.cat([torch.tensor(angles).view(-1,1,1) for angles in anglesList], -1) # STEPS, POINTS
    angleTnsr = angleTnsr.view(numSteps, 1, 2, 2) #STEPS, 2, 2
    #print(angleTnsr)
    stepTnsr = F.interpolate(stepTnsr, (mH, mW), mode='bilinear').squeeze(1).permute(1,2,0)
    angleTnsr = F.interpolate(angleTnsr, (mH, mW), mode='bilinear').squeeze(1).permute(1,2,0)
    
    #make actual kernels
    kernelImageList = []
    kernelList = []

    # we turn angles and steps into complex numbers
    complexIncrements = _polar2zTensor(stepTnsr, angleTnsr)
    # generate path as the cumsum of these increments
    pathComplex = torch.cumsum(complexIncrements, dim=2)
    # find center of mass of path
    comComplex = torch.sum(pathComplex, dim=2, keepdim=True) / numSteps
    # Shift path s.t. center of mass lies in the middle of
    # the kernel and a apply a random rotation
    ###

    # center it on COM
    centerOfKernel = (kHSS + 1j * kWSS) / 2
    pathComplex -= comComplex
    # randomly rotate path by an angle a in (0, PI)
    pathComplex *= np.exp(1j * uniform(0, PI))
    # center COM on center of kernel
    pathComplex += centerOfKernel
    # calc. diagonal length
    diagonal = _diagonal(kWSS, kHSS)

    # coord list tensors to torch.sparse_coo_tensors
    wT = torch.tensor(range(mW), dtype=torch.int16, device=DEVICE).view(-1,1).repeat(1,mH).view(-1)
    wT = wT.view(-1,1).repeat(1, numSteps).view(-1)
    hT = torch.tensor(range(mH), dtype=torch.int16, device=DEVICE).repeat(mW)
    hT = hT.view(-1,1).repeat(1, numSteps).view(-1)
    yT = pathComplex.real.short().clamp(0,kHSS-1).view(-1)
    xT = pathComplex.imag.short().clamp(0,kWSS-1).view(-1)
    coordTensor = torch.stack((hT, wT, yT, xT), 0)
    valueTensor = torch.ones_like(xT, dtype=torch.float32)#*255

    pathTensor = torch.sparse_coo_tensor(
                coordTensor, #i
                valueTensor, #v
                [mH, mW, kHSS, kWSS]
            ).coalesce()

    kernelTensor = pathTensor.to_dense()

    del coordTensor
    del valueTensor
    del pathTensor
    gc.collect()

    #sigma = rad ** 2 / 3
    kernelTensor = kernelTensor.view(-1,kHSS,kWSS).unsqueeze(1)

    kernelTensor = F.interpolate(kernelTensor, (kH, kW), mode='bicubic') if superSampleScale != 1 else kernelTensor
    kernelTensor = kernelTensor.clamp(0,1)

    kernelTensor = vision.Gaussian(kernelTensor, 3, 1.0)

    kernelTensor = kernelTensor / kernelTensor.sum(dim=(2,3), keepdim=True)

    kernelTensor = kernelTensor.squeeze(1).view(mH,mW,kH,kW)
    
    #store kernel example
    '''
    tt = time.perf_counter()
    for h in range(mH):
        kernelImageList_t = []
        for w in range(mW):
            kT_t = kernelTensor[h,w,:,:].cpu()
            kernelImage = aug._toPIL(kT_t.view(1,*kT_t.size()) * diagonal)

            kernelImageList_t.append(kernelImage)
            
        kernelImage_t = Image.new('L', (kW * mW, kH))
        x_offset = 0
        for im in kernelImageList_t:
            kernelImage_t.paste(im, (x_offset,0))
            x_offset += im.size[0]
        kernelImageList.append(kernelImage_t)

    kernelImage = Image.new('L', (kW * mW, kH * mH))
    y_offset = 0
    for im in kernelImageList:
        kernelImage.paste(im, (0,y_offset))
        y_offset += im.size[1]
    print(time.perf_counter() - tt)

    kernelImage.save("kernel.png")
    '''
    
    kernel = kernelTensor.view(1,1,1,*kernelTensor.size())#torch.stack(kernelList)
    return kernel




