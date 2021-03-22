'''
vision.py
'''
version = '1.91.201008'

#from Python
import time
import csv
import os
import math
import numpy as np
import sys
import random
from shutil import copyfile

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

#from Image Library
import cv2
from PIL import Image, ImageDraw


#from this project
from backbone.config import Config
import backbone.utils as utils

eps = 1e-6 if Config.param.train.method.mixedPrecision == False else 1e-4
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])



def E2EBlending(input1, input2, IQAMap, superPixelMap, softMode = False):

    assert input1.dim() == 4 and input2.dim() == 4, f'vision.py :: tensor dim must be 4 (current: {input1.dim()}, {input2.dim()})'
    
    if superPixelMap is not None:
        scoreMap = torch.zeros_like(IQAMap)
        for i in range(torch.max(superPixelMap)):
            meanScoreInGivenArea = torch.sum(IQAMap * (superPixelMap == i), dim=(1,2,3), keepdim=True) / (torch.sum(superPixelMap == i, dim=(1,2,3), keepdim=True) + eps)
            scoreMap += meanScoreInGivenArea * (superPixelMap == i)
    else:
        scoreMap = IQAMap


    scoreMap = scoreMap if softMode is True else torch.round(scoreMap)
    rst = input1 * scoreMap + input2 * (1 - scoreMap)

    return rst, scoreMap


def Laplacian(input,ksize):

    if ksize==1:
        kernel = torch.tensor(  [0,1,0],
                                [1,-4,1],
                                [0,1,0])
    else:
        kernel = torch.ones(ksize,ksize)
        kernel[int(ksize/2),int(ksize/2)] = 1 - ksize**2

    kernel = Variable(kernel.view(1,1,1,ksize,ksize))
    output = F.conv3d(input.view(input.size()[0],1,3,input.size()[2],input.size()[3]),kernel,padding = [0, int(ksize/2), int(ksize/2)]).view(input.size()[0],-1,input.size()[2],input.size()[3])
    
    return output


def Gaussian(input,ksize,sigma):

    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    kernel = torch.from_numpy(kernel).float().to(input.device).view(1,1,1,ksize,ksize)
    output = F.conv3d(input.view(input.size()[0],1,input.size()[1],input.size()[2],input.size()[3]),kernel,padding = [0, int(ksize/2), int(ksize/2)]).view(input.size()[0],-1,input.size()[2],input.size()[3])
    
    return output


def gaussianKernel(ksize, sigma,isdouble=False):
    assert ksize % 2 == 1
    ax = np.arange(-ksize // 2 + 1., ksize // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = torch.from_numpy(kernel).float() if isdouble is False else torch.from_numpy(kernel).double()
    return kernel

def HLCombine(inputH,inputL,kernel):
    H_mag,H_pha = polarFFT(inputH)
    L_mag,L_pha = polarFFT(inputL)
    '''
    ax = np.arange(-inputH.size(2) // 2 + 1., inputH.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    
    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,inputH.size(2),inputH.size(3))
    '''
    kernel = kernel.expand(inputH.size(0),inputH.size(1),inputH.size(2),inputH.size(3))
    #kernel = kernel / torch.max(kernel)
    
    H_mag = torch.cat((H_mag[:,:,int(inputH.size(2)/2):,:],H_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    H_mag = torch.cat((H_mag[:,:,:,int(inputH.size(2)/2):],H_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    L_mag = torch.cat((L_mag[:,:,int(inputH.size(2)/2):,:],L_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    L_mag = torch.cat((L_mag[:,:,:,int(inputH.size(2)/2):],L_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    H_mag = H_mag * (1-kernel)
    L_mag = L_mag * kernel

    H_mag = torch.cat((H_mag[:,:,int(inputH.size(2)/2):,:],H_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    H_mag = torch.cat((H_mag[:,:,:,int(inputH.size(2)/2):],H_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    L_mag = torch.cat((L_mag[:,:,int(inputH.size(2)/2):,:],L_mag[:,:,0:int(inputH.size(2)/2),:]),2)
    L_mag = torch.cat((L_mag[:,:,:,int(inputH.size(2)/2):],L_mag[:,:,:,0:int(inputH.size(2)/2)]),3)

    H_f_r = H_mag * torch.cos(H_pha)
    H_f_i = H_mag * torch.sin(H_pha)
    H_f = torch.cat((H_f_r.view(H_f_r.size(0),H_f_r.size(1),H_f_r.size(2),H_f_r.size(3),1),H_f_i.view(H_f_i.size(0),H_f_i.size(1),H_f_i.size(2),H_f_i.size(3),1)),4)

    L_f_r = L_mag * torch.cos(L_pha)
    L_f_i = L_mag * torch.sin(L_pha)
    L_f = torch.cat((L_f_r.view(L_f_r.size(0),L_f_r.size(1),L_f_r.size(2),L_f_r.size(3),1),L_f_i.view(L_f_i.size(0),L_f_i.size(1),L_f_i.size(2),L_f_i.size(3),1)),4)

    return torch.irfft(H_f + L_f, 2, onesided=False)
    
    

def freqMagRearrangement(magnitude):
    half = [magnitude.size(-2)//2, magnitude.size(-1)//2]
    magnitude = torch.cat((magnitude[:,:,half[0]:,:],magnitude[:,:,0:half[0],:]),-2)
    magnitude = torch.cat((magnitude[:,:,:,half[1]:],magnitude[:,:,:,0:half[1]]),-1)
    return magnitude


def freqMagToInterpretable(magnitude, rearrange = True):
    if rearrange == True: 
        mag_f_a_view = freqMagRearrangement(magnitude)
    mag_f_a_view = torch.log(mag_f_a_view)
    mag_f_a_view = (mag_f_a_view - mag_f_a_view.min()) / (mag_f_a_view.max() - mag_f_a_view.min())
    return mag_f_a_view


def polarIFFT(x_mag, x_pha):
    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)

    if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
        x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)
        output = torch.irfft(x_f,2,onesided=False)
    else:
        x_f = torch.complex(x_f_r,x_f_i)
        output = torch.fft.ifft2(x_f)

    return output


def polarFFT(input):
    if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
        x_f = torch.rfft(input,2,onesided=False)

        x_f_r = x_f[:,:,:,:,0].contiguous()
        x_f_i = x_f[:,:,:,:,1].contiguous()

        x_f_r[x_f_r==0] = eps

        x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
        x_pha = torch.atan2(x_f_i,x_f_r)

    else:
        x_f = torch.fft.fft2(input)
        x_mag = x_f.abs()
        x_pha = x_f.angle()

    return x_mag, x_pha
    

    
def HPFinFreq(input,sigma,freqReturn=False):

    half = [input.size(2)//2, input.size(3)//2]

    ax = np.arange(-input.size(2) // 2 + 1., input.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,input.size(2),input.size(3))
    kernel = kernel.expand(input.size(0),input.size(1),input.size(2),input.size(3))
    kernel = kernel / torch.max(kernel)
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
    x_pha = torch.atan2(x_f_i,x_f_r)

    x_mag = torch.cat((x_mag[:,:,half[0]:,:],x_mag[:,:,0:half[0],:]),2)
    x_mag = torch.cat((x_mag[:,:,:,half[1]:],x_mag[:,:,:,0:half[1]]),3)

    #x_pha = torch.cat((x_pha[:,:,128:,:],x_pha[:,:,0:128,:]),2)
    #x_pha = torch.cat((x_pha[:,:,:,128:],x_pha[:,:,:,0:128]),3)
    
    x_mag = x_mag * (1-kernel)
    #x_pha = x_pha * (1-kernel)

    x_mag = torch.cat((x_mag[:,:,half[0]:,:],x_mag[:,:,0:half[0],:]),2)
    x_mag = torch.cat((x_mag[:,:,:,half[1]:],x_mag[:,:,:,0:half[1]]),3)

    #x_pha = torch.cat((x_pha[:,:,128:,:],x_pha[:,:,0:128,:]),2)
    #x_pha = torch.cat((x_pha[:,:,:,128:],x_pha[:,:,:,0:128]),3)
    
    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)

    output = x_mag,x_pha

    if(freqReturn == False):
        output = torch.irfft(x_f,2,onesided=False)
    
    return output

def LPFinFreq(input,sigma,freqReturn=False):
    ax = np.arange(-input.size(2) // 2 + 1., input.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,input.size(2),input.size(3))
    kernel = kernel.expand(input.size(0),input.size(1),input.size(2),input.size(3))
    kernel = kernel / torch.max(kernel)
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
    x_pha = torch.atan2(x_f_i,x_f_r)

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)
    
    x_mag = x_mag * kernel

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)
    
    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)

    output = x_mag,x_pha

    if(freqReturn == False):
        output = torch.irfft(x_f,2,onesided=False)
    
    return output

def InverseLPFinFreq(input,sigma):
    ax = np.arange(-input.size(2) // 2 + 1., input.size(3) // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)

    kernel = Variable((torch.from_numpy(kernel).float()).cuda()).view(1,1,input.size(2),input.size(3))
    kernel = kernel.expand(input.size(0),input.size(1),input.size(2),input.size(3))
    kernel = 1/(kernel / torch.max(kernel))
    x_f = torch.rfft(input,2,onesided=False)
    x_f_r = x_f[:,:,:,:,0].contiguous()
    x_f_i = x_f[:,:,:,:,1].contiguous()
    x_mag = torch.sqrt(x_f_r*x_f_r + x_f_i*x_f_i)
    x_pha = torch.atan2(x_f_i,x_f_r)

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)
    
    x_mag = x_mag * kernel

    x_mag = torch.cat((x_mag[:,:,128:,:],x_mag[:,:,0:128,:]),2)
    x_mag = torch.cat((x_mag[:,:,:,128:],x_mag[:,:,:,0:128]),3)

    x_f_r = x_mag * torch.cos(x_pha)
    x_f_i = x_mag * torch.sin(x_pha)
    x_f = torch.cat((x_f_r.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1),x_f_i.view(x_f_r.size(0),x_f_r.size(1),x_f_r.size(2),x_f_r.size(3),1)),4)
    
    output = torch.irfft(x_f,2,onesided=False)
    
    return output

def RGB2HSI(input): #c0:Hue(0~1) c1:Sat(0~1) c2:Value(0-1)
    input = (1-Nonzero(input))*eps + input
    
    R = torch.squeeze(input[:,0,:,:])
    G = torch.squeeze(input[:,1,:,:])
    B = torch.squeeze(input[:,2,:,:])

    Allsame = (1 - Nonzero(R - G)) * (1 - Nonzero(R - B))
    BGsame = (1 - Nonzero(G - B))


    nBgtG = (1 - Allsame) * Nonzero(Max((B - G),torch.zeros(G.size()).cuda())) + BGsame
    nGgtB = (1 - Allsame) * Nonzero(Max((G - B),torch.zeros(G.size()).cuda()))

    
    BgtG = B * nBgtG  # element of B that <G was replaced by 0
    GltB = G * nBgtG
    
    GgtB = G * nGgtB # element of G that <B was replaced by 0
    BltG = B * nGgtB

    H_GgtB = torch.acos((R-GgtB+R-BltG)/(2*torch.sqrt(eps + (R-GgtB)*(R-GgtB)+(R-BltG)*(GgtB-BltG)))) * nGgtB
    H_BgtG = (2*math.pi - torch.acos(torch.clamp((R-GltB+R-BgtG)/(2*torch.sqrt(eps + (R-GltB)*(R-GltB)+(R-BgtG)*(GltB-BgtG))),-1+eps,1-eps))) * nBgtG
    
    H = (H_GgtB + H_BgtG) / (2*math.pi)
    I = (R+G+B)/3
    S = 1-3*torch.min(input,1)[0]/(R+G+B)
    

    output = torch.cat((H.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        S.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        I.view(input.size()[0],1,input.size()[2],input.size()[3])),1)
    
    return output

def HSI2RGB(input):     

    H = torch.squeeze(input[:,0,:,:]) * 6
    S = torch.squeeze(input[:,1,:,:])
    I = torch.squeeze(input[:,2,:,:])

    
    '''
    Z = 1 - torch.abs(torch.fmod(H,2) - 1)
    C = 3*I*S / (I+Z)
    X = C*Z
    '''
    
    H02 = FilterByValue(H,0,2) / H
    H24 = FilterByValue(H,2,4) / H
    H46 = FilterByValue(H,4,6.01) / H

    H = H/6

    b = (H02 * ((1 - S) / 3) +
        H24 * ((1 - ((1 - S) / 3) - (1+S*torch.cos((H-1/3)*2*math.pi)/(torch.cos((1/6 - (H-1/3))*2*math.pi)+eps))/3)) +
        H46 * ((1+S*torch.cos((H-2/3)*2*math.pi)/(torch.cos((1/6 - (H-2/3))*2*math.pi)+eps))/3))
    
    r = (H24 * ((1 - S) / 3) +
        H46 * ((1 - ((1 - S) / 3) - (1+S*torch.cos((H-2/3)*2*math.pi)/(torch.cos((1/6 - (H-2/3))*2*math.pi)+eps))/3)) +
        H02 * ((1+S*torch.cos(H*2*math.pi)/(torch.cos((1/6 - H)*2*math.pi)+eps))/3))

    g = (H46 * ((1 - S) / 3) +
        H02 * ((1 - ((1 - S) / 3) - (1+S*torch.cos(H*2*math.pi)/(torch.cos((1/6 - H)*2*math.pi)+eps))/3)) +
        H24 * ( ( 1 + S*torch.cos((H-1/3)*2*math.pi) / (torch.cos((1/6 - (H-1/3))*2*math.pi)+eps))/3))

    
    R = 3*I*r
    G = 3*I*g
    B = 3*I*b

    output = torch.cat((R.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        G.view(input.size()[0],1,input.size()[2],input.size()[3]),
                        B.view(input.size()[0],1,input.size()[2],input.size()[3])),1)
    
    return output

def gaussianKernelSpray(height, width, kernelMinCount, kernelMaxCount, rois):
    # rois -> 카우시안 커널의 기본 area을 설정 
    # rois = 0 -> 해당 기능 사용 X : np.size(rois) <= 1
    # rois = [left x, left y, width, height, ...] -> 해당 기능 사용 O : np.size(rois) > 1
    
    chikChik = random.randint(kernelMinCount, kernelMaxCount)
    dowhajee = np.zeros((height, width))
    sigmaScale = (4.3, 4.1) # sigma 기준 정립 필요 %5.0 4.8
    
    for i in range(chikChik):        
        scale_x = 1
        scale_y = 1
    
        if np.size(rois) <= 1:
            centerY = random.randint(0, height - 1)
            centerX = random.randint(0, width - 1)
            longLen = max([height, width])
            sigma = random.uniform(longLen / 50, longLen / 5)
        elif np.size(rois) > 1:
            x = rois[i][0]
            y = rois[i][1]
            x_width = rois[i][2]
            y_height = rois[i][3]

            centerY = int(y + y_height/2)
            centerX = int(x + x_width/2)
            #-4σ ~ +4σ의 범위 값은 전체 분포의 약 99.99%를 차지하기 때문에 적절한 커널 크기로 σ의 4배 * 2배가 되어야 함
            sigma = random.uniform((x_width +y_height)/sigmaScale[0], (x_width +y_height)/sigmaScale[1]) 
            # rois에 따른 scale_factor 기준 정립 필요
            if y_height/x_width > x_width/y_height:
                scale_factor = y_height/x_width
                scale_x = scale_x * scale_factor
                scale_y = scale_y * 1
            elif y_height/x_width < x_width/y_height:
                scale_factor = x_width/y_height
                scale_x = scale_x * 1
                scale_y = scale_y * scale_factor

        ax = np.arange(0 - centerX, width - centerX)
        ay = np.arange(0 - centerY, height - centerY)

        ## x에 배수하면 가우시안 커널의 좌우 너비 조정, y에 배수하면 가우시안 커널의 상하 너비 조정 
        xx, yy = np.meshgrid(ax*scale_x, ay*scale_y)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        dowhajee = np.maximum(dowhajee, kernel)    

    dowhajee = Variable((torch.from_numpy(dowhajee).float()).cuda()).view(1,1,height,width)
    return dowhajee
    #output = F.conv3d(input.view(input.size()[0],1,3,input.size()[2],input.size()[3]),kernel,padding = [0, int(ksize/2), int(ksize/2)]).view(input.size()[0],-1,input.size()[2],input.size()[3])

def rectangleKernelSpray(height, width, kernelMinCount, kernelMaxCount, rois):
    # rois -> rectangle 커널의 기본 area을 설정 
    # rois = [left x, left y, width, height, ...] -> 해당 기능 사용 O : np.size(rois) > 1
    
    chikChik = random.randint(kernelMinCount, kernelMaxCount)
    dowhajee = np.zeros((height, width))
    
    for i in range(chikChik):        
        x = rois[i][0]
        y = rois[i][1]
        x_width = rois[i][2]
        y_height = rois[i][3]

        percent = 3
        kernelWidth = int(x_width + (x_width/10)*2*percent)
        kernelHeight = int(y_height + (y_height/10)*2*percent)

        expandedkernel = np.zeros((kernelHeight, kernelWidth))
        rectangularKernel = np.zeros((kernelHeight, kernelWidth))

        x_ratio = x_width/(x_width + y_height)
        y_ratio = y_height/(x_width + y_height)

        compoundXY = x_width * x_ratio + y_height * y_ratio
        scaling = int((compoundXY/10.0)*percent)

        for i in range(scaling):
            scaledValue = i/scaling
            rectangularKernel[i:kernelHeight-i, i:kernelWidth-i] = scaledValue
        
        start_height = int((kernelHeight - y_height)/2)
        start_width = int((kernelWidth - x_width)/2)
        expandedkernel = np.maximum(expandedkernel, rectangularKernel)
        dowhajee[y-start_height:y-start_height+kernelHeight, x-start_width:x-start_width+kernelWidth] = expandedkernel

    dowhajee = Variable((torch.from_numpy(dowhajee).float()).cuda()).view(1,1,height,width)
    return dowhajee


def BlendingMethod(blending_method, source, destination, rois, colorMode):
    # 차후 불 필요한 형변환 수정 필요 (pil,cv2,pytorch tensor 사이의 형 변환)
    blending_method_list = ['simpleBlending', 'gaussianBlending', 'rectangleBlending', 'poissonBlending']

    if blending_method not in blending_method_list:
        raise ValueError("blending mothod name error")

    method = blending_method
    width, height = (destination.shape[3], destination.shape[2]) # width, height (torch image기준 -> 수, 채널, 높이, 너비) 
    numberOfRoi = int(np.size(rois)/4)
    center = (int(destination.shape[3]/2), int(destination.shape[2]/2)) # width/2, height/2
    blended_img = Image.new(mode='RGB', size=(width, height), color ='black')

    if method == "simpleBlending":
        print("simpleBlending")
        destination_temp = destination.clone()
        for i in range(0, numberOfRoi):
            x = rois[i][0]
            y = rois[i][1]
            width = rois[i][2]
            height = rois[i][3]
                        
            cropped_image = source[:,:,y:y+height, x:x+width] 
            destination_copy[:,:,y:y+height, x:x+width] = cropped_image
        blended_img = destination_copy
    elif method == "gaussianBlending":
        print("gaussianBlending mothod")
        # create gaussian map 
        gaussian = gaussianKernelSpray(destination.shape[2], destination.shape[3], numberOfRoi, numberOfRoi, rois).repeat(1,3,1,1)
        gaussianSprayKernel = gaussian
        #save_image(gaussianSprayKernel, "/home/projSR/dataset/CUSTOM/detectionTest/gaussian.png")
        # create blending image
        gaussianback = destination * gaussianSprayKernel + source * (1 - gaussianSprayKernel)
        gaussianfore = source * gaussianSprayKernel + destination * (1 - gaussianSprayKernel)

        blended_img = gaussianfore
    elif method == "rectangleBlending":
        print("rectangleBlending mothod")
        # create rectangle map 
        rectangle = rectangleKernelSpray(destination.shape[2], destination.shape[3], numberOfRoi, numberOfRoi, rois).repeat(1,3,1,1)
        rectangleSprayKernel = rectangle
        #save_image(rectangleSprayKernel, "/home/projSR/dataset/CUSTOM/detectionTest/rectangle.png")
        # create blending image
        rectangleback = destination * rectangleSprayKernel + source * (1 - rectangleSprayKernel)
        rectanglefore = source * rectangleSprayKernel + destination * (1 - rectangleSprayKernel)

        blended_img = rectanglefore
    elif method == "poissonBlending":
        print("poissonBlending mothod")
        # pytorch tensor to cv2
        destination = utils.denorm(destination.cpu().view(destination.size(0), 1 if colorMode=='grayscale' else 3, destination.size(2), destination.size(3))) 
        source = utils.denorm(source.cpu().view(source.size(0), 1 if colorMode=='grayscale' else 3, source.size(2), source.size(3)))
        
        destination_cv2 = destination.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
        destination_cv2 = cv2.cvtColor(destination_cv2, cv2.COLOR_RGB2BGR).astype(np.uint8)
        source_cv2 = source.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
        source_cv2 = cv2.cvtColor(source_cv2, cv2.COLOR_RGB2BGR).astype(np.uint8)
        
        source_cv2_temp = source_cv2.copy()
        destination_cv2_temp = destination_cv2.copy()

        mixed_clone = destination_cv2_temp
        for i in range(0, numberOfRoi):
            x = rois[i][0]
            y = rois[i][1]
            width = rois[i][2]
            height = rois[i][3]

            cropped_image = source_cv2_temp[y:y+height, x:x+width, :]
            area = (cropped_image.shape[0], cropped_image.shape[1], 3)            
            a = int(((x+x+width)/2))
            b = int(((y+y+height)/2))
            center = (a, b)

            mask_cv2 = 255 * np.ones(area, dtype=np.uint8)
            source_mask_cv2 = cropped_image
            destination_cv2 = mixed_clone
            #cv2.imwrite('/home/projSR/dgk/git/sr-research-framework/data/30-EndtoEntDeepBlending/result/2-shiftmoduletrainassssment/mask_cv2.png',mask_cv2)
            #cv2.imwrite('/home/projSR/dgk/git/sr-research-framework/data/30-EndtoEntDeepBlending/result/2-shiftmoduletrainassssment/source_mask_cv2.png',source_mask_cv2)
            #cv2.imwrite('/home/projSR/dgk/git/sr-research-framework/data/30-EndtoEntDeepBlending/result/2-shiftmoduletrainassssment/destination_cv2.png',destination_cv2)
            mixed_clone = cv2.seamlessClone(source_mask_cv2, destination_cv2, mask_cv2, center, cv2.MIXED_CLONE) # 수정사항 존재(Framework 내부)
            #cv2.imwrite('/home/projSR/dgk/git/sr-research-framework/data/30-EndtoEntDeepBlending/result/2-shiftmoduletrainassssment/1.png',mixed_clone)

        mixed_clone = cv2.cvtColor(mixed_clone, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        mixed_clone = torch.from_numpy(mixed_clone.transpose(2, 0, 1)).unsqueeze_(0)
        blended_img = mixed_clone

    return blended_img
 


