'''
model.py
'''
version = '1.60.201230'


#from Python
import time
import csv
import os
import math
import numpy as np
import sys
import functools
from shutil import copyfile

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import _utils
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.autograd import Function

#from this project
from backbone.config import Config
import backbone.vision as vision
    #CovPool
from backbone.module.module import CovpoolLayer, SqrtmLayer, CovpoolLayer3d, SqrtmLayer3d, TriuvecLayer, resBlock, Conv_ReLU_Block
    #Attentions
from backbone.module.module import SecondOrderChannalAttentionBlock, NonLocalBlock, CrissCrossAttention




class DeNIQuA_Res(nn.Module):
    def __init__(self, featureExtractor, CW=64, Blocks=9, inFeature=1, outCW=3, featureCW=1280):
        super(DeNIQuA_Res, self).__init__()

        self.featureExtractor = featureExtractor

        self.CW = CW

        self.inFeature = inFeature

        self.oxo_in = nn.Conv2d( featureCW * inFeature, CW, 1, 1, 0)
        self.res = basicResBlocks(CW=CW, Blocks=9)
        self.oxo_out = nn.Conv2d( CW, outCW, 1, 1, 0)

    def forward(self, xList):

        assert self.inFeature == len(xList)
        
        if self.featureExtractor is not None:
            rstList = []
            for i in range(self.inFeature):
                rstList.append(self.featureExtractor(xList[i]))
            x = torch.cat(rstList, 1)
        else:
            x = torch.cat(xList, 1)

        x = self.oxo_in(x)
        x = self.res(x)
        x = self.oxo_out(x)

        return F.sigmoid(x)



class DeNIQuA(nn.Module):
    def __init__(self, featureExtractor, CW=64, inFeature=1, outCW=3, featureCW=1280):
        super(DeNIQuA, self).__init__()

        self.featureExtractor = featureExtractor

        self.CW = CW

        self.inFeature = inFeature

        self.DecoderList = nn.ModuleList([ # 1/32
            nn.ConvTranspose2d( featureCW * inFeature,   CW*8, 4, 2, 1), #1/16
            nn.ConvTranspose2d( CW*8,  CW*4, 4, 2, 1), #1/8
            nn.ConvTranspose2d( CW*4,  CW*2, 4, 2, 1), #1/4
            nn.ConvTranspose2d( CW*2 , CW*1, 4, 2, 1), #1/2
            nn.ConvTranspose2d( CW*1 , outCW, 4, 2, 1), #1/1
        ])

    def forward(self, xList):

        assert self.inFeature == len(xList)
        
        if self.featureExtractor is not None:
            rstList = []
            for i in range(self.inFeature):
                rstList.append(self.featureExtractor(xList[i]))
            x = torch.cat(rstList, 1)
        else:
            x = torch.cat(xList, 1)

        for i, decoder in enumerate(self.DecoderList):
            x = decoder(x)
            if i + 1 < len(self.DecoderList):    
                x = F.relu(x)

        return F.sigmoid(x)





























class D_class(nn.Module):

    def __init__(self, NDF, NGF):
        super(D_class, self).__init__()

        self.conv1 = nn.Conv2d(NDF * 8, NDF * 4, 4, 2, 1)  # 32->16
        self.conv2 = nn.Conv2d(NDF * 4, NDF * 2, 4, 2, 1)  # 16->8
        self.conv3 = nn.Conv2d(NDF * 2, NDF * 1, 4, 2, 1)  # 8->4
        self.conv4 = nn.Conv2d(NDF * 1, 1, 2, 1, 0)  # 4->1

        self.IN_conv1 = nn.InstanceNorm2d(NGF * 4)
        self.IN_conv2 = nn.InstanceNorm2d(NGF * 2)
        self.IN_conv3 = nn.InstanceNorm2d(NGF * 1)

    def parallelPool(self, x):
        xSize = x.size()

        x = x.contiguous()

        lrX = torch.chunk(x.view(xSize[0], xSize[1], xSize[2], -1, 2), 2, 4)

        lubX = torch.chunk(lrX[0].contiguous().view(xSize[0], xSize[1], xSize[2], -1, 2), 2, 4)
        rubX = torch.chunk(lrX[1].contiguous().view(xSize[0], xSize[1], xSize[2], -1, 2), 2, 4)

        x1 = lubX[0].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))
        x2 = rubX[0].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))
        x3 = lubX[1].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))
        x4 = rubX[1].contiguous().view(xSize[0], xSize[1], xSize[2], round(xSize[3] / 2), round(xSize[4] / 2))

        x = torch.cat((x1, x2, x3, x4), 1)  # (N,C,D,H,W)->(N,C*4,D,H/2,W/2)
        return x

    def forward(self, x):

        x = F.leaky_relu(self.IN_conv1(self.conv1(x)), 0.2)

        x = F.leaky_relu(self.IN_conv2(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.IN_conv3(self.conv3(x)), 0.2)
        
        x = F.sigmoid(self.conv4(x))
        
        return x



class D_AE_class(nn.Module):

    def __init__(self, NDF, NGF):
        super(D_AE_class, self).__init__()

        self.conv1 = nn.Conv2d(3, NDF * 1, 4, 2, 1)  # 32->16
        self.conv2 = nn.Conv2d(NDF * 1, NDF * 2, 4, 2, 1)  # 32->16
        self.conv3 = nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1)  # 32->16
        self.conv4 = nn.Conv2d(NDF * 4, NDF * 4, 4, 2, 1)  # 32->16
        self.conv5 = nn.Conv2d(NDF * 4, NDF * 2, 4, 2, 1)  # 16->8
        self.conv6 = nn.Conv2d(NDF * 2, NDF * 1, 4, 2, 1)  # 8->4
        self.conv7 = nn.Conv2d(NDF * 1, 1, 4, 1, 0)  # 4->1

        self.IN_conv1 = nn.InstanceNorm2d(NGF * 4)
        self.IN_conv2 = nn.InstanceNorm2d(NGF * 2)
        self.IN_conv3 = nn.InstanceNorm2d(NGF * 1)

    def forward(self, x):

        x = F.leaky_relu(self.IN_conv1(self.conv1(x)), 0.2)

        x = F.leaky_relu(self.IN_conv2(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.IN_conv3(self.conv3(x)), 0.2)

        x = F.leaky_relu(self.IN_conv3(self.conv4(x)), 0.2)

        x = F.leaky_relu(self.IN_conv2(self.conv5(x)), 0.2)

        x = F.leaky_relu(self.IN_conv1(self.conv6(x)), 0.2)
        
        x = F.sigmoid(self.conv7(x))
        
        return x

class TiedAE(nn.Module):

    def __init__(self, NDF, NGF):
        super(TiedAE, self).__init__()

        self.conv1 = nn.Conv2d(1, NDF * 1, 4, 2, 1)  # 256->128
        self.conv2 = nn.Conv2d(NDF * 1, NDF * 2, 4, 2, 1)  # 128->64
        self.conv3 = nn.Conv2d(NDF * 2, NDF * 4, 4, 2, 1)  # 64->32
        self.conv4 = nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1)  # 32->16
        self.conv5 = nn.Conv2d(NDF * 4, NDF * 8, 4, 2, 1)  # ->8
        self.conv6 = nn.Conv2d(NDF * 8, NDF * 16, 4, 2, 1)  # ->4
        self.conv7 = nn.Conv2d(NDF * 16, NDF * 32, 4, 1, 0)  # ->1

    def inverse_leaky_relu(self, x, val):
    
        x[x<0] = x[x<0] / val 

        return x
        
    def inverse_tanh(self, x):
        return torch.log((1 + x) / (1 - x)) / 2
        

    def forward(self, x, blur=None, mode="ED"):

        if mode == "ED":

            x = F.leaky_relu(self.conv1(x), 0.2)

            x = F.leaky_relu(self.conv2(x), 0.2)

            x = F.leaky_relu(self.conv3(x), 0.2)

            x = F.leaky_relu(self.conv4(x), 0.2)



            image, blur = x[:,:x.size(1)//2,:,:], x[:,x.size(1)//2:,:,:]

            blur = F.leaky_relu(self.conv5(blur), 0.2)

            blur = F.leaky_relu(self.conv6(blur), 0.2)

            blur = F.tanh(self.conv7(blur))




            blur = F.conv_transpose2d(self.inverse_tanh(blur), self.conv7.weight, stride=1, padding=0)

            blur = F.conv_transpose2d(self.inverse_leaky_relu(blur, 0.2), self.conv6.weight, stride=2, padding=1)

            blur = F.conv_transpose2d(self.inverse_leaky_relu(blur, 0.2), self.conv5.weight, stride=2, padding=1)

            x = torch.cat((image, blur), 1)



            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv4.weight, stride=2, padding=1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv3.weight, stride=2, padding=1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv2.weight, stride=2, padding=1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv1.weight, stride=2, padding=1)
            
            return x

        elif mode == "E":

            x = F.leaky_relu(self.conv1(x), 0.2)

            x = F.leaky_relu(self.conv2(x), 0.2)

            x = F.leaky_relu(self.conv3(x), 0.2)

            x = F.leaky_relu(self.conv4(x), 0.2)

            image, blur = x[:,:x.size(1)//2,:,:], x[:,x.size(1)//2:,:,:]

            blur = F.leaky_relu(self.conv5(blur), 0.2)

            blur = F.leaky_relu(self.conv6(blur), 0.2)

            blur = F.tanh(self.conv7(blur))

            return image, blur

        elif mode == "D":

            blur = F.conv_transpose2d(self.inverse_tanh(blur), self.conv7.weight, stride=1, padding=0)

            blur = F.conv_transpose2d(self.inverse_leaky_relu(blur, 0.2), self.conv6.weight, stride=2, padding=1)

            blur = F.conv_transpose2d(self.inverse_leaky_relu(blur, 0.2), self.conv5.weight, stride=2, padding=1)

            x = torch.cat((x, blur), 1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv4.weight, stride=2, padding=1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv3.weight, stride=2, padding=1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv2.weight, stride=2, padding=1)

            x = F.conv_transpose2d(self.inverse_leaky_relu(x, 0.2), self.conv1.weight, stride=2, padding=1)

            return x




class TiedDisc(nn.Module):
    
    def __init__(self, NDF, NGF):
        super(TiedDisc, self).__init__()

        self.conv1 = nn.Conv2d(        3, NDF *   1, 4, 2, 1) # 256 -> 128
        self.conv2 = nn.Conv2d(NDF *   1, NDF *   2, 4, 2, 1) # 256 -> 128
        self.conv3 = nn.Conv2d(NDF *   2, NDF *   4, 4, 2, 1) # 256 -> 128
        self.conv4 = nn.Conv2d(NDF *   4, NDF *   8, 4, 2, 1) # 256 -> 128
        self.conv5 = nn.Conv2d(NDF *   8, NDF *  16, 4, 2, 1) # 256 -> 128
        self.conv6 = nn.Conv2d(NDF *  16, NDF *  32, 4, 2, 1) # 256 -> 128
        self.conv7 = nn.Conv2d(NDF *  32,         1, 4, 1, 0) # 256 -> 128
        

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)

        x = F.leaky_relu(self.conv3(x), 0.2)

        x = F.leaky_relu(self.conv4(x), 0.2)

        x = F.leaky_relu(self.conv5(x), 0.2)

        x = F.leaky_relu(self.conv6(x), 0.2)

        x = F.sigmoid(self.conv7(x))

        return x


class TiedGAN(nn.Module):

    def __init__(self, NDF, NGF):
        super(TiedGAN, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(NDF * 32, NDF * 16, 4, 2, 1)  #   4   4   256  ->  8   8   128
        self.deconv2 = nn.ConvTranspose2d(NDF * 16, NDF *  8, 4, 2, 1)  #   8   8   128  ->  16  16  64
        self.deconv3 = nn.ConvTranspose2d(NDF *  8, NDF *  4, 4, 2, 1)  #   16  16  64   ->  32  32  32
        self.deconv4 = nn.ConvTranspose2d(NDF *  4, NDF *  2, 4, 2, 1)  #   32  32  32   ->  64  64  16
        self.deconv5 = nn.ConvTranspose2d(NDF *  2, NDF *  1, 4, 2, 1)  #   64  64  16   ->  128 128 8
        self.deconv6 = nn.ConvTranspose2d(NDF *  1,        3, 4, 2, 1)  #   128 128 8    ->  256 256 3

        self.conv1 = nn.Conv2d(        3, NDF *   1, 4, 2, 1) # 256 -> 128
        self.conv2 = nn.Conv2d(NDF *   1, NDF *   2, 4, 2, 1) # 256 -> 128
        self.conv3 = nn.Conv2d(NDF *   2, NDF *   4, 4, 2, 1) # 256 -> 128
        self.conv4 = nn.Conv2d(NDF *   4, NDF *   8, 4, 2, 1) # 256 -> 128
        self.conv5 = nn.Conv2d(NDF *   8, NDF *  16, 4, 2, 1) # 256 -> 128
        self.conv6 = nn.Conv2d(NDF *  16, NDF *  32, 4, 2, 1) # 256 -> 128
        self.conv7 = nn.Conv2d(NDF *  32,         1, 4, 1, 0) # 256 -> 128

    def inverse_leaky_relu(self, x, val):
    
        x[x<0] = x[x<0] / val 

        return x
        
    def inverse_tanh(self, x):
        return torch.log((1 + x) / (1 - x)) / 2
        

    def forward(self, x, mode):

        if mode == "Dc":

            x = F.leaky_relu(self.deconv1(x), 0.2)

            x = F.leaky_relu(self.deconv2(x), 0.2)

            x = F.leaky_relu(self.deconv3(x), 0.2)

            x = F.leaky_relu(self.deconv4(x), 0.2)

            x = F.leaky_relu(self.deconv5(x), 0.2)

            x = F.leaky_relu(self.deconv6(x), 0.2)
            
            return x

        elif mode == "Ec":

            x = F.conv2d(self.inverse_leaky_relu(x, 0.2), self.deconv6.weight, stride=2, padding=1)

            x = F.conv2d(self.inverse_leaky_relu(x, 0.2), self.deconv5.weight, stride=2, padding=1)

            x = F.conv2d(self.inverse_leaky_relu(x, 0.2), self.deconv4.weight, stride=2, padding=1)

            x = F.conv2d(self.inverse_leaky_relu(x, 0.2), self.deconv3.weight, stride=2, padding=1)

            x = F.conv2d(self.inverse_leaky_relu(x, 0.2), self.deconv2.weight, stride=2, padding=1)

            x = F.conv2d(self.inverse_leaky_relu(x, 0.2), self.deconv1.weight, stride=2, padding=1)

            return x



