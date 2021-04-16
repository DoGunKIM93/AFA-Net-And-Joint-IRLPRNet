"""
model.py
"""


# from Python
import time
import csv
import os
import math
import numpy as np
import sys
import functools
import gc
from shutil import copyfile
from fast_pytorch_kmeans import KMeans

# from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.models import _utils
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.autograd import Function

# from this project
from backbone.config import Config
import backbone.vision as vision
import backbone.module.module as module

# CovPool
from backbone.module.module import (
    CovpoolLayer,
    SqrtmLayer,
    CovpoolLayer3d,
    SqrtmLayer3d,
    TriuvecLayer,
    resBlock,
    Conv_ReLU_Block,
)

# Attentions
from backbone.module.module import SecondOrderChannalAttentionBlock, NonLocalBlock, CrissCrossAttention



class empty(nn.Module):
    def __init__(self):
        super(empty, self).__init__()        
    
    def forward(self, x):
        return x

class IronBoy(nn.Module):
    def __init__(self, CW=16, numBlocks=9): #scale=(2,3,4)
        super(IronBoy, self).__init__()
        self.CW = CW
        self.numBlocks = numBlocks



        #self.blurryImage_inLayer = module.resBlock(CW, 3, inputCD = 3, outAct = nn.ReLU())
        self.blurryImage_inLayer = nn.Sequential(*[nn.Conv2d(3, CW, 3, 1, 1), nn.ReLU(inplace=True),])

        #self.blurryImage_res = nn.Sequential(*[module.resBlock(CW, 3, outAct=nn.ReLU()) for _ in range(numBlocks)])
        # self.blurryImage_res = nn.Sequential(*[rcab_block(CW, 3) for _ in range(numBlocks)])

        # self.blurryImage_outLayer = nn.Sequential(*[nn.Conv2d(CW, CW, 3, 1, 1), ])



        #self.kernel_inLayer = module.resBlock(CW, 3, inputCD = 3, outAct = nn.ReLU())
        self.kernel_inLayer = nn.Sequential(*[nn.Conv2d(3, CW, 4, 2, 1), nn.ReLU(inplace=True),
                                              nn.Conv2d(CW, CW*2, 4, 2, 1), nn.ReLU(inplace=True)])

        #self.kernel_res = nn.Sequential(*[module.resBlock(CW, 3, outAct=nn.ReLU()) for _ in range(numBlocks)])
        self.kernel_res = nn.Sequential(*[rcab_block(CW*2, 3) for _ in range(numBlocks)])
        
        #self.kernel_outLayer = nn.Sequential(*[nn.Conv2d(CW, CW, 3, 1, 1), ])
        self.kernel_outLayer = nn.Sequential(*[nn.ConvTranspose2d(CW*2, CW, 4, 2, 1), nn.ReLU(inplace=True),
                                               nn.ConvTranspose2d(CW, CW, 4, 2, 1),])



        # self.estimatedSharpImage_inLayer = module.resBlock(CW, 3, outAct = nn.ReLU())
        #self.estimatedSharpImage_res = nn.Sequential(*[module.resBlock(CW, 3, outAct=nn.ReLU()) for _ in range(numBlocks)])
        # self.estimatedSharpImage_res = nn.Sequential(*[rcab_block(CW, 3) for _ in range(numBlocks)])
        self.estimatedSharpImage_outLayer = nn.Sequential(*[nn.Conv2d(CW, 3, 3, 1, 1), nn.Sigmoid()])


    def wienerDeconvoltion(self, blurryImageFeature, kernelFeature):
        #g = IDFT( DFT(K)* / (DFT(K)DFT(K)* + sn/sx) )

        meanedBlurryImageFeature = F.interpolate(F.avg_pool2d(blurryImageFeature, kernel_size=3), size=blurryImageFeature.size()[-2:], mode='bicubic')

        sx = torch.std(blurryImageFeature, dim=(-1,-2), keepdim=True)
        sn = torch.var(blurryImageFeature - meanedBlurryImageFeature, dim=(-1,-2), keepdim=True)

        NSR = sn/(sx + 1e-8)
        #print(weight.max(), weight.min(), weight.mean())
        f_blurryImageFeature = torch.fft.fft2(blurryImageFeature)
        f_kernelFeature = torch.fft.fft2(kernelFeature)
        f_G = f_kernelFeature.conj() / (f_kernelFeature.conj()*f_kernelFeature + NSR)
        f_estimatedSharpImageFeature = f_G * f_blurryImageFeature

        return torch.fft.ifft2(f_estimatedSharpImageFeature).real


    def forward(self, x):
        blurryImageFeature = self.blurryImage_inLayer(x)
        # blurryImageFeature = self.blurryImage_res(blurryImageFeature)
        # blurryImageFeature = self.blurryImage_outLayer(blurryImageFeature)

        kernelFeature = self.kernel_inLayer(x)
        kernelFeature = self.kernel_res(kernelFeature)
        kernelFeature = self.kernel_outLayer(kernelFeature)

        estimatedSharpImageFeature = self.wienerDeconvoltion(blurryImageFeature, kernelFeature)
        # estimatedSharpImageFeature = self.estimatedSharpImage_inLayer(estimatedSharpImageFeature)
        # estimatedSharpImageFeature = self.estimatedSharpImage_res(estimatedSharpImageFeature)
        estimatedSharpImage = self.estimatedSharpImage_outLayer(estimatedSharpImageFeature)

        # blurryImageReconstructioned = self.estimatedSharpImage_inLayer(blurryImageFeature)
        # blurryImageReconstructioned = self.estimatedSharpImage_res(blurryImageReconstructioned)
        blurryImageReconstructioned = self.estimatedSharpImage_outLayer(blurryImageFeature)        

        return estimatedSharpImage, blurryImageReconstructioned



class Keresti(nn.Module):
    def __init__(self, n_channels, n_blocks, n_modules, kernel_size=3, normalize=False, act=nn.ReLU(True), attention=True): #scale=(2,3,4)
        super(Keresti, self).__init__()
        self.n_modules = n_modules

        ### INPUT ###
        self.input = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True)

        ### BODY ###
        if n_modules == 1:
            self.body = nn.Sequential(SingleModule(n_channels, n_blocks, act, attention))
        else:
            self.body = nn.Sequential(*[SingleModule(n_channels, n_blocks, act, attention) for _ in range(n_modules)])

        self.tail = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)

        self.upscale = UpScale(n_channels=n_channels, scale=kernel_size, act=False)

        self.output = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.normalize = normalize

        self.kernel_size = kernel_size


    def kernelMaker(self, x, kernelSize):
        body_input = self.input(x)
        body_output = self.body(body_input)

        if self.n_modules == 1:
            sr_high = self.upscale(body_output)
        else:
            sr_high = self.upscale(self.tail(body_output) + body_input)

        kernel = self.output(sr_high) # N C 3H 3W   # in-place sigmoid for Memory efficiency
        kernel = tileSplit(kernel, (kernelSize,kernelSize)) # N C(3) H W 3 3
        kernel = kernel.view(*kernel.size()[:2],1,*kernel.size()[2:]) # N oC(3) iC(1) H W 3 3
        
        #kernel_identity = torch.tensor([[1.0]]).cuda()
        #kernel_identity = F.pad(kernel_identity, (self.kernel_size//2,self.kernel_size//2,self.kernel_size//2,self.kernel_size//2)).view(1,1,1,1,1,self.kernel_size,self.kernel_size)
        #kernel_identity = kernel_identity.repeat(*kernel.size()[:5],1,1)

        #kernel = F.tanh(kernel) + 1
        print(kernel.mean())
        kernel /= (kernel.sum((-2,-1), keepdim=True) + 1e-12)
        
        return kernel



    def forward(self, x):

        x = self.kernelMaker(x, self.kernel_size)

        return x




class UpScaleAdv(nn.Sequential):
    def __init__(self, n_channels, scale, bn=False, act=nn.ReLU(inplace=True), bias=False):
        layers = []
        '''
        # power of 2
        if (scale & (scale - 1)) == 0: 
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4 * n_channels, kernel_size=3, stride=1,
                                        padding=1, bias=bias))
                layers.append(nn.PixelShuffle(2))
                if bn: layers.append(nn.BatchNorm2d(n_channels))
                if act: layers.append(act)
        elif scale % 2 == 1:
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=scale * scale * n_channels, kernel_size=3, stride=1,
                                    padding=1, bias=bias))
            layers.append(nn.PixelShuffle(scale))
            if bn: layers.append(nn.BatchNorm2d(n_channels))
            if act: layers.append(act)
        else:
            raise NotImplementedError
        '''
        
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=scale * scale * n_channels, kernel_size=3, stride=1,
                                padding=1, bias=bias))
        layers.append(nn.PixelShuffle(scale))
        if bn: layers.append(nn.BatchNorm2d(n_channels))
        if act: layers.append(act)

        super(UpScaleAdv, self).__init__(*layers)


class IPSA(nn.Module):
    def __init__(self, n_channels, n_blocks, n_modules, kernel_size=3, normalize=False, act=nn.ReLU(True), attention=True): #scale=(2,3,4)
        super(IPSA, self).__init__()
        self.n_modules = n_modules

        ### INPUT ###
        self.input = nn.ModuleList([nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.n_modules)])

        '''
        ### BODY ###
        if n_modules == 1:
            self.body = nn.Sequential(SingleModule(n_channels, n_blocks, act, attention))
        else:
            self.body = nn.Sequential(*[SingleModule(n_channels, n_blocks, act, attention) for _ in range(n_modules)])
        '''

        self.body = nn.ModuleList([SingleModule(n_channels, n_blocks, act, attention) for _ in range(self.n_modules)])

        self.tail = nn.ModuleList([nn.Conv2d(n_channels, n_channels, 3, kernel_size, 1, bias=True) for _ in range(self.n_modules)])

        self.upscale = nn.ModuleList([UpScale(n_channels=n_channels, scale=kernel_size, act=False) for _ in range(self.n_modules)])

        self.output = nn.ModuleList([nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=True) for _ in range(self.n_modules)])
        self.normalize = normalize

        self.kernel_size = kernel_size

    def kernelMaker(self, moduleIndex, x, kernelSize):

        body_input = self.input[moduleIndex](x)
        body_output = self.body[moduleIndex](body_input)
        body_tail = self.tail[moduleIndex](body_input)

        '''
        if self.n_modules == 1:
            sr_high = self.upscale(body_output)
        else:
            sr_high = self.upscale(self.tail(body_output) + body_input)
        '''
        sr_high = self.upscale[moduleIndex](body_tail)

        kernel = self.output[moduleIndex](sr_high) # N C 3H 3W   # in-place sigmoid for Memory efficiency
        kernel = tileSplit(kernel, (kernelSize,kernelSize)) # N C(3) H W 3 3
        kernel = kernel.view(*kernel.size()[:2],1,*kernel.size()[2:]) # N oC(3) iC(1) H W 3 3
        ##kernel_identity = torch.tensor([[1.0]]).cuda()
        #print(kernel_identity)
        kernel_identity = F.pad(kernel_identity, (self.kernel_size//2,self.kernel_size//2,self.kernel_size//2,self.kernel_size//2)).view(1,1,1,1,1,self.kernel_size,self.kernel_size)

        #print(kernel_identity)
        kernel_identity = kernel_identity.repeat(*kernel.size()[:5],1,1)
        kernel_hole = torch.ones_like(kernel_identity) - kernel_identity
        

        kernel = kernel_identity + kernel# * kernelSize ** 2 + kernel
        return kernel

    def Mise_en_Scene(self, moduleIndex, x):

        kernel = self.kernelMaker(moduleIndex, x, self.kernel_size) # N,oC(3 for color),iC(1),H,W,kH(3),kW(3)
        x = torch.cat([module.nonUniformInverseConv2d(x[:,i:i+1,:,:], kernel, pad=self.kernel_size//2, stride=self.kernel_size) for i in range(3)], 1)

        return x

    def forward(self, x, maxModule=None):

        for i in range(self.n_modules if maxModule is None or maxModule > self.n_modules else maxModule):
            x = self.Mise_en_Scene(i, x)

        return x



def tileSplit(x, size):
    assert len(x.size()) == 4
    h = size[0]
    w = size[1]

    b = x.view(*x.size()[:-2],1,1,*x.size()[-2:])
    b = b.split(w, dim=-2)
    b = torch.cat(b, dim=-4)
    b = b.split(h, dim=-1)
    b = torch.cat(b, dim=-3)

    return b



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class DAC(nn.Module):
    def __init__(self, n_channels):
        super(DAC, self).__init__()

        self.mean = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )
        self.std = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, 1, 0, 1, 1, False),
            # nn.BatchNorm2d(n_channels),
        )

    def forward(self, observed_feat, referred_feat):
        assert (observed_feat.size()[:2] == referred_feat.size()[:2])
        size = observed_feat.size()
        referred_mean, referred_std = calc_mean_std(referred_feat)
        observed_mean, observed_std = calc_mean_std(observed_feat)

        normalized_feat = (observed_feat - observed_mean.expand(
            size)) / observed_std.expand(size)
        referred_mean = self.mean(referred_mean)
        referred_std = self.std(referred_std)
        output = normalized_feat * referred_std.expand(size) + referred_mean.expand(size)
        return output


class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=pad,
                                 dilation=pad, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # # Freeze the MeanShift layer
        # for params in self.parameters():
        #     params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian


class rcab_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(rcab_block, self).__init__()

        block = []

        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))
        block.append(activation)
        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))

        self.block = nn.Sequential(*block)

        self.calayer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16 if n_channels >= 16 else 1, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16 if n_channels >= 16 else 1, n_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        residue = self.block(x)
        chnlatt = F.adaptive_avg_pool2d(residue, 1)
        chnlatt = self.calayer(chnlatt)
        output = x + residue * chnlatt

        return output


class mrcab_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(mrcab_block, self).__init__()

        block = []

        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))
        block.append(activation)
        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=kernel, padding=1, bias=bias))

        self.block = nn.Sequential(*block)

        self.n_channels = n_channels

        self.calayer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        rst = self.block(x)
        multiplier = rst[:,self.n_channels:,:,:]
        residue = rst[:,:self.n_channels,:,:]
        chnlatt = F.adaptive_avg_pool2d(residue, 1)
        chnlatt = self.calayer(chnlatt)
        output = x * multiplier + residue * chnlatt

        return output



class crcab_block(nn.Module):
    def __init__(self, n_channels, kernel, bias=False, activation=nn.ReLU(inplace=True)):
        super(crcab_block, self).__init__()

        self.KSIZE = 3

        block = []

        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel, padding=1, bias=bias))
        block.append(activation)
        block.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels * n_channels * self.KSIZE * self.KSIZE, kernel_size=kernel, padding=1, bias=bias))

        self.block = nn.Sequential(*block)

        self.n_channels = n_channels

        self.calayer = nn.Sequential(
            nn.Conv2d(n_channels, n_channels // 16, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):

        rst = self.block(x)

        N, C_KH_KW, H, W = rst.size()

        weight = F.tanh(rst.reshape(N, self.n_channels, self.n_channels, H, W, self.KSIZE, self.KSIZE))
        
        output = module.nonUniformConv2d(x, weight, pad=1)

                    

        #output = x * multiplier + residue * chnlatt

        return output



class DiEnDec(nn.Module):
    def __init__(self, n_channels, act=nn.ReLU(inplace=True)):
        super(DiEnDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * 2, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
            nn.Conv2d(n_channels * 2, n_channels * 4, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.Conv2d(n_channels * 4, n_channels * 8, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_channels * 8, n_channels * 4, kernel_size=3, padding=4, dilation=4, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, kernel_size=3, padding=2, dilation=2, bias=True),
            act,
            nn.ConvTranspose2d(n_channels * 2, n_channels, kernel_size=3, padding=1, dilation=1, bias=True),
            act,
        )
        self.gate = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        output = self.gate(self.decoder(self.encoder(x)))
        return output


class SingleModule(nn.Module):
    def __init__(self, n_channels, n_blocks, act, attention):
        super(SingleModule, self).__init__()
        res_blocks = [rcab_block(n_channels=n_channels, kernel=3, activation=act) for _ in range(n_blocks)]
        self.body_block = nn.Sequential(*res_blocks)
        self.attention = attention
        if attention:
            self.coder = nn.Sequential(DiEnDec(3, act))
            self.dac = nn.Sequential(DAC(n_channels))
            self.hessian3 = nn.Sequential(MSHF(n_channels, kernel=3))
            self.hessian5 = nn.Sequential(MSHF(n_channels, kernel=5))
            self.hessian7 = nn.Sequential(MSHF(n_channels, kernel=7))

    def forward(self, x):
        sz = x.size()
        resin = self.body_block(x)

        if self.attention:
            hessian3 = self.hessian3(resin)
            hessian5 = self.hessian5(resin)
            hessian7 = self.hessian7(resin)
            hessian = torch.cat((torch.mean(hessian3, dim=1, keepdim=True),
                                 torch.mean(hessian5, dim=1, keepdim=True),
                                 torch.mean(hessian7, dim=1, keepdim=True))
                                , 1)
            hessian = self.coder(hessian)
            attention = torch.sigmoid(self.dac[0](hessian.expand(sz), x))
            resout = resin * attention
        else:
            resout = resin

        output = resout + x

        return output


class Generator(nn.Module):
    def __init__(self, n_channels, n_blocks, n_modules, scale, normalize=False, act=nn.ReLU(True), attention=True): #scale=(2,3,4)
        super(Generator, self).__init__()
        self.n_modules = n_modules
        self.input = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=True)
        if n_modules == 1:
            self.body = nn.Sequential(SingleModule(n_channels, n_blocks, act, attention))
        else:
            self.body = nn.Sequential(*[SingleModule(n_channels, n_blocks, act, attention) for _ in range(n_modules)])

        self.tail = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)

        scale = [scale]
        self.upscale = nn.ModuleList([UpScale(n_channels=n_channels, scale=s, act=False) for s in scale])

        self.output = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.normalize = normalize

    def forward(self, x):

        if self.normalize in [True, 1] :
            mean = torch.tensor([0.485, 0.456, 0.406], device=torch.device('cuda')).view(1,-1,1,1)
            std = torch.tensor([0.229, 0.224, 0.225],device=torch.device('cuda')).view(1,-1,1,1)
            x = (x - mean) / std

        body_input = self.input(x)
        body_output = self.body(body_input)
        if self.n_modules == 1:
            sr_high = self.upscale[0](body_output)
        else:
            sr_high = self.upscale[0](self.tail(body_output) + body_input)
        results = self.output(sr_high)

        if self.normalize in [True, 1]:
            mean = torch.tensor([-2.118, -2.036, -1.804], device=torch.device('cuda')).view(1,-1,1,1)
            std = torch.tensor([4.367, 4.464, 4.444],device=torch.device('cuda')).view(1,-1,1,1)
            x = (x - mean) / std

        return results


class UpScale(nn.Sequential):
    def __init__(self, n_channels, scale, bn=False, act=nn.ReLU(inplace=True), bias=False):
        layers = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                layers.append(nn.Conv2d(in_channels=n_channels, out_channels=4 * n_channels, kernel_size=3, stride=1,
                                        padding=1, bias=bias))
                layers.append(nn.PixelShuffle(2))
                if bn: layers.append(nn.BatchNorm2d(n_channels))
                if act: layers.append(act)
        elif scale % 2 == 1:
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=scale * scale * n_channels, kernel_size=3, stride=1,
                                    padding=1, bias=bias))
            layers.append(nn.PixelShuffle(scale))
            if bn: layers.append(nn.BatchNorm2d(n_channels))
            if act: layers.append(act)
        else:
            raise NotImplementedError

        super(UpScale, self).__init__(*layers)




class OASIS(nn.Module):
    '''
    output image size == feature size * (2 ** power)
    memoryMode == ['poor' || 'rich']
    DO_NOT_FORCE_CUDA needs to utils.loadmodel cannot load this model cuda
    '''
    def __init__(self, ResNeStFeatureExtractor, numClass=64, inCh=2048, power=6, CWDevider=4, colorMode = 'color', memoryMode = 'rich', maxClusterMemorySize = 5):
        super(OASIS, self).__init__()

        self.DO_NOT_FORCE_CUDA = True

        assert ResNeStFeatureExtractor is not None
        self.ResNeStFeatureExtractor = ResNeStFeatureExtractor

        self.numClass = numClass
        self.inCh = inCh
        self.CWDevider = CWDevider
        assert colorMode in ['color', 'grayscale']
        self.outCh = 3 if colorMode == 'color' else 1
        self.power = power
        assert memoryMode in ['poor', 'rich']
        self.memoryMode = memoryMode

        self.maxClusterMemorySize = maxClusterMemorySize
        self.clusterMemory = []
        self.clusterer = KMeans(n_clusters=self.numClass, mode='euclidean', verbose=0, max_iter=50)

        self.reconstructorList = nn.ModuleList([self._makeReconstructor(self.inCh, self.outCh, self.CWDevider, self.power, self.memoryMode) for i in range(self.numClass)])

    def _makeReconstructor(self, inCh, outCh, CWDevider, power, memoryMode):
        return tSeNseR(inCh=inCh, outCh=outCh, CWDevider=CWDevider, power=power).cpu() if memoryMode == 'poor' else tSeNseR(inCh=inCh, outCh=outCh, CWDevider=CWDevider, power=power).cuda()

    def _clustring(self, scoreMap):

        #N 256 64 64
        N, C, H, W = scoreMap.size()
        
        scoreMap = scoreMap.view(N,C,H*W).transpose(1,2).reshape(N*H*W,C)
        #scoreMap = scoreMap.view(N*H*W, C)

        if len(self.clusterMemory) == self.maxClusterMemorySize:
            self.clusterMemory.pop(0)
        self.clusterMemory.append(scoreMap)

        self.clusterer.fit(torch.cat(self.clusterMemory, 0))

        classMap = self.clusterer.predict(scoreMap)
        classMap = classMap.view(N,H,W).unsqueeze(1)

        return classMap

    def _makeClassMap(self, f1, size):
        classMap = F.interpolate(f1, size = size, mode='bicubic')
        classMap = self._clustring(classMap)

        return classMap

    def _applyDecoder(self, fList, classNumber, LR = None):
        
        if self.memoryMode == 'poor':
            self.reconstructorList[classNumber].cuda()

        x = self.reconstructorList[classNumber](fList, LR)

        if self.memoryMode == 'poor':
            self.reconstructorList[classNumber].cpu()

        return x

    def _fusion(self, xList, classMap):
        rst = torch.zeros_like(xList[0])
        for i in range(self.numClass):
            rst = rst + xList[i] * (classMap == i)
        return rst

    def forward(self, x):

        f1, f2, f3, f4 = self.ResNeStFeatureExtractor(x)

        srList = [self._applyDecoder([f1, f2, f3, f4], i) for i in range(self.numClass)]

        classMap = self._makeClassMap(f1, srList[0].size()[-2:])

        rst = self._fusion(srList, classMap)

        return rst



class tSeNseR(nn.Module):
    def __init__(self, inCh=2048, outCh=3, CWDevider=1, power=5):
        super(tSeNseR, self).__init__()

        self.inCh = inCh
        self.CWDevider = CWDevider
        self.outCh = outCh
        self.power = power

        self.decoderList, self.oxoConvList = self._makeDecoderList(self.inCh, self.outCh, self.CWDevider, self.power)

    def _decoder(self, inCh, outCh, normLayer=nn.BatchNorm2d, act=nn.ReLU):  # C // 2 , HW X 2
        mList = []
        mList.append(nn.ConvTranspose2d(inCh, outCh, 4, 2, 1))
        if normLayer is not None:
            mList.append(normLayer(outCh))
        if act is not None:
            mList.append(act())
        return nn.Sequential(*mList)

    def _makeDecoderList(self, inCh, outCh, CWDevider, power):
        decoderList = [self._decoder(inCh, inCh // (2 * CWDevider))]
        oxoConvList = [None]

        for x in range(1, power - 1):
            _ic = inCh // (2 ** x)
            _oc = inCh // (2 ** (x+1))
            decoderList.append(self._decoder(_ic // CWDevider, _oc // CWDevider))
            oxoConvList.append(nn.Conv2d(_ic, _ic // CWDevider, kernel_size=1))

        decoderList.append(self._decoder(inCh // (2 ** (power - 1) * CWDevider), outCh, act=None))
        oxoConvList.append(nn.Conv2d(inCh // (2 ** (power - 1)), inCh // (2 ** (power - 1) * CWDevider), kernel_size=1))
        return nn.ModuleList(decoderList), nn.ModuleList(oxoConvList)


    def forward(self, fList, LR=None):

        x = None

        for i in range(len(self.decoderList)):
            decoder = self.decoderList[i]
            oxoConv = self.oxoConvList[i]

            if i == 0:
                x = decoder(fList[-1])

            elif len(self.decoderList) - 1 > i > 0:
                x = decoder(x + oxoConv(fList[-(1 + i)])) if len(fList) >= i+1 else decoder(x)
                
            elif i == len(self.decoderList) - 1:
                x = decoder(x + oxoConv(fList[-(1 + i)])) if len(fList) >= i+1 else decoder(x)
                x = F.tanh(x) + LR if LR is not None else F.sigmoid(x)

        return x



class tSeNseR_OLD(nn.Module):
    def __init__(self, CW=2048, colorMode="color"):
        super(tSeNseR_OLD, self).__init__()

        self.CW = CW

        self.decoder1 = self.decoder(CW, CW // 2)  # 2->4
        self.decoder2 = self.decoder(CW // 2, CW // 4)  # 4->8
        self.decoder3 = self.decoder(CW // 4, CW // 8)  # 8->16
        self.decoder4 = self.decoder(CW // 8, CW // 16)  # 16->32

        self.decoderHR = self.decoder(CW // 16, 3 if colorMode == "color" else 1, act=None)

    def decoder(self, inCh, outCh, normLayer=nn.BatchNorm2d, act=nn.ReLU):  # C // 2 , HW X 2
        mList = []
        mList.append(nn.ConvTranspose2d(inCh, outCh, 4, 2, 1))
        if normLayer is not None:
            mList.append(normLayer(outCh))
        if act is not None:
            mList.append(act())
        return nn.Sequential(*mList)

    def forward(self, f1, f2, f3, f4, LR=None):

        decoded_f4 = self.decoder1(f4)

        decoded_f3 = self.decoder2(decoded_f4 + f3)

        decoded_f2 = self.decoder3(decoded_f3 + f2)

        decoded_f1 = self.decoder4(decoded_f2 + f1)

        decoded = F.tanh(self.decoderHR(decoded_f1)) + LR if LR is not None else F.sigmoid(self.decoderHR(decoded_f1))

        return decoded


class DeNIQuA_Res(nn.Module):
    def __init__(self, featureExtractor, CW=64, Blocks=9, inFeature=1, outCW=3, featureCW=1280):
        super(DeNIQuA_Res, self).__init__()

        self.featureExtractor = featureExtractor

        self.CW = CW

        self.inFeature = inFeature

        self.oxo_in = nn.Conv2d(featureCW * inFeature, CW, 1, 1, 0)
        self.res = basicResBlocks(CW=CW, Blocks=9)
        self.oxo_out = nn.Conv2d(CW, outCW, 1, 1, 0)

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

        self.DecoderList = nn.ModuleList(
            [  # 1/32
                nn.ConvTranspose2d(featureCW * inFeature, CW * 8, 4, 2, 1),  # 1/16
                nn.ConvTranspose2d(CW * 8, CW * 4, 4, 2, 1),  # 1/8
                nn.ConvTranspose2d(CW * 4, CW * 2, 4, 2, 1),  # 1/4
                nn.ConvTranspose2d(CW * 2, CW * 1, 4, 2, 1),  # 1/2
                nn.ConvTranspose2d(CW * 1, outCW, 4, 2, 1),  # 1/1
            ]
        )

    def forward(self, xList, sc=None):

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

        return F.sigmoid(x + sc) if sc is not None else F.sigmoid(x)


class DeNIQuAdv(nn.Module):
    def __init__(self, featureExtractor, CW=64, inFeature=1, outCW=3, featureCW=1280):
        super(DeNIQuAdv, self).__init__()

        self.featureExtractor = featureExtractor

        self.CW = CW

        self.inFeature = inFeature

        self.oxo_in_1 = nn.Conv2d(featureCW * inFeature, featureCW, 1, 1, 0)
        self.oxo_in_2 = nn.Conv2d(featureCW, featureCW // 4, 1, 1, 0)

        self.res_1 = basicResBlocks(CW=featureCW // 4, Blocks=9, lastAct=False)
        self.res_2 = basicResBlocks(CW=featureCW // 4, Blocks=9, lastAct=False)
        self.res_3 = basicResBlocks(CW=featureCW // 4, Blocks=9, lastAct=False)

        self.oxo_out_1 = nn.Conv2d(featureCW // 4, featureCW, 1, 1, 0)
        self.oxo_out_2 = nn.Conv2d(featureCW, CW * 8, 1, 1, 0)

        self.DecoderList = nn.ModuleList(
            [  # 1/32
                nn.ConvTranspose2d(CW * 8, CW * 8, 4, 2, 1),  # 1/16
                nn.ConvTranspose2d(CW * 8, CW * 4, 4, 2, 1),  # 1/8
                nn.ConvTranspose2d(CW * 4, CW * 2, 4, 2, 1),  # 1/4
                nn.ConvTranspose2d(CW * 2, CW * 1, 4, 2, 1),  # 1/2
                nn.ConvTranspose2d(CW * 1, outCW, 4, 2, 1),  # 1/1
            ]
        )

    def forward(self, xList, sc=None):

        assert self.inFeature == len(xList)

        # FE
        if self.featureExtractor is not None:
            rstList = []
            for i in range(self.inFeature):
                rstList.append(self.featureExtractor(xList[i]))
            x = torch.cat(rstList, 1)
        else:
            x = torch.cat(xList, 1)

        # RES

        x = F.relu(self.oxo_in_1(x))
        sc2 = x
        x = F.relu(self.oxo_in_2(x))

        sc3 = x
        x = F.relu(self.res_1(x) + sc3)
        sc4 = x
        x = F.relu(self.res_2(x) + sc4)
        sc5 = x
        x = F.relu(self.res_3(x) + sc5)

        x = F.relu(self.oxo_out_1(x) + sc2)
        x = F.relu(self.oxo_out_2(x))

        # Decode
        for i, decoder in enumerate(self.DecoderList):
            x = decoder(x)
            if i + 1 < len(self.DecoderList):
                x = F.relu(x)

        return F.tanh(x) + sc if sc is not None else F.sigmoid(x)


class basicResBlocks(nn.Module):
    def __init__(self, CW, Blocks, kernelSize=3, dim=2, lastAct=True):
        super(basicResBlocks, self).__init__()

        assert dim in [2, 3]

        self.Convs = nn.ModuleList()
        self.Blocks = Blocks
        for i in range(self.Blocks):
            if dim == 2:
                self.Convs.append(nn.Conv2d(CW, CW, kernelSize, 1, 1))
                self.IN = nn.InstanceNorm2d(CW)
            elif dim == 3:
                self.Convs.append(nn.Conv3d(CW, CW, kernelSize, 1, 1))
                self.IN = nn.InstanceNorm3d(CW)
        self.act = nn.LeakyReLU(0.2)
        self.lastAct = lastAct

    def forward(self, x):

        for i in range(self.Blocks):
            res = x
            x = self.Convs[i](x)
            x = self.IN(x)
            x = x + res
            if i + 1 != self.Blocks or self.lastAct is True:
                x = self.act(x)

        return x
