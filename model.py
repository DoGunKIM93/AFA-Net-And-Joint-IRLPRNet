"""
model.py
"""
version = "1.61.210122"


# from Python
import time
import csv
import os
import math
import numpy as np
import sys
import functools
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