"""
model.py
"""
version = "1.60.201230"


# from Python
import time
import csv
import os
import math
import numpy as np
import sys
import functools
from shutil import copyfile

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


class tSeNseR(nn.Module):
    def __init__(self, CW=2048, colorMode="color"):
        super(tSeNseR, self).__init__()

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


##########DMPHN##########
class DMPHN_Encoder(nn.Module):
    def __init__(self):
        super(DMPHN_Encoder, self).__init__()
        # Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        # Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        # Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Conv1
        # print("Encoder input : ", x.shape)
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        # Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        # Conv3
        x = self.layer9(x)
        x = self.layer10(x) + x
        x = self.layer11(x) + x
        # print("Encoder output : ", x.shape)
        return x


class DMPHN_Decoder(nn.Module):
    def __init__(self):
        super(DMPHN_Decoder, self).__init__()
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        # Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, padding=1)
        )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        # Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, kernel_size=3, padding=1)
        )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # Deconv3
        # print("Decoder input : ", x.shape)
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)
        # Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        # Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        # print("Decoder output : ", x.shape)
        return x
