'''
model.py
'''
version = '1.4.200729'


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
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

#from this project
import param as p
import backbone.vision as vision
    #CovPool
from backbone.module import CovpoolLayer, SqrtmLayer, CovpoolLayer3d, SqrtmLayer3d, TriuvecLayer
    #EDVR
from backbone.EDVR.EDVR import EDVR_Predeblur_ResNet_Pyramid, EDVR_PCD_Align, EDVR_TSA_Fusion, EDVR_make_layer, EDVR_ResidualBlock_noBN
    #Attentions
from backbone.module import SecondOrderChannalAttentionBlock, NonLocalBlock, CrissCrossAttention
    #EfficientNet
from backbone.EfficientNet.EfficientNet import EfficientNetBuilder, EFF_create_conv2d, EFF_round_channels, SelectAdaptivePool2d, efficientnet_init_weights, EFF_decode_arch_def
    #ResNeSt
from backbone.ResNeSt.resnest import resnest50, resnest101, resnest200, resnest269
from backbone.ResNeSt.resnet import ResNet, Bottleneck




############################################

# 1. 2nd ORder CA
# 2. InstantNorm
# 3. EDVR
# 4. Spatial A

inputChannel = 1 if p.colorMode =='grayscale' else 3



class DeNIQuA(nn.Module):
    def __init__(self, featureExtractor, CW=64, inFeature=1):
        super(DeNIQuA, self).__init__()

        self.featureExtractor = featureExtractor

        self.CW = CW

        self.inFeature = inFeature

        self.DecoderList = nn.ModuleList([ # 1/32
            nn.ConvTranspose2d(1280*inFeature,   CW*8, 4, 2, 1), #1/16
            nn.ConvTranspose2d( CW*8,  CW*4, 4, 2, 1), #1/8
            nn.ConvTranspose2d( CW*4,  CW*2, 4, 2, 1), #1/4
            nn.ConvTranspose2d( CW*2 , CW*1, 4, 2, 1), #1/2
            nn.ConvTranspose2d( CW*1 , inputChannel, 4, 2, 1), #1/1
        ])

    def forward(self, xList):

        assert self.inFeature == len(xList)
        
        rstList = []
        for i in range(self.inFeature):
            rstList.append(self.featureExtractor(xList[i]))
        x = torch.cat(rstList, 1)

        for i, decoder in enumerate(self.DecoderList):
            x = decoder(x)
            if i + 1 < len(self.DecoderList):    
                x = F.relu(x)

        return F.sigmoid(x)



class SunnySideUp(nn.Module):
    
    def __init__(self, CW=32):
        super(SunnySideUp, self).__init__()

        self.conv1 = nn.Conv2d(        3, CW *   1, 4, 2, 1) # 256 -> 128
        self.conv2 = nn.Conv2d(CW *   1, CW *   2, 4, 2, 1) # 256 -> 128
        self.conv3 = nn.Conv2d(CW *   2, CW *   4, 4, 2, 1) # 256 -> 128
        self.conv4 = nn.Conv2d(CW *   4, CW *   8, 4, 2, 1) # 256 -> 128
        self.conv5 = nn.Conv2d(CW *   8, 1, 4, 2, 1) # 256 -> 128
        self.AAP = nn.AdaptiveAvgPool2d((3, 3))

        self.shifter = Shifter()


    def forward(self, x):

        ori = x

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)

        x = F.leaky_relu(self.conv3(x), 0.2)

        x = F.leaky_relu(self.conv4(x), 0.2)

        x = F.leaky_relu(self.conv5(x), 0.2)

        x = F.sigmoid(self.AAP(x)).repeat(1,3,1,1)
        
        ori = self.shifter(ori, x)

        return ori




class BlueLemonade(nn.Module):#v2 
    
    def __init__(self, featureExtractor, CW = p.NGF, blendingChannel = 3, shiftDistance = 9):#, CW = p.NDF, Blocks = 5, ResPerBlocks = 10):
        super(BlueLemonade, self).__init__()

        self.blendingChannel = blendingChannel
        self.shiftDistance = shiftDistance
        self.featureExtractor = featureExtractor

        self.adaptiveFilterMaker = nn.Sequential( # 1/32
            nn.ConvTranspose2d( 3840,  CW*16, 4, 2, 1), #1/16
            nn.ReLU(),
            nn.ConvTranspose2d( CW*16, CW*8, 4, 2, 1), #1/8
            nn.ReLU(),
            nn.ConvTranspose2d( CW*8,  CW*4, 4, 2, 1), #1/4
            nn.ReLU(),
            nn.ConvTranspose2d( CW*4,  CW, 4, 2, 1), #1/2
            nn.ReLU(),
            nn.ConvTranspose2d( CW, inputChannel * self.blendingChannel, 4, 2, 1), #1/1
            nn.Tanh(),
        )

        self.shiftKernelMaker = nn.Sequential( # 1/32
            nn.ConvTranspose2d( 3840,  CW*16, 4, 2, 1), #1/16
            nn.ReLU(),
            nn.ConvTranspose2d( CW*16, CW*8, 4, 2, 1), #1/8
            nn.ReLU(),
            nn.ConvTranspose2d( CW*8,  CW*4, 4, 2, 1), #1/4
            nn.ReLU(),
            nn.ConvTranspose2d( CW*4,  CW, 4, 2, 1), #1/2
            nn.ReLU(),
            nn.ConvTranspose2d( CW, self.blendingChannel, 4, 2, 1), #1/1
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.shiftDistance, self.shiftDistance)),
            nn.Sigmoid(),
        )
        #for i in range(Blocks):
        #    self.kernelMaker.append(ReconstructionBlock(CW, ResPerBlocks, dim=2, use_attention=False, attention_sub_sample=None))

        self.shifter = Shifter()

        self.iteration = iteration
    
    def kernelMaking(self, a, b, c):
        cated = torch.cat((a,b,c),1)
        adaptiveFilter = self.adaptiveFilterMaker(cated) * (2 / self.iteration) 
        shiftKernel = self.shiftKernelMaker(cated)
        return adaptiveFilter, [shiftKernel[:,0:1,:,:].repeat(1,3,1,1), shiftKernel[:,1:2,:,:].repeat(1,3,1,1), shiftKernel[:,2:3,:,:].repeat(1,3,1,1)]

    def applyFilter(self, ori, aFilter):
        aFilter = aFilter.view(aFilter.size(0), aFilter.size(1) // inputChannel, inputChannel, *aFilter.size()[2:])
        aFilter = F.softmax(aFilter, dim=1)#.view(x.size(0), -1, *x.size()[2:])

        ori = ori.view(ori.size(0), ori.size(1) // inputChannel, inputChannel, *ori.size()[2:])

        return (aFilter * ori).sum(dim=1)

    def forward(self, x1, x2):

        xb = x1
        x1f = self.featureExtractor(x1)
        x2f = self.featureExtractor(x2)

        for i in range(self.iteration):

            xbf = self.featureExtractor(xb)

            adaptiveFilter, shiftKernel = self.kernelMaking(x1f, x2f, xbf)

            shiftedX1 = self.shifter(x1, shiftKernel[0])
            shiftedX2 = self.shifter(x2, shiftKernel[1])
            shiftedXB = self.shifter(xb, shiftKernel[2])

            catedX = torch.cat((shiftedX1, shiftedX2, shiftedXB), 1)
            filterdX = self.applyFilter(catedX, adaptiveFilter)

            xb = filterdX

        return xb


class ISAF(nn.Module):#v2 
    
    def __init__(self, featureExtractor, CW = p.NGF, iteration = 5, blendingChannel = 3, shiftDistance = 3):#, CW = p.NDF, Blocks = 5, ResPerBlocks = 10):
        super(ISAF, self).__init__()

        self.blendingChannel = blendingChannel
        self.shiftDistance = shiftDistance
        self.featureExtractor = featureExtractor

        self.adaptiveFilterMaker = nn.Sequential( # 1/32
            nn.ConvTranspose2d( 3840,  CW*16, 4, 2, 1), #1/16
            nn.ReLU(),
            nn.ConvTranspose2d( CW*16, CW*8, 4, 2, 1), #1/8
            nn.ReLU(),
            nn.ConvTranspose2d( CW*8,  CW*4, 4, 2, 1), #1/4
            nn.ReLU(),
            nn.ConvTranspose2d( CW*4,  CW, 4, 2, 1), #1/2
            nn.ReLU(),
            nn.ConvTranspose2d( CW, inputChannel * self.blendingChannel, 4, 2, 1), #1/1
            nn.Tanh(),
        )

        self.shiftKernelMaker = nn.Sequential( # 1/32
            nn.ConvTranspose2d( 3840,  CW*16, 4, 2, 1), #1/16
            nn.ReLU(),
            nn.ConvTranspose2d( CW*16, CW*8, 4, 2, 1), #1/8
            nn.ReLU(),
            nn.ConvTranspose2d( CW*8,  CW*4, 4, 2, 1), #1/4
            nn.ReLU(),
            nn.ConvTranspose2d( CW*4,  CW, 4, 2, 1), #1/2
            nn.ReLU(),
            nn.ConvTranspose2d( CW, self.blendingChannel, 4, 2, 1), #1/1
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.shiftDistance, self.shiftDistance)),
            nn.Sigmoid(),
        )
        #for i in range(Blocks):
        #    self.kernelMaker.append(ReconstructionBlock(CW, ResPerBlocks, dim=2, use_attention=False, attention_sub_sample=None))

        self.shifter = Shifter()

        self.iteration = iteration
    
    def kernelMaking(self, a, b, c):
        cated = torch.cat((a,b,c),1)
        adaptiveFilter = self.adaptiveFilterMaker(cated) * (2 / self.iteration) 
        shiftKernel = self.shiftKernelMaker(cated)
        return adaptiveFilter, [shiftKernel[:,0:1,:,:].repeat(1,3,1,1), shiftKernel[:,1:2,:,:].repeat(1,3,1,1), shiftKernel[:,2:3,:,:].repeat(1,3,1,1)]

    def applyFilter(self, ori, aFilter):
        aFilter = aFilter.view(aFilter.size(0), aFilter.size(1) // inputChannel, inputChannel, *aFilter.size()[2:])
        aFilter = F.softmax(aFilter, dim=1)#.view(x.size(0), -1, *x.size()[2:])

        ori = ori.view(ori.size(0), ori.size(1) // inputChannel, inputChannel, *ori.size()[2:])

        return (aFilter * ori).sum(dim=1)

    def forward(self, x1, x2):

        xb = x1
        x1f = self.featureExtractor(x1)
        x2f = self.featureExtractor(x2)

        for i in range(self.iteration):

            xbf = self.featureExtractor(xb)

            adaptiveFilter, shiftKernel = self.kernelMaking(x1f, x2f, xbf)

            shiftedX1 = self.shifter(x1, shiftKernel[0])
            shiftedX2 = self.shifter(x2, shiftKernel[1])
            shiftedXB = self.shifter(xb, shiftKernel[2])

            catedX = torch.cat((shiftedX1, shiftedX2, shiftedXB), 1)
            filterdX = self.applyFilter(catedX, adaptiveFilter)

            xb = filterdX

        return xb




class ShiftKernelMaker(nn.Module):
    def __init__(self):
        super(ShiftKernelMaker, self).__init__()

    def forward(self, shiftKernel):
        assert shiftKernel.size(2) % 2 == 1
        assert shiftKernel.size(3) % 2 == 1

        #Construct Channel
        shiftKernel = shiftKernel.view(1,-1,*shiftKernel.size()[2:])
        #print(shiftKernel.size())
        skList = []
        for i in range(shiftKernel.size(1)):
            skSlice = shiftKernel[:,i:i+1,:,:]
            skSlice = (skSlice == skSlice.max()).type_as(skSlice)
            skList.append(skSlice)
        skTensor = torch.cat(skList, 0)
        #skTensor = skTensor.unsqueeze(2)#.permute(1,0,2,3)

        return skTensor



class Shifter(nn.Module):
    
    def __init__(self):
        super(Shifter, self).__init__()
        self.shiftKernelMaker = ShiftKernelMaker()

    def forward(self, x, shiftKernel):

        shiftKernel = self.shiftKernelMaker(shiftKernel)
        #print(shiftKernel[0,0,:,:])

        N = x.size(0)
        C = x.size(1)

        #print(x.size(), shiftKernel.size())
        assert x.size(1)*x.size(0) == shiftKernel.size(0)
        assert x.size(2) >= shiftKernel.size(2)
        assert x.size(3) >= shiftKernel.size(3)
        assert shiftKernel.size(2) % 2 == 1
        assert shiftKernel.size(3) % 2 == 1

        x = x.view(1,-1,*x.size()[2:])
        paddingSize = shiftKernel.size(2)//2
        x = F.conv2d(x, shiftKernel, padding=paddingSize, groups=x.size(1))
        x = x.view(N,C,*x.size()[2:])
        return x


#sk1 = torch.randn(2,3,5,5)
#sk2 = ShiftKernelMaker()(sk1)
#inp = torch.randn(2,3,5,5)
#out = Shifter()(inp, sk2)
#print(sk1)
#print(sk2)
#print(inp)
#print(out)
#




class UMP(nn.Module):

    def __init__(self, CW):
        super(UMP, self).__init__()

        self.CW = CW
        self.DecoderList = nn.ModuleList([ # 1/32
            nn.ConvTranspose2d(2560,  512, 4, 2, 1), #1/16
            nn.ConvTranspose2d( 512,  256, 4, 2, 1), #1/8
            nn.ConvTranspose2d( 256,  128, 4, 2, 1), #1/4
            nn.ConvTranspose2d( 128,  32, 4, 2, 1), #1/2
            nn.ConvTranspose2d( 32, inputChannel, 4, 2, 1), #1/1
        ])

    def forward(self, x_a_f, x_b_f):

        x = torch.cat([x_a_f, x_b_f], 1)

        for i, decoder in enumerate(self.DecoderList):
            x = decoder(x)
            if i + 1 < len(self.DecoderList):    
                x = F.relu(x)

        return F.tanh(x)


class PSY(nn.Module):
    
    def __init__(self, CW, Blocks = 5, ResPerBlocks = 10, Attention = True, attention_sub_sample = None):
        super(PSY, self).__init__()

        self.CW = CW
        self.Blocks = Blocks
        self.ResPerBlocks = ResPerBlocks
        self.Attention = Attention

        self.ReconstructionBlocks = nn.ModuleList()
        for i in range(self.Blocks):
            self.ReconstructionBlocks.append(ReconstructionBlock(CW, ResPerBlocks, dim=2, use_attention=Attention, attention_sub_sample=None))

        self.Encoder = nn.Conv2d(inputChannel * 2, CW, 4, 2, 1)
        self.Decoder = nn.ConvTranspose2d(CW, inputChannel * 2, 4, 2, 1)


    def forward(self, x):

        ori = x[:,:,:,:]
        
        x = self.Encoder(x)
        x = F.relu(x)
        
        for i in range(self.Blocks):
            x = self.ReconstructionBlocks[i](x)
        
        x = self.Decoder(x)
        x = x.view(x.size(0), x.size(1) // inputChannel, inputChannel, *x.size()[2:])
        x = F.softmax(x, dim=1)#.view(x.size(0), -1, *x.size()[2:])

        ori = ori.view(ori.size(0), ori.size(1) // inputChannel, inputChannel, *ori.size()[2:])

        return (x * ori).sum(dim=1)



class WENDY(nn.Module):
    
    def __init__(self, CW, Blocks = 5, ResPerBlocks = 10, Attention = True, attention_sub_sample = None):
        super(WENDY, self).__init__()

        self.CW = CW
        self.Blocks = Blocks
        self.ResPerBlocks = ResPerBlocks
        self.Attention = Attention

        self.ReconstructionBlocks = nn.ModuleList()
        for i in range(self.Blocks):
            self.ReconstructionBlocks.append(ReconstructionBlock(CW, ResPerBlocks, dim=2, use_attention=Attention, attention_sub_sample=None))

        self.Encoder = nn.Conv2d(inputChannel * 2, CW, 4, 2, 1)
        self.Decoder = nn.ConvTranspose2d(CW, inputChannel, 4, 2, 1)


    def forward(self, x):

        sc = x[:,0:3,:,:]
        x = self.Encoder(x)
        x = F.relu(x)
        
        for i in range(self.Blocks):
            x = self.ReconstructionBlocks[i](x)
        
        x = self.Decoder(x)
        x = F.tanh(x)

        return x + sc


        
    










class IRENE(nn.Module):

    def __init__(self, scaleFactor, nframes, CW, SLRBs, ILRBs, basicRBs, center = None, attention_sub_sample = None):
        super(IRENE, self).__init__()

        assert scaleFactor in [2,4,8,16]

        self.nframes = nframes
        if center is not None:
            self.center = center
        else:
            self.center = self.nframes // 2

        self.PCDAlignModule = Align(CW, front_RBs=5, center=self.center)
        self.SequenceLevelReconstructionModule = nn.ModuleList()
        self.FusionModule = nn.ModuleList()
        self.ImageLevelReconstructionModule = nn.ModuleList()
        self.UpScalingModule = nn.ModuleList()
        self.scaleFactor = scaleFactor
        self.CW = CW
        self.SLRBs = SLRBs
        self.ILRBs = ILRBs


        self.recCount = int(math.log(scaleFactor, 2))


        for i in range(self.recCount):

            self.SequenceLevelReconstructionModule.append(nn.ModuleList())
            for j in range(self.SLRBs):
                self.SequenceLevelReconstructionModule[i].append(ReconstructionBlock(CW=CW, basicRBs = basicRBs, dim = 3, attention_sub_sample = attention_sub_sample))

            self.FusionModule.append(Fusion(nframes = self.nframes, CW = self.CW))

            self.ImageLevelReconstructionModule.append(nn.ModuleList())
            for j in range(self.ILRBs):
                self.ImageLevelReconstructionModule[i].append(ReconstructionBlock(CW=CW, basicRBs = basicRBs,dim = 2, attention_sub_sample = attention_sub_sample))

            if i == self.recCount:
                self.UpScalingModule.append(Upscale(CW = self.CW, scaleFactor=2**(i)))
            else:
                self.UpScalingModule.append(Upscale(CW = self.CW, scaleFactor=2**(i+1)))



    def doubleScale(self, alignedFeatureSeq, UltraLongSkipConnection, scaleLevel):

        assert scaleLevel in range(self.recCount)
        # Seq. Level Recon.
        reconstructedFeatureSeq = alignedFeatureSeq
        #sharedSource = alignedFeatureSeq

        #TEST
        #print("IN", scaleLevel, reconstructedFeatureSeq.data.mean(), reconstructedFeatureSeq.data.max(), reconstructedFeatureSeq.data.min())

        for j in range(self.SLRBs):
            reconstructedFeatureSeq = self.SequenceLevelReconstructionModule[scaleLevel][j](reconstructedFeatureSeq)# + sharedSource
        reconstructedFeatureSeq += alignedFeatureSeq

        #TEST
        #print("SR", scaleLevel, reconstructedFeatureSeq.data.mean(), reconstructedFeatureSeq.data.max(), reconstructedFeatureSeq.data.min())

        # Fusion
        fusionedFeature = self.FusionModule[scaleLevel](reconstructedFeatureSeq)

        #TEST
        #print("FS", scaleLevel, fusionedFeature.data.mean(), fusionedFeature.data.max(), fusionedFeature.data.min())

        # Img. Level Recon.
        reconstructedFeature = fusionedFeature
        #sharedSource = fusionedFeature
        for j in range(self.ILRBs):
            reconstructedFeature = self.ImageLevelReconstructionModule[scaleLevel][j](reconstructedFeature)# + sharedSource
        reconstructedFeature += fusionedFeature

        #TEST
        #print("IR", scaleLevel, reconstructedFeature.data.mean(), reconstructedFeature.data.max(), reconstructedFeature.data.min())
        
        # UPSCALE
        upscaledImage = self.UpScalingModule[scaleLevel](reconstructedFeature) + F.interpolate(UltraLongSkipConnection, scale_factor=2, mode='bilinear', align_corners=False)

        #TEST
        #print("UP", scaleLevel, upscaledImage.data.mean(), upscaledImage.data.max(), upscaledImage.data.min())

        return upscaledImage

    def textureRestoration(self, alignedFeatureSeq, SkipConnection):

        # Seq. Level Recon.
        reconstructedFeatureSeq = alignedFeatureSeq
        #sharedSource = alignedFeatureSeq

        #TEST
        #print("IN", -1, reconstructedFeatureSeq.data.mean(), reconstructedFeatureSeq.data.max(), reconstructedFeatureSeq.data.min())

        for j in range(self.SLRBs):
            reconstructedFeatureSeq = self.SequenceLevelReconstructionModule[self.recCount][j](reconstructedFeatureSeq)# + sharedSource
        reconstructedFeatureSeq += alignedFeatureSeq

        #TEST
        #print("SR", -1, reconstructedFeatureSeq.data.mean(), reconstructedFeatureSeq.data.max(), reconstructedFeatureSeq.data.min())

        # Fusion
        fusionedFeature = self.FusionModule[self.recCount](reconstructedFeatureSeq)

        #TEST
        #print("FS", -1, fusionedFeature.data.mean(), fusionedFeature.data.max(), fusionedFeature.data.min())

        # Img. Level Recon.
        reconstructedFeature = fusionedFeature
        #sharedSource = fusionedFeature
        for j in range(self.ILRBs):
            reconstructedFeature = self.ImageLevelReconstructionModule[self.recCount][j](reconstructedFeature)# + sharedSource
        reconstructedFeature += fusionedFeature

        #TEST
        #print("IR", -1, reconstructedFeature.data.mean(), reconstructedFeature.data.max(), reconstructedFeature.data.min())
        
        # UPSCALE
        upscaledImage = self.UpScalingModule[self.recCount](reconstructedFeature) + SkipConnection

        #TEST
        #print("UP", -1, upscaledImage.data.mean(), upscaledImage.data.max(), upscaledImage.data.min())

        return upscaledImage

    def forward(self, lawInput = None, alignedFeatureSeq = None, UltraLongSkipConnection = None, scaleLevel = None):

        #print(self.ImageLevelReconstructionModule)
        assert ((lawInput is None and alignedFeatureSeq is not None and UltraLongSkipConnection is not None) or
               (lawInput is not None and alignedFeatureSeq is None and UltraLongSkipConnection is None))
        assert ((scaleLevel in range(self.recCount)) or scaleLevel is None or scaleLevel is 'TR')


        if lawInput is not None:
            #x = torch.cat(x.split(1, dim=1),2).squeeze(1)

            # PCD Align
            alignedFeatureSeq = self.PCDAlignModule(lawInput).permute(0,2,1,3,4) # # [B, N, C, H, W] -> [B, C, N, H, W]
            #alignedFeatureSeq = lawInput.permute(0,2,1,3,4)
            
            UltraLongSkipConnection = lawInput[:,self.center,:,:,:]
            

        if scaleLevel == None:
            for i in range(self.recCount):
                # X2 Upscale
                upscaledImage = self.doubleScale(alignedFeatureSeq, UltraLongSkipConnection, i)
                UltraLongSkipConnection = upscaledImage
        elif scaleLevel == 'TR':
            upscaledImage = self.textureRestoration(alignedFeatureSeq, UltraLongSkipConnection)
        else:
            # X2 Upscale
            upscaledImage = self.doubleScale(alignedFeatureSeq, UltraLongSkipConnection, scaleLevel)
        #print("")
        return alignedFeatureSeq, upscaledImage




class ReconstructionBlock(nn.Module):
    def __init__(self, CW, basicRBs, dim=2, use_attention=True, attention_sub_sample=None):
        super(ReconstructionBlock, self).__init__()

        assert dim in [2,3]

        self.dim = dim

        self.use_attention = use_attention

        if self.use_attention is True:
            self.AB = AttentionBlock(CW, dim=self.dim, sub_sample=attention_sub_sample)
        self.RB = basicResBlocks(CW, basicRBs, dim=self.dim)

        

    def forward(self, x):
        
        if self.use_attention is True:
            x = self.AB(x)
        x = self.RB(x)

        return x

class basicResBlocks(nn.Module):
    def __init__(self, CW, Blocks, kernelSize=3, dim=2):
        super(basicResBlocks, self).__init__()

        assert dim in [2,3]

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

    def forward(self, x):

        for i in range(self.Blocks):
            res = x
            x = self.Convs[i](x)
            x = self.IN(x)
            x = x + res
            x = self.act(x)

        return x

class AttentionBlock(nn.Module):
    def __init__(self, CW, dim=2, sub_sample=None):
        super(AttentionBlock, self).__init__()

        assert dim in [2,3]
        self.dim = dim
        self.CW = CW

        self.CA = SecondOrderChannalAttentionBlock(self.CW, dim=self.dim, sub_sample=sub_sample)
        self.NL1 = CrissCrossAttention(CW, dim=self.dim, r=2)
        self.NL2 = CrissCrossAttention(CW, dim=self.dim, r=2)

    def forward(self, x):

        
        x = self.NL1(x)
        x = self.NL2(x)
        x = self.CA(x)

        return x

class Align(nn.Module):
    def __init__(self, CW, front_RBs, center):
        super(Align, self).__init__()

        self.center = center

        self.conv_first = nn.Conv2d(3, CW, 3, 1, 1, bias=True)
        ResidualBlock_noBN_f = functools.partial(EDVR_ResidualBlock_noBN, nf=CW)
        self.feature_extraction = EDVR_make_layer(ResidualBlock_noBN_f, front_RBs)
        
        self.fea_L2_conv1 = nn.Conv2d(CW, CW, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(CW, CW, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(CW, CW, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(CW, CW, 3, 1, 1, bias=True)

        self.pcd_align = EDVR_PCD_Align(nf=CW)
        

        self.act = nn.LeakyReLU(0.2)
        


    def forward(self, x):

        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        L1_fea = self.act(self.conv_first(x.view(-1, C, H, W)))

        L1_fea = self.feature_extraction(L1_fea)
        
        # L2
        L2_fea = self.act(self.fea_L2_conv1(L1_fea))
        L2_fea = self.act(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.act(self.fea_L3_conv1(L2_fea))
        L3_fea = self.act(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]
        
        

        return aligned_fea#L1_fea.view(B, N, -1, H ,W)
        


class Fusion(nn.Module):
    def __init__(self, nframes, CW):
        super(Fusion, self).__init__()

        self.conv = nn.Conv2d(nframes * CW, CW, 1, 1, bias=True)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):

        x = x.reshape(x.size(0),-1,x.size(3),x.size(4))
        x = self.act(self.conv(x))

        return x

class Upscale(nn.Module):
    def __init__(self, CW, scaleFactor):
        super(Upscale, self).__init__()

        assert scaleFactor in [2,4,8,16]
        self.scaleFactor = scaleFactor
        self.recCount = int(math.log(scaleFactor,2))

        self.upconv = nn.ModuleList()
        for i in range(self.recCount):
            self.upconv.append(nn.Conv2d(CW, CW * 4, 3, 1, 1, bias=True))

        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(CW, CW, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(CW, 3, 3, 1, 1, bias=True)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):

        for i in range(self.recCount):
            x = self.act(self.pixel_shuffle(self.upconv[i](x)))
        x = self.act(self.HRconv(x))
        x = self.conv_last(x)

        return x













def ResNeSt(tp, **kwargs):
    assert tp in ['50', '101', '200', '269']
    if tp == '50': #224
        return ResNet(Bottleneck, [3, 4, 6, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    elif tp == '101': #256
        return ResNet(Bottleneck, [3, 4, 23, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    elif tp == '200': #320
        return ResNet(Bottleneck, [3, 24, 36, 3],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)
    elif tp == '269': #416
        return ResNet(Bottleneck, [3, 30, 48, 8],
                   radix=2, groups=1, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, **kwargs)







class EfficientNet(nn.Module):
    """ (Generic) EfficientNet
    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """ 

    def __init__(self, name, num_classes=1000, num_features=1280, in_chans=3, stem_size=32,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', fix_stem=False, act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg', mode='classifier'):
                 #mode : 'classifier' || 'feature_extractor' || 'perceptual'
        super(EfficientNet, self).__init__()


        assert name in ['b0','b1','b2','b3','b4','b5','b6','b7','b8','l2']
        assert mode in ['classifier' , 'feature_extractor' , 'perceptual']

        arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
        ]


        if name == 'b0':
            channel_multiplier = 1.0
            depth_multiplier = 1.0
            input_res = 224
            drop_rate = 0.2
        if name == 'b1':
            channel_multiplier = 1.0
            depth_multiplier = 1.1
            input_res = 240
            drop_rate = 0.2
        if name == 'b2':
            channel_multiplier = 1.1
            depth_multiplier = 1.2
            input_res = 260
            drop_rate = 0.3
        if name == 'b3':
            channel_multiplier = 1.2
            depth_multiplier = 1.4
            input_res = 300
            drop_rate = 0.3
        if name == 'b4':
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            input_res = 380
            drop_rate = 0.4
        if name == 'b5':
            channel_multiplier = 1.6
            depth_multiplier = 2.2
            input_res = 456
            drop_rate = 0.4
        if name == 'b6':
            channel_multiplier = 1.8
            depth_multiplier = 2.6
            input_res = 528
            drop_rate = 0.5
        if name == 'b7':
            channel_multiplier = 2.0
            depth_multiplier = 3.1
            input_res = 600
            drop_rate = 0.5
        if name == 'b8':
            channel_multiplier = 2.2
            depth_multiplier = 3.6
            input_res = 672
            drop_rate = 0.5
        if name == 'l2':
            channel_multiplier = 4.3
            depth_multiplier = 5.3
            input_res = 800
            drop_rate = 0.5

        self.input_res = input_res
        block_args = EFF_decode_arch_def(arch_def, depth_multiplier)

        norm_kwargs = norm_kwargs or {}

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self._in_chs = in_chans



        self.mode = mode

        # Stem
        if not fix_stem:
            stem_size = EFF_round_channels(stem_size, channel_multiplier, channel_divisor, channel_min)
        self.conv_stem = EFF_create_conv2d(self._in_chs, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size, **norm_kwargs)
        self.act1 = act_layer(inplace=True)
        self._in_chs = stem_size

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=False)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = EFF_create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        efficientnet_init_weights(self)
    
        

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(
            self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.mode == 'classifier' or self.mode == 'feature_extractor':
            x = self.blocks(x)
            x = self.conv_head(x)
            x = self.bn2(x)
            x = self.act2(x)
            return x
        elif self.mode == 'perceptual':
            rstList = []
            for blkmdl in self.blocks:
                x = blkmdl(x)
                rstList.append(x.mean().unsqueeze(0))
            return torch.sum(torch.cat(rstList, 0))


    def forward(self, x):

        if self.mode in ['classifier']:
            assert x.size(2) == self.input_res
            assert x.size(3) == self.input_res

        x = self.forward_features(x)

        if self.mode == 'classifier':
            x = self.global_pool(x)
            x = x.flatten(1)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            return F.sigmoid(self.classifier(x))

        elif self.mode == 'perceptual' or self.mode == 'feature_extractor':
            return x





class EDVR(nn.Module):
    def __init__(self, nf=64, nframes=7, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(EDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
        self.nframes = nframes
        ResidualBlock_noBN_f = functools.partial(EDVR_ResidualBlock_noBN, nf=nf)

        #### extract features (for each frame)
        if self.is_predeblur:
            self.pre_deblur = EDVR_Predeblur_ResNet_Pyramid(nf=nf, HR_in=self.HR_in)
            self.conv_1x1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        else:
            if self.HR_in:
                self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
                self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
                self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            else:
                self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = EDVR_make_layer(ResidualBlock_noBN_f, front_RBs)
        self.fea_L2_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L2_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.fea_L3_conv1 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.fea_L3_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.pcd_align = EDVR_PCD_Align(nf=nf, groups=groups)
        if self.w_TSA:
            self.tsa_fusion = EDVR_TSA_Fusion(nf=nf, nframes=nframes, center=self.center)
        else:
            self.tsa_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        #### reconstruction
        self.recon_trunk = EDVR_make_layer(ResidualBlock_noBN_f, back_RBs)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.unsqueeze(1).repeat(1,self.nframes,1,1,1)
        B, N, C, H, W = x.size()  # N video frames
        x_center = x[:, self.center, :, :, :].contiguous()

        #### extract LR features
        # L1
        if self.is_predeblur:
            L1_fea = self.pre_deblur(x.view(-1, C, H, W))
            L1_fea = self.conv_1x1(L1_fea)
            if self.HR_in:
                H, W = H // 4, W // 4
        else:
            if self.HR_in:
                L1_fea = self.lrelu(self.conv_first_1(x.view(-1, C, H, W)))
                L1_fea = self.lrelu(self.conv_first_2(L1_fea))
                L1_fea = self.lrelu(self.conv_first_3(L1_fea))
                H, W = H // 4, W // 4
            else:
                L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction(L1_fea)
        # L2
        L2_fea = self.lrelu(self.fea_L2_conv1(L1_fea))
        L2_fea = self.lrelu(self.fea_L2_conv2(L2_fea))
        # L3
        L3_fea = self.lrelu(self.fea_L3_conv1(L2_fea))
        L3_fea = self.lrelu(self.fea_L3_conv2(L3_fea))

        L1_fea = L1_fea.view(B, N, -1, H, W)
        L2_fea = L2_fea.view(B, N, -1, H // 2, W // 2)
        L3_fea = L3_fea.view(B, N, -1, H // 4, W // 4)

        #### pcd align
        # ref feature list
        ref_fea_l = [
            L1_fea[:, self.center, :, :, :].clone(), L2_fea[:, self.center, :, :, :].clone(),
            L3_fea[:, self.center, :, :, :].clone()
        ]
        aligned_fea = []
        for i in range(N):
            nbr_fea_l = [
                L1_fea[:, i, :, :, :].clone(), L2_fea[:, i, :, :, :].clone(),
                L3_fea[:, i, :, :, :].clone()
            ]
            aligned_fea.append(self.pcd_align(nbr_fea_l, ref_fea_l))
        aligned_fea = torch.stack(aligned_fea, dim=1)  # [B, N, C, H, W]

        if not self.w_TSA:
            aligned_fea = aligned_fea.view(B, -1, H, W)
        fea = self.tsa_fusion(aligned_fea)

        out = self.recon_trunk(fea)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out)
        if self.HR_in:
            base = x_center
        else:
            base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base
        return out


 


class resBlock(nn.Module):
    
    def __init__(self, channelDepth, windowSize=5, inputCD=None):
        
        super(resBlock, self).__init__()
        if inputCD == None:
            inputCD = channelDepth
        padding = math.floor(windowSize/2)
        self.conv1 = nn.Conv2d(inputCD, channelDepth, windowSize, 1, padding)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding)
        self.conv3 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding)

                      
    def forward(self, x):

        res = x
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = self.conv3(x + res)
        
        return x
    

class VESPCN(nn.Module):
    def __init__(self, upscale_factor=p.scaleFactor):
        super(VESPCN, self).__init__()
        
        inputCh = (1 if p.colorMode =='grayscale' else 3) * p.sequenceLength

        self.conv1 = nn.Conv2d(inputCh, 256, (9, 9), (1, 1), (4, 4))

        self.res1 = resBlock(256, windowSize=3)
        self.res2 = resBlock(256, windowSize=3)

        self.conv2 = nn.Conv2d(256, 128, (5, 5), (1, 1), (2, 2))

        self.res3 = resBlock(128, windowSize=3)
        self.res4 = resBlock(128, windowSize=3)
        self.res5 = resBlock(128, windowSize=3)
        self.res6 = resBlock(128, windowSize=3)

        self.res7 = resBlock(128, windowSize=3)
        self.res8 = resBlock(128, windowSize=3)
        self.res9 = resBlock(128, windowSize=3)
        self.res10 = resBlock(128, windowSize=3)

        self.conv3 = nn.Conv2d(128, (1 if p.colorMode =='grayscale' else 3) * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        sc = x[:,p.sequenceLength//2,:,:,:]
        x = torch.cat(x.split(1, dim=1),2).squeeze(1)

        x = F.leaky_relu(self.conv1(x), 0.2)

        res = x
        x = F.leaky_relu(self.res1(x), 0.2)
        x = F.leaky_relu(self.res2(x) + res, 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)
        
        res = x
        x = F.leaky_relu(self.res3(x), 0.2)
        x = F.leaky_relu(self.res4(x), 0.2)
        x = F.leaky_relu(self.res5(x), 0.2)
        x = F.leaky_relu(self.res6(x) + res, 0.2)

        res = x
        x = F.leaky_relu(self.res7(x), 0.2)
        x = F.leaky_relu(self.res8(x), 0.2)
        x = F.leaky_relu(self.res9(x), 0.2)
        x = F.leaky_relu(self.res10(x) + res, 0.2)
        
        x = F.tanh(self.pixel_shuffle(self.conv3(x))) + F.interpolate(sc, scale_factor=p.scaleFactor, mode='bicubic')
        return x


class ESPCN(nn.Module):
    def __init__(self, upscale_factor=p.scaleFactor):
        super(ESPCN, self).__init__()

        self.conv1 = nn.Conv2d(inputChannel, 256, (9, 9), (1, 1), (4, 4))

        self.res1 = resBlock(256, windowSize=3)
        self.res2 = resBlock(256, windowSize=3)

        self.conv2 = nn.Conv2d(256, 128, (5, 5), (1, 1), (2, 2))

        self.res3 = resBlock(128, windowSize=3)
        self.res4 = resBlock(128, windowSize=3)
        self.res5 = resBlock(128, windowSize=3)
        self.res6 = resBlock(128, windowSize=3)

        self.res7 = resBlock(128, windowSize=3)
        self.res8 = resBlock(128, windowSize=3)
        self.res9 = resBlock(128, windowSize=3)
        self.res10 = resBlock(128, windowSize=3)

        self.conv3 = nn.Conv2d(128, inputChannel * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        sc = x
        x = F.leaky_relu(self.conv1(x), 0.2)

        res = x
        x = F.leaky_relu(self.res1(x), 0.2)
        x = F.leaky_relu(self.res2(x) + res, 0.2)

        x = F.leaky_relu(self.conv2(x), 0.2)
        
        res = x
        x = F.leaky_relu(self.res3(x), 0.2)
        x = F.leaky_relu(self.res4(x), 0.2)
        x = F.leaky_relu(self.res5(x), 0.2)
        x = F.leaky_relu(self.res6(x) + res, 0.2)

        res = x
        x = F.leaky_relu(self.res7(x), 0.2)
        x = F.leaky_relu(self.res8(x), 0.2)
        x = F.leaky_relu(self.res9(x), 0.2)
        x = F.leaky_relu(self.res10(x) + res, 0.2)
        
        x = F.tanh(self.pixel_shuffle(self.conv3(x))) + F.interpolate(sc, scale_factor=p.scaleFactor, mode='bicubic')
        return (x + 1)/2 if p.valueRangeType == '0~1' else x


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.VDSR_make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def VDSR_make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return 0*out + residual




class D_class(nn.Module):

    def __init__(self):
        super(D_class, self).__init__()

        self.conv1 = nn.Conv2d(p.NDF * 8, p.NDF * 4, 4, 2, 1)  # 32->16
        self.conv2 = nn.Conv2d(p.NDF * 4, p.NDF * 2, 4, 2, 1)  # 16->8
        self.conv3 = nn.Conv2d(p.NDF * 2, p.NDF * 1, 4, 2, 1)  # 8->4
        self.conv4 = nn.Conv2d(p.NDF * 1, 1, 2, 1, 0)  # 4->1

        self.IN_conv1 = nn.InstanceNorm2d(p.NGF * 4)
        self.IN_conv2 = nn.InstanceNorm2d(p.NGF * 2)
        self.IN_conv3 = nn.InstanceNorm2d(p.NGF * 1)

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

    def __init__(self):
        super(D_AE_class, self).__init__()

        self.conv1 = nn.Conv2d(3, p.NDF * 1, 4, 2, 1)  # 32->16
        self.conv2 = nn.Conv2d(p.NDF * 1, p.NDF * 2, 4, 2, 1)  # 32->16
        self.conv3 = nn.Conv2d(p.NDF * 2, p.NDF * 4, 4, 2, 1)  # 32->16
        self.conv4 = nn.Conv2d(p.NDF * 4, p.NDF * 4, 4, 2, 1)  # 32->16
        self.conv5 = nn.Conv2d(p.NDF * 4, p.NDF * 2, 4, 2, 1)  # 16->8
        self.conv6 = nn.Conv2d(p.NDF * 2, p.NDF * 1, 4, 2, 1)  # 8->4
        self.conv7 = nn.Conv2d(p.NDF * 1, 1, 4, 1, 0)  # 4->1

        self.IN_conv1 = nn.InstanceNorm2d(p.NGF * 4)
        self.IN_conv2 = nn.InstanceNorm2d(p.NGF * 2)
        self.IN_conv3 = nn.InstanceNorm2d(p.NGF * 1)

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

    def __init__(self):
        super(TiedAE, self).__init__()

        self.conv1 = nn.Conv2d(1, p.NDF * 1, 4, 2, 1)  # 256->128
        self.conv2 = nn.Conv2d(p.NDF * 1, p.NDF * 2, 4, 2, 1)  # 128->64
        self.conv3 = nn.Conv2d(p.NDF * 2, p.NDF * 4, 4, 2, 1)  # 64->32
        self.conv4 = nn.Conv2d(p.NDF * 4, p.NDF * 8, 4, 2, 1)  # 32->16
        self.conv5 = nn.Conv2d(p.NDF * 4, p.NDF * 8, 4, 2, 1)  # ->8
        self.conv6 = nn.Conv2d(p.NDF * 8, p.NDF * 16, 4, 2, 1)  # ->4
        self.conv7 = nn.Conv2d(p.NDF * 16, p.NDF * 32, 4, 1, 0)  # ->1

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
    
    def __init__(self):
        super(TiedDisc, self).__init__()

        self.conv1 = nn.Conv2d(        3, p.NDF *   1, 4, 2, 1) # 256 -> 128
        self.conv2 = nn.Conv2d(p.NDF *   1, p.NDF *   2, 4, 2, 1) # 256 -> 128
        self.conv3 = nn.Conv2d(p.NDF *   2, p.NDF *   4, 4, 2, 1) # 256 -> 128
        self.conv4 = nn.Conv2d(p.NDF *   4, p.NDF *   8, 4, 2, 1) # 256 -> 128
        self.conv5 = nn.Conv2d(p.NDF *   8, p.NDF *  16, 4, 2, 1) # 256 -> 128
        self.conv6 = nn.Conv2d(p.NDF *  16, p.NDF *  32, 4, 2, 1) # 256 -> 128
        self.conv7 = nn.Conv2d(p.NDF *  32,         1, 4, 1, 0) # 256 -> 128
        

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

    def __init__(self):
        super(TiedGAN, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(p.NDF * 32, p.NDF * 16, 4, 2, 1)  #   4   4   256  ->  8   8   128
        self.deconv2 = nn.ConvTranspose2d(p.NDF * 16, p.NDF *  8, 4, 2, 1)  #   8   8   128  ->  16  16  64
        self.deconv3 = nn.ConvTranspose2d(p.NDF *  8, p.NDF *  4, 4, 2, 1)  #   16  16  64   ->  32  32  32
        self.deconv4 = nn.ConvTranspose2d(p.NDF *  4, p.NDF *  2, 4, 2, 1)  #   32  32  32   ->  64  64  16
        self.deconv5 = nn.ConvTranspose2d(p.NDF *  2, p.NDF *  1, 4, 2, 1)  #   64  64  16   ->  128 128 8
        self.deconv6 = nn.ConvTranspose2d(p.NDF *  1,        3, 4, 2, 1)  #   128 128 8    ->  256 256 3

        self.conv1 = nn.Conv2d(        3, p.NDF *   1, 4, 2, 1) # 256 -> 128
        self.conv2 = nn.Conv2d(p.NDF *   1, p.NDF *   2, 4, 2, 1) # 256 -> 128
        self.conv3 = nn.Conv2d(p.NDF *   2, p.NDF *   4, 4, 2, 1) # 256 -> 128
        self.conv4 = nn.Conv2d(p.NDF *   4, p.NDF *   8, 4, 2, 1) # 256 -> 128
        self.conv5 = nn.Conv2d(p.NDF *   8, p.NDF *  16, 4, 2, 1) # 256 -> 128
        self.conv6 = nn.Conv2d(p.NDF *  16, p.NDF *  32, 4, 2, 1) # 256 -> 128
        self.conv7 = nn.Conv2d(p.NDF *  32,         1, 4, 1, 0) # 256 -> 128

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