'''
model.py
'''
version = '1.2.200423'


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
from backbone.module import EDVR_Predeblur_ResNet_Pyramid, EDVR_PCD_Align, EDVR_TSA_Fusion, EDVR_make_layer, EDVR_ResidualBlock_noBN
    #Attentions
from backbone.module import SecondOrderChannalAttentionBlock, NonLocalBlock, CrissCrossAttention
    #EfficientNet
from backbone.module import EFF_MBConvBlock, EFF_round_filters, EFF_round_repeats, EFF_drop_connect, EFF_get_same_padding_conv2d, EFF_get_model_params, EFF_efficientnet_params, EFF_load_pretrained_weights, EFF_Swish, EFF_MemoryEfficientSwish



############################################

# 1. 2nd ORder CA
# 2. InstantNorm
# 3. EDVR
# 4. Spatial A

inputChannel = 1 if p.colorMode =='grayscale' else 3
















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
    def __init__(self, CW, basicRBs, dim=2, attention_sub_sample=None):
        super(ReconstructionBlock, self).__init__()

        assert dim in [2,3]

        self.dim = dim

        self.AB = AttentionBlock(CW, dim=self.dim, sub_sample=attention_sub_sample)
        self.RB = basicResBlocks(CW, basicRBs, dim=self.dim)

    def forward(self, x):
        
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


















class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = EFF_get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = EFF_round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=EFF_round_filters(block_args.input_filters, self._global_params),
                output_filters=EFF_round_filters(block_args.output_filters, self._global_params),
                num_repeat=EFF_round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(EFF_MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(EFF_MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = EFF_round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, 1)#self._global_params.num_classes)
        self._swish = EFF_MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = EFF_MemoryEfficientSwish() if memory_efficient else EFF_Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)


    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            EFF_drop_connect_rate = self._global_params.EFF_drop_connect_rate
            if EFF_drop_connect_rate:
                EFF_drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, EFF_drop_connect_rate=EFF_drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)

        ##추가
        x = F.sigmoid(x)
        #x = F.softmax(F.sigmoid(x), dim=1)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = EFF_get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels = 3):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        EFF_load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        if in_channels != 3:
            Conv2d = EFF_get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = EFF_round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model
    
    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        EFF_load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))

        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = EFF_efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))

























class EDVR(nn.Module):
    def __init__(self, nf=64, nframes=7, groups=8, front_RBs=5, back_RBs=10, center=None,
                 predeblur=False, HR_in=False, w_TSA=True):
        super(EDVR, self).__init__()
        self.nf = nf
        self.center = nframes // 2 if center is None else center
        self.is_predeblur = True if predeblur else False
        self.HR_in = True if HR_in else False
        self.w_TSA = w_TSA
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

        self.conv1 = nn.Conv2d(1, 256, (9, 9), (1, 1), (4, 4))

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

        self.conv3 = nn.Conv2d(128, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
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
        return x


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
        self.residual_layer = self.EDVR_make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    def EDVR_make_layer(self, block, num_of_layer):
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