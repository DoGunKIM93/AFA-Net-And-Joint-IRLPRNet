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

#SPSR
import backbone.module.SPSR.architecture as SPSR_arch



# Generator
def SPSR_Generator(scaleFactor, inputChannel=3, outputChannel=3, nf=64, nb=23, gc=32):

    netG = SPSR_arch.SPSRNet(in_nc=inputChannel, out_nc=outputChannel, nf=nf,
        nb=nb, gc=gc, upscale=scaleFactor, norm_type=None,
        act_type='leakyrelu', mode="CNA", upsample_mode='upconv')

    return netG


# Discriminator
def SPSR_Discriminator(size):
    
        
    which_model = 'discriminator_vgg_' + str(size)

    if which_model == 'discriminator_vgg_128':
        netD = SPSR_arch.Discriminator_VGG_128(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_96':
        netD = SPSR_arch.Discriminator_VGG_96(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_64':
        netD = SPSR_arch.Discriminator_VGG_64(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_32':
        netD = SPSR_arch.Discriminator_VGG_32(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_16':
        netD = SPSR_arch.Discriminator_VGG_16(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_192':
        netD = SPSR_arch.Discriminator_VGG_192(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_128_SN':
        netD = SPSR_arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


def SPSR_FeatureExtractor(nf=34, use_bn=False, use_input_norm=True):
    # pytorch pretrained VGG19-54, before ReLU.
    #if use_bn:
    #    feature_layer = 49
    #else:
    #    feature_layer = 34
    #print("netF start")
    #netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True)

    VGG = SPSR_arch.VGGFeatureExtractor(feature_layer=nf, use_bn=False, use_input_norm=True)

    return VGG



class SPSR_Get_gradient(nn.Module):
    def __init__(self):
        super(SPSR_Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class SPSR_Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(SPSR_Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0], 
                    [0, 0, 0], 
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0], 
                    [-1, 0, 1], 
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


