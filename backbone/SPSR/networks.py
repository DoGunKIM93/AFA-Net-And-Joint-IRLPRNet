import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

import backbone.SPSR.architecture as arch

logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        if m.affine != False:

            init.constant_(m.weight.data, 1.0)
            init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G():
    dataComp = Config.paramDict['data']['dataLoader']['validation']['datasetComponent']
    scaleF = Config.paramDict['data']['datasetComponent'][dataComp[0]]['classParameter']['scaleFactor']

    netG = arch.SPSRNet(in_nc=3, out_nc=3, nf=64,
        nb=23, gc=32, upscale=scaleF[0], norm_type=None,
        act_type='leakyrelu', mode="CNA", upsample_mode='upconv')
    return netG



# Discriminator
def define_D(size):
    
        
    which_model = 'discriminator_vgg_' + str(size)

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_64':
        netD = arch.Discriminator_VGG_64(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_32':
        netD = arch.Discriminator_VGG_32(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_16':
        netD = arch.Discriminator_VGG_16(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

def define_D_grad(size):

    which_model = "discriminator_vgg_" + str(size)

    if which_model == 'discriminator_vgg_128':
        netD = arch.Discriminator_VGG_128(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_64':
        netD = arch.Discriminator_VGG_64(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_32':
        netD = arch.Discriminator_VGG_32(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_16':
        netD = arch.Discriminator_VGG_16(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=3, base_nf=64, \
            norm_type="batch", mode="CNA", act_type="leakyrelu")
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


def define_F(VGG):
    # pytorch pretrained VGG19-54, before ReLU.
    #if use_bn:
    #    feature_layer = 49
    #else:
    #    feature_layer = 34
    #print("netF start")
    #netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, use_input_norm=True)
    netF = VGG
    netF.eval()  
    return netF
