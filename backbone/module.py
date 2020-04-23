'''
module.py
'''
version = '1.0.200423'

#from Python
import time
import csv
import os
import math
import numpy as np
import sys
import functools
import re
import collections
from shutil import copyfile
from functools import partial


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
from torch.utils import model_zoo

#from this project
import param as p
import backbone.vision as vision
from backbone.dcn.deform_conv import ModulatedDeformConvPack as DCN






######################################################################################################################################################################## 

# EFFicientNET Modules

######################################################################################################################################################################## 








class EFF_MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = EFF_get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = EFF_MemoryEfficientSwish()

    def forward(self, inputs, EFF_drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param EFF_drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if EFF_drop_connect_rate:
                x = EFF_drop_connect(x, p=EFF_drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = EFF_MemoryEfficientSwish() if memory_efficient else EFF_Swish()









########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'EFF_drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class EFF_SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class EFF_MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return EFF_SwishImplementation.apply(x)

class EFF_Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def EFF_round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def EFF_round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def EFF_drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def EFF_get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def EFF_efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 EFF_drop_connect_rate=0.2, image_size=None, num_classes=2):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        EFF_drop_connect_rate=EFF_drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def EFF_get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = EFF_efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
}


def EFF_load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))









######################################################################################################################################################################## 

# Attention BLocks

######################################################################################################################################################################## 





## second-order Channel attention (SOCA)
class SecondOrderChannalAttentionBlock(nn.Module):
    def __init__(self, channel, dim=2, reduction=8, sub_sample=None):
        super(SecondOrderChannalAttentionBlock, self).__init__()

        assert dim in [2,3]

        self.dim = dim
        self.sub_sample = sub_sample



        if self.dim == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=2)

            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )
        elif self.dim == 3:
            self.max_pool = nn.MaxPool3d(kernel_size=2)

            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                nn.Conv3d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

    def forward(self, input):

        if self.sub_sample is None:
            x = input
        else:
            imode = 'bicubic'
            #x = F.interpolate(input.reshape(input.size(0), -1, *input.size()[-2:]), scale_factor=1/self.sub_sample, mode=imode, align_corners=False)
            #x = x.reshape(*input.size()[0:-2],*x.size()[-2:])
            x = F.max_pool3d(input, kernel_size=[1,self.sub_sample,self.sub_sample]) if self.dim == 3 else F.max_pool2d(input, kernel_size=[self.sub_sample,self.sub_sample])

        if self.dim==2:
            batch_size, C, h, w = x.shape  # x: BxCxHxW
            N = int(h * w)
        elif self.dim==3:
            batch_size, C, s, h, w = x.shape  # x: BxCxNxHxW
            N = int(s * h * w)


        if self.dim==2:

            ## MPN-COV
            cov_mat = CovpoolLayer(x) # Global Covariance pooling layer
            #cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
            ##
            cov_mat_sum = torch.mean(cov_mat,1)
            cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1)

        elif self.dim==3:
            

            ## MPN-COV
            cov_mat = CovpoolLayer3d(x) # Global Covariance pooling layer
            #cov_mat_sqrt = SqrtmLayer(cov_mat,5) # Matrix square root layer( including pre-norm,Newton-Schulz iter. and post-com. with 5 iteration)
            #cov_mat_sqrt = torch.sqrt(cov_mat/10000)  ######### NaN 문제로 삭제
            #print("  CA cov_sq", cov_mat_sqrt.mean())
            ##
            cov_mat_sum = torch.mean(cov_mat,1)
            cov_mat_sum = cov_mat_sum.view(batch_size,C,1,1,1)

        

        y_cov = self.conv_du(cov_mat_sum)


        return y_cov*input




## non_local module
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, r=8, mode='second_order_embedded_gaussian',
                 sub_sample=None, bn_layer=True):
        super(NonLocalBlock, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['second_order_embedded_gaussian']

        # print('Dimension: %d, mode: %s' % (dimension, mode))

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // r
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            #sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None
        
        if mode in ['second_order_embedded_gaussian']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'second_order_embedded_gaussian':
                self.operation_function = self._second_order_embedded_gaussian

        '''
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
        '''

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    # https://arxiv.org/pdf/1909.00295.pdf
    def _second_order_embedded_gaussian(self, input):

        if self.sub_sample is None:
            x = input
        else:
            imode = 'bicubic'
            #x = F.interpolate(input.reshape(input.size(0), -1, *input.size()[-2:]), scale_factor=1/self.sub_sample, mode=imode, align_corners=False)
            #x = x.reshape(*input.size()[0:-2],*x.size()[-2:])
            x = F.max_pool3d(input, kernel_size=[1,self.sub_sample,self.sub_sample]) if self.dimension == 3 else F.max_pool2d(input, kernel_size=[self.sub_sample,self.sub_sample])

        if self.dimension == 2:
            batch_size, C, H, W = x.shape
            M = H * W
        elif self.dimension == 3:
            batch_size, C, N, H, W = x.shape
            M = N * H * W

        #print(M, self.inter_channels)
        #print(self.g)
        ##
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        #g_x = g_x.permute(0, 2, 1)

        #print(g_x.size())


        theta_x_T = self.theta(x).view(batch_size, self.inter_channels, -1) # b, 0.5c, thw
        theta_x = theta_x_T.permute(0, 2, 1) # b, thw, 0.5c

        I_hat = (-1. / M / M) * torch.ones(M , M , device = x.device) + (1. / M) * torch.eye(M , M , device = x.device)
        I_hat = I_hat.view(1 , M , M).repeat(batch_size , 1 , 1).type(x.dtype) # b, thw, thw

        sigma = theta_x_T.bmm(I_hat).bmm(theta_x) # b, 0.5c, 0.5c
        sigma /= math.sqrt(self.inter_channels)
        sigma_div_C = F.softmax(sigma, dim=-1)
        
        # return f_div_C
        # (b, 0.5c, 0.5c)dot(b, thw, 0.5c) = (b, thw, 0.5c)->(b, 0.5c, t, h, w)->(b, c, t, h, w)
        y = torch.matmul(sigma_div_C, g_x)


        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        


        if self.sub_sample is None:
            z = W_y + input
        else:
            z = F.interpolate(W_y.reshape(W_y.size(0), -1, *W_y.size()[-2:]), size = input.size()[-2:], mode=imode, align_corners=False).reshape(input.size()) + input 

        return z



class CrissCrossAttention(nn.Module):
    def __init__(self,CW,r,dim=2):
        super(CrissCrossAttention, self).__init__()
        

        self.dim = dim

        if self.dim == 2:
            self.query_conv = nn.Conv2d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.key_conv = nn.Conv2d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.value_conv = nn.Conv2d(in_channels=CW, out_channels=CW, kernel_size=1)
            self.softmax = nn.Softmax(dim=3)
            self.IN = nn.InstanceNorm2d(CW)
        elif self.dim == 3:
            self.query_conv = nn.Conv3d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.key_conv = nn.Conv3d(in_channels=CW, out_channels=CW//r, kernel_size=1)
            self.value_conv = nn.Conv3d(in_channels=CW, out_channels=CW, kernel_size=1)
            self.softmax = nn.Softmax(dim=4)
            self.IN = nn.InstanceNorm3d(CW)

        
        self.gamma = nn.Parameter(torch.zeros(1))
        

    def INF(self,B,H,W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1) 

    def INF3dH(self,B,S,H,W):
        return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*S*W,1,1) 

    def INF3dS(self,B,S,H,W):
        rtn = -torch.diag(torch.tensor(float("inf")).cuda().repeat(S),0).unsqueeze(0).repeat(B*H*W,1,1) 
        return rtn

    def CC3d(self, x):
        m_batchsize, _, seq, height, width = x.size()
        #print("MBSZ", m_batchsize)
        proj_query = self.query_conv(x)
        #print("CC3 1", proj_query.mean(), proj_query.max(), proj_query.min())

        proj_query_S = proj_query.permute(0,3,4,1,2).contiguous().view(m_batchsize * height * width, -1, seq).permute(0, 2, 1)
        proj_query_H = proj_query.permute(0,2,4,1,3).contiguous().view(m_batchsize * width * seq, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,3,1,4).contiguous().view(m_batchsize * height * seq, -1, width).permute(0, 2, 1)

        proj_key = self.key_conv(x)
        #print("CC3 2", proj_key.mean(), proj_key.max(), proj_key.min())

        proj_key_S = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize * height * width, -1, seq)
        proj_key_H = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize * width * seq, -1, height)
        proj_key_W = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize * height * seq, -1, width)

        proj_value = self.value_conv(x)
        #print("CC3 3", proj_value.mean(), proj_value.max(), proj_value.min())

        proj_value_S = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize * height * width, -1, seq)
        proj_value_H = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize * width * seq, -1, height)
        proj_value_W = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize * height * seq, -1, width)

        #print("CC3 3-1", proj_value_S.mean(), proj_value_S.max(), proj_value_S.min())
        #print("CC3 3-1", proj_value_H.mean(), proj_value_H.max(), proj_value_H.min())

        #print("CC3 3-1", proj_value_W.mean(), proj_value_W.max(), proj_value_W.min())

        energy_S = (torch.matmul(proj_query_S, proj_key_S)+self.INF3dS(m_batchsize, seq, height, width)).view(m_batchsize,height,width,seq,seq).permute(0,3,1,2,4)      
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF3dH(m_batchsize, seq, height, width)).view(m_batchsize,seq,width,height,height).permute(0,1,3,2,4)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,seq,height,width,width)


        concate = self.softmax(torch.cat([energy_S, energy_H, energy_W], 4))
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()

        #print("CC3 3-2", energy_S.mean(), energy_S.max(), energy_S.min())
        #print("CC3 3-2", energy_H.mean(), energy_H.max(), energy_H.min())
        #print("CC3 3-2", energy_W.mean(), energy_W.max(), energy_W.min())

        att_S = concate[:,:,:,:,0:seq].permute(0,2,3,1,4).contiguous().view(m_batchsize*width*height,seq,seq)
        att_H = concate[:,:,:,:,seq:seq+height].permute(0,1,3,2,4).contiguous().view(m_batchsize*seq*width,height,height)
        att_W = concate[:,:,:,:,seq+height:seq+height+width].contiguous().view(m_batchsize*seq*height,width,width)

        #print("CC3 4", att_S.mean(), att_S.max(), att_S.min())
        #print("CC3 4", att_H.mean(), att_H.max(), att_H.min())
        #print("CC3 4", att_W.mean(), att_W.max(), att_W.min())

        maxMBSize = 40960

        proj_value_S_Chunks = proj_value_S.split(maxMBSize, dim=0)
        att_S_Chunks = att_S.split(maxMBSize, dim=0)

        out_S_tmp = []
        for proj_value_S, att_S in zip(proj_value_S_Chunks, att_S_Chunks):
            out_S_tmp.append(torch.bmm(proj_value_S, att_S.permute(0, 2, 1)))
        out_S = torch.cat(out_S_tmp,0).view(m_batchsize,height,width,-1,seq).permute(0,3,4,1,2)

        #out_S = torch.bmm(proj_value_S, att_S.permute(0, 2, 1)).view(m_batchsize,height,width,-1,seq).permute(0,3,4,1,2)#torch.bmm(proj_value_S, att_S.permute(0, 2, 1)).view(m_batchsize,height,width,-1,seq).permute(0,3,4,1,2)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,seq,width,-1,height).permute(0,3,1,4,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,seq,height,-1,width).permute(0,3,1,2,4)

        #print("CC3 5", out_S.mean(), out_S.max(), out_S.min())
        #print("CC3 5", out_H.mean(), out_H.max(), out_H.min())
        #print("CC3 5", out_W.mean(), out_W.max(), out_W.min())

        #print(out_H.size(),out_W.size())
        out_sum = self.IN(out_S + out_H + out_W)
        return self.gamma*(out_sum) + x

    def CC2d(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)

        #print("CC2 1", proj_query.mean())

        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)

        #print("CC2 2", proj_key.mean())

        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)

        #print("CC2 3", proj_value.mean())

        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        #print("CC2 4", concate.mean())

        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)

        #print("CC2 5", att_H.mean())

        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)

        #print("CC2 6", out_W.mean())

        #print(out_H.size(),out_W.size())
        out_sum = self.IN(out_W + out_H)
        return self.gamma*(out_sum) + x

    def forward(self, x):
        if self.dim == 2:
            return self.CC2d(x)
        elif self.dim == 3:
            return self.CC3d(x)









######################################################################################################################################################################## 

# EDVR Modules

######################################################################################################################################################################## 





class EDVR_Predeblur_ResNet_Pyramid(nn.Module):
    def __init__(self, nf=128, HR_in=False):
        '''
        HR_in: True if the inputs are high spatial size
        '''

        super(EDVR_Predeblur_ResNet_Pyramid, self).__init__()
        self.HR_in = True if HR_in else False
        if self.HR_in:
            self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
            self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
            self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        else:
            self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(EDVR_ResidualBlock_noBN, nf=nf)
        self.RB_L1_1 = basic_block()
        self.RB_L1_2 = basic_block()
        self.RB_L1_3 = basic_block()
        self.RB_L1_4 = basic_block()
        self.RB_L1_5 = basic_block()
        self.RB_L2_1 = basic_block()
        self.RB_L2_2 = basic_block()
        self.RB_L3_1 = basic_block()
        self.deblur_L2_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.deblur_L3_conv = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        if self.HR_in:
            L1_fea = self.lrelu(self.conv_first_1(x))
            L1_fea = self.lrelu(self.conv_first_2(L1_fea))
            L1_fea = self.lrelu(self.conv_first_3(L1_fea))
        else:
            L1_fea = self.lrelu(self.conv_first(x))
        L2_fea = self.lrelu(self.deblur_L2_conv(L1_fea))
        L3_fea = self.lrelu(self.deblur_L3_conv(L2_fea))
        L3_fea = F.interpolate(self.RB_L3_1(L3_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L2_fea = self.RB_L2_1(L2_fea) + L3_fea
        L2_fea = F.interpolate(self.RB_L2_2(L2_fea), scale_factor=2, mode='bilinear',
                               align_corners=False)
        L1_fea = self.RB_L1_2(self.RB_L1_1(L1_fea)) + L2_fea
        out = self.RB_L1_5(self.RB_L1_4(self.RB_L1_3(L1_fea)))
        return out


class EDVR_PCD_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(EDVR_PCD_Align, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff 
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.L3_dcnpack = EDVR_DeformConv2d(nf, nf, 3, stride=1, padding=1, deformable_groups=groups, extra_offset_mask=True, modulation=True) #inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False, extra_offset_mask=False):
        self.L3_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.L2_dcnpack = EDVR_DeformConv2d(nf, nf, 3, stride=1, padding=1, deformable_groups=groups, extra_offset_mask=True, modulation=True)
        self.L2_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #self.L1_dcnpack = EDVR_DeformConv2d(nf, nf, 3, stride=1, padding=1, deformable_groups=groups, extra_offset_mask=True, modulation=True)
        self.L1_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading EDVR_DeformConv2d
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        #self.cas_dcnpack = EDVR_DeformConv2d(nf, nf, 3, stride=1, padding=1, deformable_groups=groups, extra_offset_mask=True, modulation=True)
        self.cas_dcnpack = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups, extra_offset_mask=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea = self.lrelu(self.L3_dcnpack([nbr_fea_l[2], L3_offset]))
        # L2
        L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea = self.L2_dcnpack([nbr_fea_l[1], L2_offset])
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea = self.L1_dcnpack([nbr_fea_l[0], L1_offset])
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea = self.lrelu(self.cas_dcnpack([L1_fea, offset]))

        return L1_fea


class EDVR_TSA_Fusion(nn.Module):
    ''' Temporal Spatial Attention fusion module
    Temporal: correlation;
    Spatial: 3 pyramid levels.
    '''

    def __init__(self, nf=64, nframes=5, center=2):
        super(EDVR_TSA_Fusion, self).__init__()
        self.center = center
        # temporal attention (before fusion conv)
        self.tAtt_1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.tAtt_2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # fusion conv: using 1x1 to save parameters and computation
        self.fea_fusion = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)

        # spatial attention (after fusion conv)
        self.sAtt_1 = nn.Conv2d(nframes * nf, nf, 1, 1, bias=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(3, stride=2, padding=1)
        self.sAtt_2 = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.sAtt_3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_L1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_L2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.sAtt_L3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.sAtt_add_1 = nn.Conv2d(nf, nf, 1, 1, bias=True)
        self.sAtt_add_2 = nn.Conv2d(nf, nf, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, aligned_fea):
        B, N, C, H, W = aligned_fea.size()  # N video frames
        #### temporal attention
        emb_ref = self.tAtt_2(aligned_fea[:, self.center, :, :, :].clone())
        emb = self.tAtt_1(aligned_fea.view(-1, C, H, W)).view(B, N, -1, H, W)  # [B, N, C(nf), H, W]

        cor_l = []
        for i in range(N):
            emb_nbr = emb[:, i, :, :, :]
            cor_tmp = torch.sum(emb_nbr * emb_ref, 1).unsqueeze(1)  # B, 1, H, W
            cor_l.append(cor_tmp)
        cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))  # B, N, H, W
        cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
        aligned_fea = aligned_fea.view(B, -1, H, W) * cor_prob

        #### fusion
        fea = self.lrelu(self.fea_fusion(aligned_fea))

        #### spatial attention
        att = self.lrelu(self.sAtt_1(aligned_fea))
        att_max = self.maxpool(att)
        att_avg = self.avgpool(att)
        att = self.lrelu(self.sAtt_2(torch.cat([att_max, att_avg], dim=1)))
        # pyramid levels
        att_L = self.lrelu(self.sAtt_L1(att))
        att_max = self.maxpool(att_L)
        att_avg = self.avgpool(att_L)
        att_L = self.lrelu(self.sAtt_L2(torch.cat([att_max, att_avg], dim=1)))
        att_L = self.lrelu(self.sAtt_L3(att_L))
        att_L = F.interpolate(att_L, scale_factor=2, mode='bilinear', align_corners=False)

        att = self.lrelu(self.sAtt_3(att))
        att = att + att_L
        att = self.lrelu(self.sAtt_4(att))
        att = F.interpolate(att, scale_factor=2, mode='bilinear', align_corners=False)
        att = self.sAtt_5(att)
        att_add = self.sAtt_add_2(self.lrelu(self.sAtt_add_1(att)))
        att = torch.sigmoid(att)

        fea = fea * att * 2 + att_add
        return fea


class EDVR_DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, deformable_groups=8, modulation=False, extra_offset_mask=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(EDVR_DeformConv2d, self).__init__()
        self.extra_offset_mask = extra_offset_mask
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)


    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        
        if self.extra_offset_mask:
            # x = [input, features]
            out = x[1]
            x = x[0]
        else:
            out = x
        
        offset = self.p_conv(out)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(out))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset


def EDVR_initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def EDVR_make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class EDVR_ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(EDVR_ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        EDVR_initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out



def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'
    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output




















######################################################################################################################################################################## 

# Covariance pooling 

######################################################################################################################################################################## 


class Covpool(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         h = x.data.shape[2]
         w = x.data.shape[3]
         M = h*w
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,h,w)
         return grad_input

class Covpool3d(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         n = x.data.shape[2]
         h = x.data.shape[3]
         w = x.data.shape[4]
         M = h*w*n
         x = x.reshape(batchSize,dim,M)
         I_hat = (-1./M/M)*torch.ones(M,M,device = x.device) + (1./M)*torch.eye(M,M,device = x.device)
         I_hat = I_hat.view(1,M,M).repeat(batchSize,1,1).type(x.dtype)
         y = x.bmm(I_hat).bmm(x.transpose(1,2))
         ctx.save_for_backward(input,I_hat)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,I_hat = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         n = x.data.shape[2]
         h = x.data.shape[3]
         w = x.data.shape[4]
         M = h*w*n
         x = x.reshape(batchSize,dim,M)
         grad_input = grad_output + grad_output.transpose(1,2)
         grad_input = grad_input.bmm(x).bmm(I_hat)
         grad_input = grad_input.reshape(batchSize,dim,n,h,w)
         return grad_input

class Sqrtm(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,device = x.device).view(1,dim,dim).repeat(batchSize,iterN,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
            Z[:,0,:,:] = ZY
            for i in range(1, iterN-1):
                ZY = 0.5*(I3 - Z[:,i-1,:,:].bmm(Y[:,i-1,:,:]))
                Y[:,i,:,:] = Y[:,i-1,:,:].bmm(ZY)
                Z[:,i,:,:] = ZY.bmm(Z[:,i-1,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:].bmm(I3 - Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:].bmm(Z[:,iterN-2,:,:])) -
                          Z[:,iterN-2,:,:].bmm(Y[:,iterN-2,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:].bmm(Z[:,i,:,:])
               ZY = Z[:,i,:,:].bmm(Y[:,i,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:].bmm(dldZ).bmm(Z[:,i,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:].bmm(dldY).bmm(Y[:,i,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

class Sqrtm3d(Function):
     @staticmethod
     def forward(ctx, input, iterN):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         I3 = 3.0*torch.eye(dim,dim,device = x.device).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
         normA = (1.0/3.0)*x.mul(I3).sum(dim=1).sum(dim=1)
         A = x.div(normA.view(batchSize,1,1).expand_as(x))
         Y = torch.zeros(batchSize, iterN, dim, dim, dim, requires_grad = False, device = x.device)
         Z = torch.eye(dim,dim,dim,device = x.device).view(1,dim,dim,dim).repeat(batchSize,iterN,1,1,1)
         if iterN < 2:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:] = A.bmm(ZY)
         else:
            ZY = 0.5*(I3 - A)
            Y[:,0,:,:,:] = A.bmm(ZY)
            Z[:,0,:,:,:] = ZY
            for i in range(1, iterN-1):
               ZY = 0.5*(I3 - Z[:,i-1,:,:,:].bmm(Y[:,i-1,:,:,:]))
               Y[:,i,:,:,:] = Y[:,i-1,:,:,:].bmm(ZY)
               Z[:,i,:,:,:] = ZY.bmm(Z[:,i-1,:,:,:])
            ZY = 0.5*Y[:,iterN-2,:,:,:].bmm(I3 - Z[:,iterN-2,:,:,:].bmm(Y[:,iterN-2,:,:,:]))
         y = ZY*torch.sqrt(normA).view(batchSize, 1, 1, 1).expand_as(x)
         ctx.save_for_backward(input, A, ZY, normA, Y, Z)
         ctx.iterN = iterN
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input, A, ZY, normA, Y, Z = ctx.saved_tensors
         iterN = ctx.iterN
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         der_postCom = grad_output*torch.sqrt(normA).view(batchSize, 1, 1, 1).expand_as(x)
         der_postComAux = (grad_output*ZY).sum(dim=1).sum(dim=1).div(2*torch.sqrt(normA))
         I3 = 3.0*torch.eye(dim,dim,dim,device = x.device).view(1, dim, dim, dim).repeat(batchSize,1,1,1).type(dtype)
         if iterN < 2:
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace))
         else:
            dldY = 0.5*(der_postCom.bmm(I3 - Y[:,iterN-2,:,:,:].bmm(Z[:,iterN-2,:,:,:])) -
                          Z[:,iterN-2,:,:,:].bmm(Y[:,iterN-2,:,:,:]).bmm(der_postCom))
            dldZ = -0.5*Y[:,iterN-2,:,:,:].bmm(der_postCom).bmm(Y[:,iterN-2,:,:,:])
            for i in range(iterN-3, -1, -1):
               YZ = I3 - Y[:,i,:,:,:].bmm(Z[:,i,:,:,:])
               ZY = Z[:,i,:,:,:].bmm(Y[:,i,:,:,:])
               dldY_ = 0.5*(dldY.bmm(YZ) - 
                         Z[:,i,:,:,:].bmm(dldZ).bmm(Z[:,i,:,:,:]) - 
                             ZY.bmm(dldY))
               dldZ_ = 0.5*(YZ.bmm(dldZ) - 
                         Y[:,i,:,:,:].bmm(dldY).bmm(Y[:,i,:,:,:]) -
                            dldZ.bmm(ZY))
               dldY = dldY_
               dldZ = dldZ_
            der_NSiter = 0.5*(dldY.bmm(I3 - A) - dldZ - A.bmm(dldY))
         grad_input = der_NSiter.div(normA.view(batchSize,1,1,1).expand_as(x))
         grad_aux = der_NSiter.mul(x).sum(dim=1).sum(dim=1)
         for i in range(batchSize):
             grad_input[i,:,:,:] += (der_postComAux[i] \
                                   - grad_aux[i] / (normA[i] * normA[i])) \
                                   *torch.ones(dim,device = x.device).diag()
         return grad_input, None

class Triuvec(Function):
     @staticmethod
     def forward(ctx, input):
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         x = x.reshape(batchSize, dim*dim)
         I = torch.ones(dim,dim).triu().t().reshape(dim*dim)
         index = I.nonzero()
         y = torch.zeros(batchSize,dim*(dim+1)/2,device = x.device)
         for i in range(batchSize):
            y[i, :] = x[i, index].t()
         ctx.save_for_backward(input,index)
         return y
     @staticmethod
     def backward(ctx, grad_output):
         input,index = ctx.saved_tensors
         x = input
         batchSize = x.data.shape[0]
         dim = x.data.shape[1]
         dtype = x.dtype
         grad_input = torch.zeros(batchSize,dim,dim,device = x.device,requires_grad=False)
         grad_input = grad_input.reshape(batchSize,dim*dim)
         for i in range(batchSize):
            grad_input[i,index] = grad_output[i,:].reshape(index.size(),1)
         grad_input = grad_input.reshape(batchSize,dim,dim)
         return grad_input

def CovpoolLayer(var):
    return Covpool.apply(var)

def SqrtmLayer(var, iterN):
    return Sqrtm.apply(var, iterN)

def CovpoolLayer3d(var):
    return Covpool3d.apply(var)

def SqrtmLayer3d(var, iterN):
    return Sqrtm3d.apply(var, iterN)

def TriuvecLayer(var):
    return Triuvec.apply(var)

