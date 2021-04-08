'''
module.py
'''

#from Python
import time
import csv
import os
import math
import numpy as np
import sys
import warnings

#from Pytorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable,grad
from torch.autograd import Function
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

#from this project
from backbone.config import Config
import backbone.vision as vision





#####################################################################################################################

# Non-Uniform Operators

#####################################################################################################################


def nonUniformConv2d(inp, weight, padding=0, padding_mode='constant'):
    '''
    Non-Uniform Convolution Funcfion
    input : 4-dim image tensor (N,C,H,W)
    weight : 7-dim tensor (N, out_ch, in_ch, weight_height, weight_width, kernel_height, kernel_width)

    expected weight size:
        height: H - kernel_height + 1 + 2*pad
        width:  W - kernel_width  + 1 + 2*pad
    '''
    assert len(inp.size()) == 4
    N, C, H, W = inp.size()


    assert len(weight.size()) == 7
    wN, oC, iC, wH, wW, kH, kW = weight.size() 
    sH = H - kH + 1 + 2*padding
    sW = W - kW + 1 + 2*padding

    assert N == wN
    assert C == iC
    assert sH == wH
    assert sW == wW

    assert padding_mode in ['constant', 'reflect', 'replicate', 'circular']


    inp = inp if padding == 0 else F.pad(inp, (padding, padding, padding, padding), mode=padding_mode)

    weight = weight.permute(0,3,4,1,2,5,6).contiguous()
    weight = weight.view(N * sH * sW, oC, iC, kH, kW)

    weight = weight.view(N * sH * sW, weight.size(1), -1)
    weight = weight.permute(0,2,1)


    inp_unf = F.unfold(inp, (kH, kW))
    out_unf = inp_unf.transpose(1, 2)

    out_unf = out_unf.reshape(N * sH * sW, 1, C * kH * kW)
    out_unf = out_unf.bmm(weight)
    out_unf = out_unf.view(N, sH * sW, C)

    out_unf = out_unf.transpose(1, 2)

    out = out_unf.view(N, C, sH, sW)
    return out


def wienerDeconvoltionOperator(inp, weight):
    #g = IDFT( DFT(K)* / (DFT(K)DFT(K)* + sn/sx) )
    kH, kW = weight.size()[-2:]
    H, W = inp.size()[-2:]

    windowedInp = torch.cat([F.unfold(inp[:,i:i+1,:,:], (kH,kW), padding=(kH//2, kW//2)).unsqueeze(1) for i in range(inp.size(1))], 1).permute(0,1,3,2)
    windowedInp = windowedInp.view(*windowedInp.size()[:-1],kH,kW)

    meanedInp = F.interpolate(F.avg_pool2d(inp, kernel_size=3), size=inp.size()[-2:], mode='bicubic')
    windowedMeanedInp = torch.cat([F.unfold(meanedInp[:,i:i+1,:,:], (kH,kW), padding=(kH//2, kW//2)).unsqueeze(1) for i in range(meanedInp.size(1))], 1).permute(0,1,3,2)
    windowedMeanedInp = windowedMeanedInp.view(*windowedMeanedInp.size()[:-1],kH,kW)

    sx = torch.std(windowedInp, dim=(-1,-2), keepdim=True).unsqueeze(1).view(1,1,1,H,W,1,1)
    sn = torch.var(windowedMeanedInp - windowedInp, dim=(-1,-2), keepdim=True).unsqueeze(1).view(1,1,1,H,W,1,1)
    f_weight = torch.fft.fft2(weight)
    G = torch.fft.ifft2( f_weight.conj() / (f_weight.conj()*f_weight + sn/sx) )
    
    return G.real



def nonUniformInverseConv2d(inp, weight, pad=0, stride=1):
    '''
    Non-Uniform inverse Convolution Funcfion
    input : 4-dim image tensor (N,C,H,W)
    weight : 7-dim tensor (N, out_ch, in_ch, weight_height, weight_width, kernel_height, kernel_width)

    expected weight size:
        height: H - kernel_height + 1 + 2*pad
        width:  W - kernel_width  + 1 + 2*pad
    '''
    assert len(inp.size()) == 4
    N, C, H, W = inp.size()


    assert len(weight.size()) == 7
    wN, oC, iC, wH, wW, kH, kW = weight.size() 
    sH = H - kH + 1 + 2*pad
    sW = W - kW + 1 + 2*pad

    assert N == wN
    assert C == iC == oC == 1
    assert math.ceil(sH / stride) == wH
    assert math.ceil(sW / stride) == wW


    inp = inp if pad == 0 else F.pad(inp, (pad, pad, pad, pad))

    NORM = 'backward'

    # kernel Fourier transform
    #weight_f_mag, weight_f_pha = vision.polarFFT(weight)
    #weight_f_mag = weight_f_mag.view(-1, kH, kW)
    weight_f = torch.fft.fft2(weight).view(-1, kH, kW)#.view(N, -1, kH, kW)

    #print(weight_f_mag.max(), weight_f_mag.min())

    #unfold it
    inp_unf = F.unfold(inp, (kH, kW), stride=stride)

    norm_map = F.fold(F.unfold(torch.ones_like(inp), kernel_size=(kH,kW), stride=stride), (sH, sW), kernel_size=(kH,kW), padding=pad, stride=stride)

    # input fourier transform
    inp_unf = inp_unf.view(N, kH, kW, -1).permute(0,3,1,2)
    #inp_unf_f_mag, inp_unf_f_pha = vision.polarFFT(inp_unf)
    #inp_unf_f_mag = inp_unf_f_mag.view(-1, kH, kW)
    inp_unf_f = torch.fft.fft2(inp_unf).view(-1, kH, kW)
    #print(inp_unf_f_mag.max(), inp_unf_f_mag.min())

    # apply kernel in freq. domain

    #out_unf_f_mag = inp_unf_f_mag * (1 / (weight_f_mag + 1e-12))
    #out_unf_f_mag = out_unf_f_mag.view(N, C * H * W, kH, kW)
    out_unf_f = (inp_unf_f * (1/ (weight_f + 1e-12))).view(N, C * wH * wW, kH, kW)

    # bring it real world
    #out_unf = vision.polarIFFT(out_unf_f_mag, inp_unf_f_pha).real
    out_unf = torch.fft.ifft2(out_unf_f).real
    out_unf = out_unf.view(N, C * wH * wW, kH * kW)
    out_unf = out_unf.permute(0,2,1)

    out = F.fold(out_unf, (sH, sW), (kH, kW), padding=pad, stride=stride) / norm_map



    return out


def getDegradationKernel(inp, gt, kernelSize):

    assert len(inp.size()) == 4
    N, C, H, W = inp.size()

    assert inp.size() == gt.size()

    kH = kernelSize
    kW = kernelSize
    pad = kernelSize // 2


    inp = inp if pad == 0 else F.pad(inp, (pad, pad, pad, pad))

    NORM = 'backward'

    #unfold it
    inp_unf = F.unfold(inp, (kH, kW))

    norm_map = F.fold(F.unfold(torch.ones_like(inp), kernel_size=(kH,kW)), (sH, sW), kernel_size=(kH,kW), padding=pad)

    # input fourier transform
    inp_unf = inp_unf.view(N, kH, kW, -1).permute(0,3,1,2)
    inp_unf_f_mag, inp_unf_f_pha = vision.polarFFT(inp_unf)
    inp_unf_f_mag = inp_unf_f_mag.view(-1, kH, kW)

    # gt fourier transform
    gt_unf = gt_unf.view(N, kH, kW, -1).permute(0,3,1,2)
    gt_unf_f_mag, gt_unf_f_pha = vision.polarFFT(gt_unf)
    gt_unf_f_mag = gt_unf_f_mag.view(-1, kH, kW)


    # get degradation kernel
    out_unf_f_mag = inp_unf_f_mag * (1 / (weight_f_mag + 1e-12))
    out_unf_f_mag = out_unf_f_mag.view(N, C * H * W, kH, kW)

    # bring it real world
    out_unf = vision.polarIFFT(out_unf_f_mag, inp_unf_f_pha).real
    out_unf = out_unf.view(N, C * H * W, kH * kW)
    out_unf = out_unf.permute(0,2,1)

    out = F.fold(out_unf, (sH, sW), (kH, kW), padding=pad) / norm_map


    return out













######################################################################################################################################################################## 

# Basic Blocks

######################################################################################################################################################################## 

class resBlock(nn.Module):
    
    def __init__(self, channelDepth, windowSize=5, inputCD=None, outAct=None):
        
        super(resBlock, self).__init__()
        self.isOxO = False
        if inputCD == None:
            inputCD = channelDepth
        elif inputCD != channelDepth:
            self.oxo = nn.Conv2d(inputCD, channelDepth, 1, 1, 0)
            self.isOxO = True
        padding = math.floor(windowSize/2)
        self.conv1 = nn.Conv2d(inputCD, channelDepth, windowSize, 1, padding)
        self.conv2 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding)
        self.conv3 = nn.Conv2d(channelDepth, channelDepth, windowSize, 1, padding)
        self.outAct = outAct

                      
    def forward(self, x):

        res = x
        x = F.leaky_relu(self.conv1(x),0.2)
        x = F.leaky_relu(self.conv2(x),0.2)
        x = self.conv3(x + res if self.isOxO is False else x + self.oxo(res))
        
        return x if self.outAct is None else self.outAct(x)
    


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        





######################################################################################################################################################################## 

# Learning Rate Schedulers

######################################################################################################################################################################## 


class NotOneCycleLR(_LRScheduler):
    r"""Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: "cos" for cosine annealing, "linear" for
            linear annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1

    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """
    def __init__(self,
                 optimizer,
                 max_lr,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 pct_start=0.3,
                 anneal_strategy='cos',
                 cycle_momentum=True,
                 base_momentum=0.85,
                 max_momentum=0.95,
                 div_factor=25.,
                 final_div_factor=1e4,
                 last_epoch=-1):

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected non-negative integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected non-negative integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected non-negative integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch
        self.step_size_up = float(pct_start * self.total_steps) - 1
        self.step_size_down = float(self.total_steps - self.step_size_up) - 1

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))

        # Validate anneal_strategy
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError("anneal_strategy must by one of 'cos' or 'linear', instead got {}".format(anneal_strategy))
        elif anneal_strategy == 'cos':
            self.anneal_func = self._annealing_cos
        elif anneal_strategy == 'linear':
            self.anneal_func = self._annealing_linear

        # Initialize learning rate variables
        max_lrs = self._format_param('max_lr', self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group['initial_lr'] = max_lrs[idx] / div_factor
                group['max_lr'] = max_lrs[idx]
                group['min_lr'] = group['initial_lr'] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if 'momentum' not in self.optimizer.defaults and 'betas' not in self.optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')
            self.use_beta1 = 'betas' in self.optimizer.defaults
            max_momentums = self._format_param('max_momentum', optimizer, max_momentum)
            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        _, beta2 = group['betas']
                        group['betas'] = (m_momentum, beta2)
                    else:
                        group['momentum'] = m_momentum
                    group['max_momentum'] = m_momentum
                    group['base_momentum'] = b_momentum

        super(NotOneCycleLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", DeprecationWarning)

        lrs = []
        step_num = self.last_epoch % self.total_steps

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        for group in self.optimizer.param_groups:
            if step_num <= self.step_size_up:
                computed_lr = self.anneal_func(group['initial_lr'], group['max_lr'], step_num / self.step_size_up)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['max_momentum'], group['base_momentum'],
                                                         step_num / self.step_size_up)
            else:
                down_step_num = step_num - self.step_size_up
                computed_lr = self.anneal_func(group['max_lr'], group['min_lr'], down_step_num / self.step_size_down)
                if self.cycle_momentum:
                    computed_momentum = self.anneal_func(group['base_momentum'], group['max_momentum'],
                                                         down_step_num / self.step_size_down)

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    _, beta2 = group['betas']
                    group['betas'] = (computed_momentum, beta2)
                else:
                    group['momentum'] = computed_momentum

        return lrs









######################################################################################################################################################################## 

# Loss Functions

######################################################################################################################################################################## 



class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):# eps=1e-6 # eps=1e-3 for MPRNet
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        #loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps))) #for MPRNet
        
        return loss


class EdgeLoss(nn.Module):

    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)      # filter
        down        = filtered[:,:,::2,::2]         # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4            # upsample
        filtered    = self.conv_gauss(new_filter)   # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss



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
            der_NSiter = 0.5*(der_postCom.bmm(I3 - A) - A.bmm(der_sacleTrace)) #
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


######################################################################################################################################################################## 

# Image pre-processing

######################################################################################################################################################################## 

def multiplePadding(x, multipleSize, mode = 'reflect'):
    """
    padding mode
    'constant', # value – fill value for 'constant' padding. Default: 0 
    'reflect', 
    'replicate', 
    'circular'
    """

    _, cH, cW = list(x.size())[-3:]

    H,W = ((cH + multipleSize) // multipleSize) * multipleSize, ((cW + multipleSize) // multipleSize) * multipleSize
    padh = H-cH if cH%multipleSize!=0 else 0
    padw = W-cW if cW%multipleSize!=0 else 0
    x = F.pad(x, (0, padw, 0, padh), mode)

    return x
