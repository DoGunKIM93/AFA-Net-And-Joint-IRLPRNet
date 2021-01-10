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

# EfficientNet
from backbone.module.EfficientNet.EfficientNet import (
    EfficientNetBuilder,
    EFF_create_conv2d,
    EFF_round_channels,
    SelectAdaptivePool2d,
    efficientnet_init_weights,
    EFF_decode_arch_def,
)


class EfficientNet(nn.Module):
    """(Generic) EfficientNet
    A flexible and performant PyTorch implementation of efficient network architectures, including:
      * EfficientNet B0-B8, L2
      * EfficientNet-EdgeTPU
      * EfficientNet-CondConv
      * MixNet S, M, L, XL
      * MnasNet A1, B1, and small
      * FBNet C
      * Single-Path NAS Pixel1
    """

    def __init__(
        self,
        name,
        num_classes=1000,
        num_features=1280,
        in_chans=3,
        stem_size=32,
        channel_multiplier=1.0,
        channel_divisor=8,
        channel_min=None,
        output_stride=32,
        pad_type="",
        fix_stem=False,
        act_layer=nn.ReLU,
        drop_rate=0.0,
        drop_path_rate=0.0,
        se_kwargs=None,
        norm_layer=nn.BatchNorm2d,
        norm_kwargs=None,
        global_pool="avg",
        mode="classifier",
    ):
        # mode : 'classifier' || 'feature_extractor' || 'perceptual'
        super(EfficientNet, self).__init__()

        assert name in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "l2"]
        assert mode in ["classifier", "feature_extractor", "perceptual"]

        arch_def = [
            ["ds_r1_k3_s1_e1_c16_se0.25"],
            ["ir_r2_k3_s2_e6_c24_se0.25"],
            ["ir_r2_k5_s2_e6_c40_se0.25"],
            ["ir_r3_k3_s2_e6_c80_se0.25"],
            ["ir_r3_k5_s1_e6_c112_se0.25"],
            ["ir_r4_k5_s2_e6_c192_se0.25"],
            ["ir_r1_k3_s1_e6_c320_se0.25"],
        ]

        if name == "b0":
            channel_multiplier = 1.0
            depth_multiplier = 1.0
            input_res = 224
            drop_rate = 0.2
        if name == "b1":
            channel_multiplier = 1.0
            depth_multiplier = 1.1
            input_res = 240
            drop_rate = 0.2
        if name == "b2":
            channel_multiplier = 1.1
            depth_multiplier = 1.2
            input_res = 260
            drop_rate = 0.3
        if name == "b3":
            channel_multiplier = 1.2
            depth_multiplier = 1.4
            input_res = 300
            drop_rate = 0.3
        if name == "b4":
            channel_multiplier = 1.4
            depth_multiplier = 1.8
            input_res = 380
            drop_rate = 0.4
        if name == "b5":
            channel_multiplier = 1.6
            depth_multiplier = 2.2
            input_res = 456
            drop_rate = 0.4
        if name == "b6":
            channel_multiplier = 1.8
            depth_multiplier = 2.6
            input_res = 528
            drop_rate = 0.5
        if name == "b7":
            channel_multiplier = 2.0
            depth_multiplier = 3.1
            input_res = 600
            drop_rate = 0.5
        if name == "b8":
            channel_multiplier = 2.2
            depth_multiplier = 3.6
            input_res = 672
            drop_rate = 0.5
        if name == "l2":
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
            channel_multiplier,
            channel_divisor,
            channel_min,
            output_stride,
            pad_type,
            act_layer,
            se_kwargs,
            norm_layer,
            norm_kwargs,
            drop_path_rate,
            verbose=False,
        )
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

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes) if num_classes else None

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.mode == "classifier" or self.mode == "feature_extractor":
            x = self.blocks(x)
            x = self.conv_head(x)
            if self.mode == "classifier":
                x = self.bn2(x)
                x = self.act2(x)
            return x

        elif self.mode == "perceptual":
            rstList = []
            for blkmdl in self.blocks:
                x = blkmdl(x)
                rstList.append(x.mean().unsqueeze(0))
            return torch.sum(torch.cat(rstList, 0))

    def decode_features(self, x):
        x = self.conv_head_reverse(x)
        x = self.blocks_reverse(x)

        x = self.act1_reverse(x)
        x = self.bn1_reverse(x)
        x = self.conv_stem_reverse(x)

    def forward(self, x):

        if self.mode in ["classifier"]:
            assert x.size(2) == self.input_res
            assert x.size(3) == self.input_res

        x = self.forward_features(x)

        if self.mode == "classifier":
            x = self.global_pool(x)
            x = x.flatten(1)
            if self.drop_rate > 0.0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            return F.sigmoid(self.classifier(x))

        elif self.mode == "perceptual" or self.mode == "feature_extractor":
            return x
