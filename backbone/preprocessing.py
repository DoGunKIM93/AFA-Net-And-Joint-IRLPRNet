'''
preprocessing.py
'''
version = '1.0.200706'


#FROM Python LIBRARY
import os
import random
import math
import numpy as np

from PIL import Image
from PIL import PngImagePlugin


#FROM PyTorch
import torch

from torchvision import transforms
from torchvision import datasets




######################################################################################################################################################################## 

# PIL Image Preprocessings

# base : torchvision.transforms

######################################################################################################################################################################## 


from torchvision.transforms import ToTensor

class To1Channel(object):

    def __init__(self):
        pass

    def __call__(self, x):

        return 

class To3Channel(object):

    def __init__(self):
        pass

    def __call__(self, x):

        return 

class To4Channel(object):

    def __init__(self):
        pass

    def __call__(self, x):

        return 




######################################################################################################################################################################## 

# Text Preprocessings

######################################################################################################################################################################## 