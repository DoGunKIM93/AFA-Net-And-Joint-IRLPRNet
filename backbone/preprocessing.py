'''
preprocessing.py
'''


#FROM Python LIBRARY
import os
import random
import math
import numpy as np

from PIL import Image
from PIL import PngImagePlugin
from PIL.JpegImagePlugin import JpegImageFile


#FROM PyTorch
import torch

from torchvision import transforms
from torchvision import datasets




######################################################################################################################################################################## 

# PIL Image Preprocessings

# base : torchvision.transforms

######################################################################################################################################################################## 


from torchvision.transforms import ToTensor


class Pass(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return x

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
class To1DLabel(object):
    def __init__(self, img_dim = 640, rgb_means = (123, 117, 104)):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, x):
        ## input : Labels - LabelsPerImage
        ## output : 1DtensorLabels
        if type(x) == JpegImageFile:
            return x
        else :
            self.targets = x
            OriLabels = self.targets
            annotations = np.zeros((0, 15))

            if len(OriLabels) == 0:
                return annotations
            for idx, label in enumerate(OriLabels):
                annotation = np.zeros((1, 15))
                # bbox
                annotation[0, 0] = label[0]  # x1
                annotation[0, 1] = label[1]  # y1
                annotation[0, 2] = label[0] + label[2]  # x2
                annotation[0, 3] = label[1] + label[3]  # y2

                # landmarks
                annotation[0, 4] = label[4]    # l0_x
                annotation[0, 5] = label[5]    # l0_y
                annotation[0, 6] = label[7]    # l1_x
                annotation[0, 7] = label[8]    # l1_y
                annotation[0, 8] = label[10]   # l2_x
                annotation[0, 9] = label[11]   # l2_y
                annotation[0, 10] = label[13]  # l3_x
                annotation[0, 11] = label[14]  # l3_y
                annotation[0, 12] = label[16]  # l4_x
                annotation[0, 13] = label[17]  # l4_y
                if (annotation[0, 4]<0):
                    annotation[0, 14] = -1
                else:
                    annotation[0, 14] = 1

                annotations = np.append(annotations, annotation, axis=0)
            target = np.array(annotations)

            return torch.from_numpy(target)