'''
edit.py
'''
editversion = "1.20.201230"


#FROM Python LIBRARY
import time
import math
import numpy as np
import psutil
import random
from collections import OrderedDict

#FROM PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


#from iqa-pytorch
from IQA_pytorch import MS_SSIM, SSIM, GMSD, LPIPSvgg, DISTS



#from this project
import backbone.vision as vision
import model
import backbone.utils as utils
import backbone.structure as structure
import backbone.module.module as module
import backbone.predefined as predefined
from backbone.utils import loadModels, saveModels, backproagateAndWeightUpdate        
from backbone.config import Config
from backbone.module.SPSR.loss import GANLoss, GradientPenaltyLoss
from backbone.structure import Epoch


#from backbone.PULSE.stylegan import G_synthesis,G_mapping
#from backbone.PULSE.SphericalOptimizer import SphericalOptimizer
#from backbone.PULSE.loss import LossBuilder



################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)

version = 'ESPCN'
subversion = '1'

# VERSION END (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)
###############################################


#################################################
###############  EDIT THIS AREA  ################
#################################################



#################################################################################
#                                     MODEL                                     #
#################################################################################

class ModelList(structure.ModelListBase):
    def __init__(self):
        super(ModelList, self).__init__()


        ##############################################################
        # self.(모델이름)           :: model                   :: 필 수                     
        # self.(모델이름)_optimizer :: optimizer               :: 없어도됨
        # self.(모델이름)_scheduler :: Learning Rate Scheduler :: 없어도됨
        #-------------------------------------------------------------
        # self.(모델이름)_pretrained :: pretrained 파일 경로 :: ** /model/ 폴더 밑에 저장된 모델이 없을 시 OR optimizer 가 없을 시 ---> pretrained 경로에서 로드
        #
        # trainStep() 에서 사용 방법
        # modelList.(모델 인스턴스 이름)_optimizer
        ##############################################################


        self.NET = predefined.ESPCN(4)
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.0001)
        #self.NET_pretrained = "ESPCN-General-201007.pth"
        

        self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()



#################################################################################
#                                     STEPS                                     #
#################################################################################

def trainStep(epoch, modelList, dataDict):

    #define loss function
    mse_criterion = nn.MSELoss()

    #train mode
    modelList.NET.train()

    #define input data and label
    LRImages = dataDict['LR']
    HRImages = dataDict['HR']
    
    #inference
    SRImages = modelList.NET(LRImages)

    #calculate loss and backpropagation
    loss = mse_criterion(SRImages, HRImages)
    backproagateAndWeightUpdate(modelList, loss)

    #return values
    lossDict = {'MSE': loss}
    SRImagesDict = {'SR' : SRImages}
    
    return lossDict, SRImagesDict
     


def validationStep(epoch, modelList, dataDict):

    #define loss function
    mse_criterion = nn.MSELoss()

    #eval mode
    modelList.NET.eval()

    #no grad for validation
    with torch.no_grad():
        #define input data and label
        LRImages = dataDict['LR']
        HRImages = dataDict['HR']
        
        #inference
        SRImages = modelList.NET(LRImages)

        #calculate loss
        loss = mse_criterion(SRImages, HRImages)

        #return values
        lossDict = {'MSE': loss}
        SRImagesDict = {'SR' : SRImages}
    
    return lossDict, SRImagesDict

def inferenceStep(epoch, modelList, dataDict):

    #eval mode
    modelList.NET.eval()

    #no grad for inference
    with torch.no_grad():

        #define input data
        LRImages = dataDict['LR']

        #inference
        SRImages = modelList.NET(LRImages)

        #return values
        SRImagesDict = {'SR': SRImages} 
        
    return {}, SRImagesDict



#################################################
###############  EDIT THIS AREA  ################
#################################################
