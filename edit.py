'''
edit.py
'''


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
from dataLoader import DataLoader


#from backbone.PULSE.stylegan import G_synthesis,G_mapping
#from backbone.PULSE.SphericalOptimizer import SphericalOptimizer
#from backbone.PULSE.loss import LossBuilder



################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)

version = '54-gomtang'
subversion = '0-test'

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

        self.NET = predefined.DeFiAN(64, 20, 20, 4)
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.0003)
        self.NET_pretrained = "DeFiAN_L_x4.pth"
        
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

    #SR
    SRImages = modelList.NET(dataDict['LR'])

    #calculate loss and backpropagation
    loss = mse_criterion(SRImages, dataDict['GT'])
    backproagateAndWeightUpdate(modelList, loss, modelNames='NET')

    #return values
    lossDict = {'mse': loss}
    resultImagesDict = {'SR': SRImages}
    
    return lossDict, resultImagesDict
     


def validationStep(epoch, modelList, dataDict):


    #define loss function
    mse_criterion = nn.MSELoss()

    #eval mode
    modelList.NET.eval()

    with torch.no_grad():
        #SR
        SRImages = modelList.NET(dataDict['LR'])

        #calculate loss and backpropagation
        loss = mse_criterion(SRImages, dataDict['GT'])

        #return values
        lossDict = {'mse': loss}
        resultImagesDict = {'SR': SRImages}
    
    return lossDict, resultImagesDict

def inferenceStep(epoch, modelList, dataDict):

    #eval mode
    modelList.NET.eval()

    with torch.no_grad():
        #SR
        SRImages = modelList.NET(dataDict['LR'])
        #return values
        resultImagesDict = {'SR': SRImages}
    
    return {}, resultImagesDict






#################################################################################
#                                     EPOCH                                     #
#################################################################################

modelList = ModelList()

trainEpoch = Epoch( 
                    dataLoader = DataLoader('train'),
                    modelList = modelList,
                    step = trainStep,
                    researchVersion = version,
                    researchSubVersion = subversion,
                    writer = utils.initTensorboardWriter(version, subversion),
                    scoreMetricDict = { 'PSNR': {
                                        'function' : utils.calculateImagePSNR, 
                                        'argDataNames' : ['SR', 'GT'], 
                                        'additionalArgs' : ['$RANGE'],},
                                    }, 
                    resultSaveData = ['LR', 'SR', 'GT'] ,
                    resultSaveFileName = 'train',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'TRAIN'
                    )


validationEpoch = Epoch( 
                    dataLoader = DataLoader('validation'),
                    modelList = modelList,
                    step = validationStep,
                    researchVersion = version,
                    researchSubVersion = subversion,
                    writer = utils.initTensorboardWriter(version, subversion),
                    scoreMetricDict = { 'PSNR': {
                                        'function' : utils.calculateImagePSNR, 
                                        'argDataNames' : ['SR', 'GT'], 
                                        'additionalArgs' : ['$RANGE'],},
                                    }, 
                    resultSaveData = ['LR', 'SR', 'GT'] ,
                    resultSaveFileName = 'validation',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'VALIDATION'
                    )


inferenceEpoch = Epoch( 
                    dataLoader = DataLoader('inference'),
                    modelList = modelList,
                    step = inferenceStep,
                    researchVersion = version,
                    researchSubVersion = subversion,
                    writer = utils.initTensorboardWriter(version, subversion),
                    scoreMetricDict = {}, 
                    resultSaveData = ['LR', 'SR'] ,
                    resultSaveFileName = 'inference',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'INFERENCE'
                    )



#################################################
###############  EDIT THIS AREA  ################
#################################################
