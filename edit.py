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
from backbone.structure import Epoch
from dataLoader import DataLoader
from warmup_scheduler import GradualWarmupScheduler



################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)

version = '1-BMVC'
subversion = '1-AFA-Net3'

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

        # SR 1) DeFiAN
        self.SR = predefined.DeFiAN(32, 10, 5, 4)
        self.SR_optimizer = torch.optim.Adam(self.SR.parameters(), lr=0.0003)
        self.SR_pretrained = "DeFiAN_S_x4.pth"
        
        # Deblur 2) MPRNet
        self.Deblur = predefined.MPRNet()
        self.Deblur_pretrained = "MPRNet_pretrained.pth"
        # self.Deblur_optimizer = torch.optim.Adam(self.Deblur.parameters(), lr=0.00003)

        self.Deblur_optimizer = torch.optim.Adam(self.Deblur.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8)
        num_epochs = 3000
        warmup_epochs = 3
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.Deblur_optimizer, num_epochs-warmup_epochs, eta_min=1e-6)
        self.Deblur_scheduler = GradualWarmupScheduler(self.Deblur_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

        # AFA-Net
        self.BLENDER_FE = predefined.ResNeSt("200", mode="feature_extractor")
        self.BLENDER_FE_optimizer = torch.optim.Adam(self.BLENDER_FE.parameters(), lr=0.0001)
        self.BLENDER_FE_pretrained = "BLENDER_FE_ResNeSt200-2.pth"

        self.BLENDER_RES_f4 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=2048, featureCW=2048)
        self.BLENDER_RES_f4_optimizer = torch.optim.Adam(self.BLENDER_RES_f4.parameters(), lr=0.0003)
        self.BLENDER_RES_f3 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=1024, featureCW=1024)
        self.BLENDER_RES_f3_optimizer = torch.optim.Adam(self.BLENDER_RES_f3.parameters(), lr=0.0003)
        self.BLENDER_RES_f2 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=512, featureCW=512)
        self.BLENDER_RES_f2_optimizer = torch.optim.Adam(self.BLENDER_RES_f2.parameters(), lr=0.0003)
        self.BLENDER_RES_f1 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=256, featureCW=256)
        self.BLENDER_RES_f1_optimizer = torch.optim.Adam(self.BLENDER_RES_f1.parameters(), lr=0.0003)

        self.BLENDER_DECO = model.tSeNseR_AFA(CW=2048, inFeatures=1)
        self.BLENDER_DECO_optimizer = torch.optim.Adam(self.BLENDER_DECO.parameters(), lr=0.0001)
        self.BLENDER_DECO_pretrained = "BLENDER_DECO_ResNeSt200-2.pth"

        self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()



def _blend(modelList, SRImages, deblurredImages):
    SR_f1, SR_f2, SR_f3, SR_f4 = modelList.BLENDER_FE(SRImages)
    De_f1, De_f2, De_f3, De_f4 = modelList.BLENDER_FE(deblurredImages)

    W_f1 = modelList.BLENDER_RES_f1([SR_f1, De_f1])
    W_f2 = modelList.BLENDER_RES_f2([SR_f2, De_f2])
    W_f3 = modelList.BLENDER_RES_f3([SR_f3, De_f3])
    W_f4 = modelList.BLENDER_RES_f4([SR_f4, De_f4])
    f1 = W_f1 * SR_f1 + (1 - W_f1) * De_f1
    f2 = W_f2 * SR_f2 + (1 - W_f2) * De_f2
    f3 = W_f3 * SR_f3 + (1 - W_f3) * De_f3
    f4 = W_f4 * SR_f4 + (1 - W_f4) * De_f4

    return modelList.BLENDER_DECO([[f1, f2, f3, f4]]), [f1, f2, f3, f4]


#################################################################################
#                                     STEPS                                     #
#################################################################################

def trainStep(epoch, modelList, dataDict):
    LRImages = dataDict['LR']
    HRImages = dataDict['GT']

    #define loss function
    mse_criterion = nn.MSELoss()
    dists_criterion = DISTS(channels=3).cuda()

    modelList.SR.train()
    modelList.Deblur.train()
    
    #train mode
    modelList.BLENDER_RES_f1.train()
    modelList.BLENDER_RES_f2.train()
    modelList.BLENDER_RES_f3.train()
    modelList.BLENDER_RES_f4.train()

    modelList.BLENDER_FE.train()
    modelList.BLENDER_DECO.train()

    # with torch.no_grad():
    #SR
    SRImages = modelList.SR(LRImages)
    #Deblur
    deblurredImages = modelList.Deblur(LRImages)[0]
    deblurredImages = F.interpolate(deblurredImages, size=SRImages.size()[-2:])

    blendedImages, _ = _blend(modelList, SRImages, deblurredImages)
    
    #calculate loss and backpropagation
    lossSR_MSE = mse_criterion(SRImages, dataDict['GT'])
    lossSR_DISTS = dists_criterion(SRImages, dataDict['GT'])

    lossDeblur = mse_criterion(deblurredImages, dataDict['GT'])

    lossBlend_MSE = mse_criterion(blendedImages, HRImages)
    lossBlend_DISTS = dists_criterion(blendedImages, HRImages)

#     loss = lossSR_MSE * 0.3 + lossSR_DISTS * 0.3 + lossDeblur * 0.3 + lossBlend_MSE + lossBlend_DISTS
    loss = lossBlend_MSE

    backproagateAndWeightUpdate(
        modelList, 
        loss, 
        modelNames=["SR", "Deblur", "BLENDER_RES_f1", "BLENDER_RES_f2", "BLENDER_RES_f3", "BLENDER_RES_f4", "BLENDER_FE", "BLENDER_DECO"]
    )

    #return values
    lossDict = {
        'train_lossSR_MSE': lossSR_MSE,
        'train_lossSR_DISTS': lossSR_DISTS,
        'train_lossDeblur': lossDeblur,
        'train_lossBlend_MSE': lossBlend_MSE,
        'train_lossBlend_DISTS': lossBlend_DISTS,
        'train_loss': loss
    }
    resultImagesDict = {
        "SR": SRImages, 
        "Deblur": deblurredImages, 
        "Blend": blendedImages
    }
    
    return lossDict, resultImagesDict
     


def validationStep(epoch, modelList, dataDict):
    LRImages = dataDict['LR']
    HRImages = dataDict['GT']

    #define loss function
    mse_criterion = nn.MSELoss()
    dists_criterion = DISTS(channels=3).cuda()

    #eval mode
    modelList.SR.eval()
    modelList.Deblur.eval()
    
    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()

    with torch.no_grad():
        ###### SR
        SRImages = modelList.SR(LRImages)

        ###### DeBlur
        deblurredImages_small = modelList.Deblur(LRImages)[0]
        deblurredImages = F.interpolate(deblurredImages_small, size=SRImages.size()[-2:])

        # SR-DEBLUR
        SR_deblurredImages = modelList.Deblur(SRImages)[0]

        # DEBLUR-SR
        deblurred_SRImages = modelList.SR(deblurredImages_small)

        # BLEND
        blendedImages, _ = _blend(modelList, SRImages, deblurredImages)

    #calculate loss and backpropagation
    lossSR_MSE = mse_criterion(SRImages, dataDict['GT'])
    lossSR_DISTS = dists_criterion(SRImages, dataDict['GT'])

    lossDeblur = mse_criterion(deblurredImages, dataDict['GT'])

    lossBlend_MSE = mse_criterion(blendedImages, HRImages)
    lossBlend_DISTS = dists_criterion(blendedImages, HRImages)

#     loss = lossBlend_MSE * 0.7 + lossBlend_DISTS * 0.3
    loss = lossBlend_MSE

    #return values
    lossDict = {
        'valid_lossSR_MSE': lossSR_MSE,
        'valid_lossSR_DISTS': lossSR_DISTS,
        'valid_lossDeblur': lossDeblur,
        'valid_lossBlend_MSE': lossBlend_MSE,
        'valid_lossBlend_DISTS': lossBlend_DISTS,
        'valid_loss': loss
    }
    resultImagesDict = {
        "SR": SRImages, 
        "Deblur": deblurredImages, 
        "SR_Deblur": SR_deblurredImages, 
        "Deblur_SR": deblurred_SRImages, 
        "Blend": blendedImages
    }
    
    return lossDict, resultImagesDict

def inferenceStep(epoch, modelList, dataDict):
    LRImages = dataDict['LR']
    HRImages = dataDict['GT']

    #eval mode
    modelList.SR.eval()
    modelList.Deblur.eval()
    
    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()

    with torch.no_grad():
        ###### SR
        SRImages = modelList.SR(LRImages)

        ###### DeBlur
        deblurredImages_small = modelList.Deblur(LRImages)[0]
        deblurredImages = F.interpolate(deblurredImages_small, size=SRImages.size()[-2:])

        # SR-DEBLUR
        SR_deblurredImages = modelList.Deblur(SRImages)[0]

        # DEBLUR-SR
        deblurred_SRImages = modelList.SR(deblurredImages_small)

        # BLEND
        blendedImages, _ = _blend(modelList, SRImages, deblurredImages)

    #return values
    resultImagesDict = {
        "SR": SRImages, 
        "Deblur": deblurredImages, 
        "SR_Deblur": SR_deblurredImages, 
        "Deblur_SR": deblurred_SRImages, 
        "Blend": blendedImages
    }
    
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
                    resultSaveData = ['LR', 'SR', 'Deblur', 'Blend', 'GT'] ,
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
                    resultSaveData = ['LR', 'SR', 'Deblur', 'SR_Deblur', 'Deblur_SR', 'Blend', 'GT'] ,
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
                    resultSaveData = ['LR', 'SR', 'Deblur', 'SR_Deblur', 'Deblur_SR', 'Blend'] ,
                    resultSaveFileName = 'inference',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'INFERENCE'
                    )



#################################################
###############  EDIT THIS AREA  ################
#################################################
