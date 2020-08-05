'''
edit.py
'''
editversion = "1.13.200729"


#FROM Python LIBRARY
import time
import math
import numpy as np
import psutil
import random

#FROM PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


#from pytorch_msssim (github/jorge_pessoa)
from pytorch_msssim import MS_SSIM



#from this project
import backbone.vision as vision
import model
import param as p
import backbone.utils as utils
import backbone.structure as structure
import backbone.module as module
from backbone.utils import loadModels, saveModels, backproagateAndWeightUpdate        
from backbone.config import Config



################ V E R S I O N ################
# VERSION
version = '32-ANDLTest'
subversion = '0-Test'
###############################################


#################################################
###############  EDIT THIS AREA  ################
#################################################
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
        # train() 및 valid() 에서 사용 방법
        # modelList.(모델 인스턴스 이름)_optimizer
        ##############################################################

        

        self.NET = model.VDSR()
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.0003)


        #self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()


def trainStep(epoch, modelList, LRImages, HRImages):



    # loss
    mse_criterion = nn.MSELoss()

    modelList.NET.train()


    # batch size
    batchSize = LRImages.size(0)

    SRImages = modelList.NET(LRImages)


    loss = mse_criterion(SRImages, HRImages)  


    # Update All model weights
    # if modelNames = None, this function updates all trainable model.
    # if modelNames is a String, updates one model
    # if modelNames is a List of string, updates those models.
    #backproagateAndWeightUpdate(modelList, loss, modelNames = "Ensemble")
    backproagateAndWeightUpdate(modelList, loss, modelNames = "NET")

    # return losses
    lossList = [loss]
    #lossList = [srAdversarialLoss, hrAdversarialLoss, loss_disc, loss_pixelwise, loss_adversarial, loss]
    # return List of Result Images (also you can add some intermediate results).
    #SRImagesList = [gaussianSprayKernel,bImage1,bImage2,blendedImages] 
    SRImagesList = [SRImages]

    
    return lossList, SRImagesList
     


def validationStep(epoch, modelList, LRImages, HRImages):
   ############# Inference Example code (with GT)#############
    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = module.CharbonnierLoss(eps=1e-3)
 
    # model
    modelList.Entire.eval()

    ####################################################### Preproc. #######################################################
    # SR Processing
    with torch.no_grad():
        SRImages   = modelList.Entire(LRImages)
    ####################################################### Ensemble. #######################################################
    #SRImages_Ensembled = modelList.Ensemble(SRImages_Cat)
    #loss = mse_criterion(SRImages_Ensembled, HRImages)
    if SRImages.shape != HRImages.shape:
        print(f"edit.py :: ERROR : \"SRImages.shape\" : {SRImages.shape} doesn't matach \"HRImages.shape\" : {HRImages.shape}")
        return
    else:
        loss = mse_criterion(SRImages, HRImages)
        # return List of Result Images (also you can add some intermediate results).
        #SRImagesList = [SRImages_Entire,SRImages_Face,SRImages_Ensembled] 
        SRImagesList = [SRImages]

    return loss, SRImagesList

    ''' inferenceStep
    ############# Inference Example code (without GT)#############
    # model
    modelList.Entire.eval()
    modelList.Face.eval()

    ####################################################### Preproc. #######################################################
    # SR Processing
    with torch.no_grad():
        print("LRImages.shape : ", LRImages.shape)
        SRImages_Entire = modelList.Entire(LRImages)
        SRImages_Face = modelList.Face(LRImages)
        print("SRImages_Entire.shape : ", SRImages_Entire.shape)
        print("SRImages_Face.shape : ", SRImages_Face.shape)
    ####################################################### Blending. #######################################################
    if p.blendingMode != None:
        destination = SRImages_Entire
        source = SRImages_Face
        
        ## test를 위한 roi (Detection 추가하면 roi 매핑 부분 수정 예정)
        roi = (1300, 700, 400, 400, 400, 900, 500, 300, 300, 400, 300, 500)
        blended_img = vision.BlendingMethod(p.blendingMode, source, destination, roi)
        SRImages_Ensembled = blended_img
    else:
        SRImages_Ensembled = SRImages_Entire

    # return List of Result Images (also you can add some intermediate results).
    SRImagesList = [SRImages_Ensembled] 
    
    '''
def inferenceStep(modelList, LRImages):
    # model
    modelList.Entire.eval()
    modelList.Face.eval()

    ####################################################### Preproc. #######################################################
    # SR Processing
    with torch.no_grad():
        SRImages_Entire = modelList.Entire(LRImages)
        SRImages_Face = modelList.Face(LRImages)
    ####################################################### Blending. #######################################################
    if p.blendingMode != None:
        destination = SRImages_Entire
        source = SRImages_Face
        
        # test를 위한 roi (Detection 추가하면 roi 매핑 부분 수정 예정)
        roi = (1300, 700, 400, 400, 400, 900, 500, 300, 300, 400, 300, 500)
        blended_img = vision.BlendingMethod(p.blendingMode, source, destination, roi)
        SRImages_Ensembled = blended_img
    else:
        SRImages_Ensembled = SRImages_Entire

    # return List of Result Images (also you can add some intermediate results).
    SRImagesList = [SRImages_Ensembled] 
    
    return SRImagesList

#################################################
###############  EDIT THIS AREA  ################
#################################################
