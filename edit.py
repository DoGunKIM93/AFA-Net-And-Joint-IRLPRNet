'''
edit.py
'''
editversion = "1.0.200423"


#FROM Python LIBRARY
import time
import math
import numpy as np
import psutil

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
from backbone.utils import loadModels, saveModels, backproagateAndWeightUpdate        



################ V E R S I O N ################
# VERSION
version = '27-IRENE-APEX'
subversion = '0-test'
###############################################


#################################################
###############  EDIT THIS AREA  ################
#################################################
class ModelList(utils.ModelListBase):
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


        
        #(모델 인스턴스 이름)
        self.NET = model.EDVR(nf=128, nframes=7, groups=8, front_RBs=5, back_RBs=40)

        #(모델 인스턴스 이름)_optimizer로 선언
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=p.learningRate)

        #Learning Rate 스케쥴러 (없어도 됨)
        #self.NET_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.NET_optimizer, 0.0003, total_steps=200)
        #self.NET_scheduler = utils.NotOneCycleLR(self.NET_optimizer, p.learningRate, total_steps=p.schedulerPeriod)
        #self.NET_scheduler = torch.optim.lr_scheduler.CyclicLR(self.NET_optimizer, 0, 0.0003, step_size_up=50, step_size_down=150, cycle_momentum=False)
        #self.NET_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.NET_optimizer, 200)

        self.JUDGE = model.EfficientNet.from_name('efficientnet-b0')
        self.JUDGE_pretrained = 'efficientnet_b0_ns.pth'







        self.initApexAMP()
        self.initDataparallel()


def trainStep(epoch, modelList, LRImages, HRImages):


    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = utils.CharbonnierLoss(eps=1e-3)
    ssim_criterion = MS_SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
 
    # model
    modelList.NET.train() 


    # batch size
    batchSize = LRImages.size(0)

    ####################################################### X 2 #######################################################
    # SR Processing
    SRImages = modelList.NET(LRImages)

    #for MISR
    if len(HRImages.size()) == 5:
        loss = cpl_criterion(SRImages, HRImages[:,p.sequenceLength//2,:,:,:])
    else:
        loss = cpl_criterion(SRImages, HRImages) 

    
    rst = modelList.JUDGE(F.interpolate(HRImages[:,p.sequenceLength//2,:,:,:], [224,224]))



    # Update All model weights
    # if modelNames = None, this function updates all trainable model.
    # if modelNames is a String, updates one model
    # if modelNames is a List of string, updates those models.
    backproagateAndWeightUpdate(modelList, loss, modelNames = None)


    #AF = AF.detach()
    #SRImagesX2 = SRImages.detach()

    '''
    ####################################################### X 4 #######################################################
    # SR Processing
    AF, SRImages = modelList.NET(alignedFeatureSeq = AF, UltraLongSkipConnection = SRImagesX2, scaleLevel=1)

    #for MISR
    if len(HRImages.size()) == 5:
        loss = cpl_criterion(SRImages, HRImages[:,p.sequenceLength//2,:,:,:]) 
    else:
        loss = cpl_criterion(SRImages, HRImages) 

    # Update All model weights
    backproagateAndWeightUpdate(modelList, loss, modelNames = None)

    AF = AF.detach()
    SRImagesX4 = SRImages.detach()
    '''
    '''
    ############################################# Texture Restoration ##############################################
     # SR Processing
    _, SRImages = modelList.NET(alignedFeatureSeq = AF, UltraLongSkipConnection = SRImagesX4, scaleLevel='TR')

    #for MISR
    if len(HRImages.size()) == 5:
        
        sM, sP = vision.HPFinFreq(SRImages, 3, freqReturn = True)
        gM, gP = vision.HPFinFreq(HRImages[:,p.sequenceLength//2,:,:,:], 3, freqReturn = True)
        #sM, sP = vision.polarFFT(SRImages)
        #gM, gP = vision.polarFFT(HRImages[:,p.sequenceLength//2,:,:,:])
        #print(sM.max(), sM.min())
        #lossM = (1 - ssim_criterion( sM / 65535 , gM / 65535 )) * 200
        #lossC = cpl_criterion(SRImages, HRImages[:,p.sequenceLength//2,:,:,:])
        #lossM = cpl_criterion(sM, gM)
        #lossP = cpl_criterion(sP, gP)
        #loss = lossM + lossC
        loss = cpl_criterion(SRImages, torch.zeros_like(SRImages))#HRImages[:,p.sequenceLength//2,:,:,:]) 
        
        
    else:
        loss = cpl_criterion(vision.HistogramEqualization(SRImages), 
                             vision.HistogramEqualization(HRImages)) 

    # Update All model weights
    backproagateAndWeightUpdate(modelList, loss, modelNames = None)
    #loss.backward()

    SRImages = SRImages.detach()
    '''


    # return List of Result Images (also you can add some intermediate results).
    SRImagesList = [SRImages] 

    
    return loss, SRImagesList
     



def inferenceStep(epoch, modelList, LRImages, HRImages):


    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = utils.CharbonnierLoss(eps=1e-3)
 
    # model
    modelList.NET.eval()

    # batch size
    batchSize = LRImages.size(0)


    # SR Processing
    SRImages  = modelList.NET(LRImages)

    loss = mse_criterion(SRImages, HRImages)  
    
    return loss, SRImages

#################################################
###############  EDIT THIS AREA  ################
#################################################
