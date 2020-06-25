'''
edit.py
'''
editversion = "1.10.200527"


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
from backbone.utils import loadModels, saveModels, backproagateAndWeightUpdate        



################ V E R S I O N ################
# VERSION
version = '30-EndtoEntDeepBlending'
subversion = '2-shiftmoduletrainassssment'
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


        
        #(모델 인스턴스 이름)
        #self.Entire = model.EDVR(nf=128, nframes=7, groups=8, front_RBs=5, back_RBs=40)
        #self.Entire_pretrained = "EDVR_Vimeo90K_SR_L.pth"
        
        #self.Face = model.ESPCN()
        #self.Face_pretrained = "ESPCN_face.pth"

        #self.Face = model.EDVR(nf=128, nframes=1, groups=1, front_RBs=5, back_RBs=40)
        #self.Face_pretrained = "EDVR_Face.pth"

        #Learning Rate 스케쥴러 (없어도 됨)
        #self.NET_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.NET_optimizer, 0.0003, total_steps=200)
        #self.NET_scheduler = utils.NotOneCycleLR(self.NET_optimizer, p.learningRate, total_steps=p.schedulerPeriod)
        #self.NET_scheduler = torch.optim.lr_scheduler.CyclicLR(self.NET_optimizer, 0, 0.0003, step_size_up=50, step_size_down=150, cycle_momentum=False)
        #self.NET_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.NET_optimizer, 200)

        #self.Ensemble = model.WENDY(CW=64, Blocks=10, Attention=False)
        #self.Ensemble_optimizer = torch.optim.Adam(self.Ensemble.parameters(), lr=p.learningRate)

        #self.E_FE = model.EfficientNet('b0', num_classes=1, mode='feature_extractor')
        #self.E_FE_optimizer = torch.optim.Adam(self.E_FE.parameters(), lr=p.learningRate)
        #self.E_FE_pretrained = 'efficientnet_b0_ns.pth'

        self.E_FE = model.SunnySideUp()
        self.E_FE_optimizer = torch.optim.Adam(self.E_FE.parameters(), lr=p.learningRate)

        #self.E_Deco = model.ISAF(featureExtractor = self.E_FE)
        #self.E_Deco_optimizer = torch.optim.Adam(self.E_Deco.parameters(), lr=p.learningRate * 3)

        #self.Perceptual = model.EfficientNet('b0', mode='feature_extractor')
        #self.Perceptual_pretrained = 'efficientnet_b0_ns.pth'

        #self.Disc = model.EfficientNet('b0', num_classes=1)
        #self.Disc_pretrained = 'efficientnet_b0_ns.pth'
        #self.Disc_optimizer = torch.optim.Adam(self.Disc.parameters(), lr=p.learningRate)

        self.initApexAMP()
        self.initDataparallel()


def trainStep(epoch, modelList, LRImages, HRImages):



    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = utils.CharbonnierLoss(eps=1e-3)
    bce_criterion = nn.BCELoss()
    ssim_criterion = MS_SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
 
    # model
    #modelList.Entire.eval()
    #modelList.Face.eval()
    modelList.E_FE.train()
    #modelList.E_Deco.train()
    #modelList.Disc.train()


    # batch size
    batchSize = LRImages.size(0)

    ####################################################### Preproc. #######################################################
    # SR Processing

    with torch.no_grad():
        #SRImages_Entire = modelList.Entire(LRImages)
        #SRImages_Face   = modelList.Face(LRImages)
        pass

    '''
    gsklst = []
    for bb in range(batchSize):
        gsklst.append(vision.GaussianSpray(LRImages.size(2), LRImages.size(3), 3, 10).repeat(1,3,1,1))
    gaussianSprayKernel = torch.cat(gsklst, 0)

    bImage1 = LRImages * gaussianSprayKernel + HRImages * (1 - gaussianSprayKernel)
    bImage2 = HRImages * gaussianSprayKernel + LRImages * (1 - gaussianSprayKernel)

    if random.randint(0,1) == 1:
        t = bImage1 
        bImage1 = bImage2
        bImage2 = t
    '''

    '''
    ####################################################### Train Disc. ####################################################

    with torch.no_grad():
        SRImages_Ensembled = modelList.E_Deco(SRImages_Entire, SRImages_Face)

    for xx in range(1):
        #lrAdversarialScore = modelList.Disc(F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic'))
        srAdversarialScore = modelList.Disc(SRImages_Ensembled)
        hrAdversarialScore = modelList.Disc(HRImages)
        #lrAdversarialLoss = bce_criterion(lrAdversarialScore, torch.zeros_like(lrAdversarialScore))
        srAdversarialLoss = bce_criterion(srAdversarialScore, torch.zeros_like(srAdversarialScore))
        hrAdversarialLoss = bce_criterion(hrAdversarialScore, torch.ones_like(hrAdversarialScore))
        loss_disc = srAdversarialLoss + hrAdversarialLoss
        backproagateAndWeightUpdate(modelList, loss_disc, modelNames = "Disc")

    ####################################################### Ensemble. #######################################################
    '''

    
    trI = Variable(torch.ones(batchSize,3,256,256), requires_grad=True).cuda()
    trI[:,:,1,1] = 0.

    gtI = torch.ones(batchSize,3,256,256).cuda()
    gtI[:,:,2,2] = 0.

    trI = modelList.E_FE(trI)



    #blendedImages = modelList.E_Deco(bImage1, bImage2)
    

    #srAdversarialScore = modelList.Disc(SRImages_Ensembled)

    #srPerceptureScore = modelList.Perceptual(SRImages_Ensembled)
    #with torch.no_grad():
    #    hrPerceptureScore = modelList.Perceptual(HRImages)

    #loss_perceptual = mse_criterion(srPerceptureScore * 100000, hrPerceptureScore * 100000) 
    loss_pixelwise = mse_criterion(trI, gtI)
    #loss_adversarial = bce_criterion(srAdversarialScore, torch.ones_like(srAdversarialScore))

    #print(loss_disc, loss_perceptual, loss_pixelwise, loss_adversarial)

    loss = loss_pixelwise# + loss_adversarial * 0.002


    #for MISR
    #if len(HRImages.size()) == 5:
    #    loss = cpl_criterion(SRImages_Ensembled, HRImages[:,p.sequenceLength//2,:,:,:])
    #else:
    #    loss = cpl_criterion(SRImages_Ensembled, HRImages) 



    # Update All model weights
    # if modelNames = None, this function updates all trainable model.
    # if modelNames is a String, updates one model
    # if modelNames is a List of string, updates those models.
    #backproagateAndWeightUpdate(modelList, loss, modelNames = "Ensemble")
    backproagateAndWeightUpdate(modelList, loss, modelNames = ["E_FE"])


    #AF = AF.detach()
    #SRImagesX2 = SRImages.detach()

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

    # return losses
    lossList = [loss_pixelwise]
    #lossList = [srAdversarialLoss, hrAdversarialLoss, loss_disc, loss_pixelwise, loss_adversarial, loss]
    # return List of Result Images (also you can add some intermediate results).
    #SRImagesList = [gaussianSprayKernel,bImage1,bImage2,blendedImages] 
    SRImagesList = [trI,gtI]

    
    return lossList, SRImagesList
     



def inferenceStep(epoch, modelList, LRImages, HRImages):

    

    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = utils.CharbonnierLoss(eps=1e-3)
 
    # model
    modelList.Entire.eval()
    modelList.Face.eval()
    modelList.E_FE.train()
    modelList.E_Deco.train()
    # batch size
    batchSize = LRImages.size(0)

    ####################################################### Preproc. #######################################################
    # SR Processing

    with torch.no_grad():
        SRImages_Entire = modelList.Entire(LRImages)
        SRImages_Face   = modelList.Face(LRImages * 2 - 1)
        SRImages_Cat = torch.cat([SRImages_Entire, SRImages_Face], 1)

    ####################################################### Ensemble. #######################################################

    SRImages_Ensembled = modelList.Ensemble(SRImages_Cat)

    loss = mse_criterion(SRImages_Ensembled, HRImages)  

    # return List of Result Images (also you can add some intermediate results).
    SRImagesList = [SRImages_Entire,SRImages_Face,SRImages_Ensembled] 

    
    return loss, SRImagesList

#################################################
###############  EDIT THIS AREA  ################
#################################################
