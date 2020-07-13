'''
edit.py
'''
editversion = "1.12.200706"


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



################ V E R S I O N ################
# VERSION
version = '31-DeepIQA'
subversion = '1-diff'
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

        
        self.Entire = model.EDVR(nf=128, nframes=1, groups=1, front_RBs=5, back_RBs=40)
        self.Entire_pretrained = "EDVR_SISR_general.pth"

        self.E_FE = model.EfficientNet('b0', num_classes=1, mode='feature_extractor')
        self.E_FE_optimizer = torch.optim.Adam(self.E_FE.parameters(), lr=p.learningRate)
        self.E_FE_pretrained = 'efficientnet_b0_ns.pth'

        self.E_Deco = model.DeNIQuA(featureExtractor = self.E_FE, inFeature=2)
        self.E_Deco_optimizer = torch.optim.Adam(self.E_Deco.parameters(), lr=p.learningRate)

        self.initApexAMP()
        self.initDataparallel()


def trainStep(epoch, modelList, LRImages, HRImages):



    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = module.CharbonnierLoss(eps=1e-3)
    bce_criterion = nn.BCELoss()
    ssim_criterion = MS_SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
 
    # model
    #modelList.Entire.eval()
    #modelList.Face.eval()
    modelList.E_FE.eval()
    modelList.E_Deco.train()
    #modelList.Disc.train()


    # batch size
    batchSize = LRImages.size(0)

    ####################################################### Preproc. #######################################################
    # SR Processing

    with torch.no_grad():
        SRImages_Entire = modelList.Entire(LRImages)
        #SRImages_Face   = modelList.Face(LRImages)
        pass

    
    gsklst = []
    for bb in range(batchSize):
        gsklst.append(torch.clamp(vision.GaussianSpray(HRImages.size(2), HRImages.size(3), 5, 10).repeat(1,3,1,1) - vision.GaussianSpray(HRImages.size(2), HRImages.size(3), 2, 7).repeat(1,3,1,1), 0, 1))
    gaussianSprayKernel = torch.cat(gsklst, 0)

    bImageLH = HRImages * gaussianSprayKernel + F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic') * (1 - gaussianSprayKernel)
    bImageLS = SRImages_Entire * gaussianSprayKernel + F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')  * (1 - gaussianSprayKernel)
    bImageSH = HRImages * gaussianSprayKernel + SRImages_Entire * (1 - gaussianSprayKernel)

    gaussianSprayKernel = 1 - gaussianSprayKernel
    bImageHL = HRImages * gaussianSprayKernel + F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic') * (1 - gaussianSprayKernel)
    bImageSL = SRImages_Entire * gaussianSprayKernel + F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')  * (1 - gaussianSprayKernel)
    bImageHS = HRImages * gaussianSprayKernel + SRImages_Entire * (1 - gaussianSprayKernel)

    gaussianSprayKernel = 1 - gaussianSprayKernel
    '''

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

    '''
    #SHIFT MODULE TEST CODE
    trI = Variable(torch.ones(batchSize,3,256,256), requires_grad=True).cuda()
    trI[:,:,1,1] = 0.

    gtI = torch.ones(batchSize,3,256,256).cuda()
    gtI[:,:,2,2] = 0.

    trI = modelList.E_FE(trI)
    '''


    IQAMapSRHR = modelList.E_Deco([bImageSH, bImageHS])
    IQAMapSRHRInv = modelList.E_Deco([bImageHS, bImageSH])
    IQAMapSRHRIde = modelList.E_Deco([bImageSH, bImageSH])

    IQAMapLRSR = modelList.E_Deco([bImageLS, bImageSL])
    IQAMapLRSRInv = modelList.E_Deco([bImageSL, bImageLS])
    IQAMapLRSRIde = modelList.E_Deco([bImageLS, bImageLS])

    IQAMapLRSROri = modelList.E_Deco([F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic'), SRImages_Entire])
    IQAMapLRSROriInv = modelList.E_Deco([SRImages_Entire, F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')])

    IQAMapSRHROri = modelList.E_Deco([SRImages_Entire, HRImages])
    IQAMapSRHROriInv = modelList.E_Deco([HRImages, SRImages_Entire])

    IQAMapLRHR = modelList.E_Deco([bImageLH, bImageHL])
    IQAMapLRHRInv = modelList.E_Deco([bImageHL, bImageLH])
    IQAMapLRHRIde = modelList.E_Deco([bImageLS, bImageLS])

    


    with torch.no_grad():
        #IQAMapLRHROri = modelList.E_Deco([F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic'), HRImages])
        #IQAMapLRHROriInv = modelList.E_Deco([HRImages, F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')])
        IQAMapTest = modelList.E_Deco([SRImages_Entire, bImageLS])
        IQAMapTestInv = modelList.E_Deco([bImageLS, F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')])

        IQAMapTest2 = modelList.E_Deco([SRImages_Entire, bImageSH])
        IQAMapTest2Inv = modelList.E_Deco([bImageHS, F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')])

        IQAMapTest3 = modelList.E_Deco([SRImages_Entire, bImageLH])
        IQAMapTest3Inv = modelList.E_Deco([bImageHL, F.interpolate(LRImages, scale_factor=p.scaleFactor, mode='bicubic')])


    loss_Diff_SRHR = mse_criterion(IQAMapSRHR, gaussianSprayKernel) + mse_criterion(IQAMapSRHRInv, 1 - gaussianSprayKernel) 
    loss_Diff_LRSR = mse_criterion(IQAMapLRSR, gaussianSprayKernel) + mse_criterion(IQAMapLRSRInv, 1 - gaussianSprayKernel) 
    loss_Diff_LRHR = mse_criterion(IQAMapLRHR, gaussianSprayKernel) + mse_criterion(IQAMapLRHRInv, 1 - gaussianSprayKernel) 

    loss_Identity = mse_criterion(IQAMapSRHRIde, torch.ones_like(IQAMapSRHRIde) / 2) + mse_criterion(IQAMapLRSRIde, torch.ones_like(IQAMapLRSRIde) / 2) + mse_criterion(IQAMapLRHRIde, torch.ones_like(IQAMapLRHRIde) / 2)

    loss_Absolute = mse_criterion(IQAMapSRHROri, torch.ones_like(IQAMapSRHROri)) + mse_criterion(IQAMapSRHROriInv, torch.zeros_like(IQAMapSRHROriInv)) + mse_criterion(IQAMapLRSROri, torch.ones_like(IQAMapLRSROri)) + mse_criterion(IQAMapLRSROriInv, torch.zeros_like(IQAMapLRSROriInv)) 

    loss = loss_Diff_SRHR + loss_Diff_LRSR + loss_Diff_LRHR + loss_Identity + loss_Absolute * 0.25  # + loss_adversarial * 0.002


    # Update All model weights
    # if modelNames = None, this function updates all trainable model.
    # if modelNames is a String, updates one model
    # if modelNames is a List of string, updates those models.
    #backproagateAndWeightUpdate(modelList, loss, modelNames = "Ensemble")
    backproagateAndWeightUpdate(modelList, loss, modelNames = ["E_FE", "E_Deco"])

    # return losses
    lossList = [loss_Diff_SRHR, loss_Diff_LRSR, loss_Diff_LRHR, loss_Identity, loss_Absolute, loss]
    #lossList = [srAdversarialLoss, hrAdversarialLoss, loss_disc, loss_pixelwise, loss_adversarial, loss]
    # return List of Result Images (also you can add some intermediate results).
    #SRImagesList = [gaussianSprayKernel,bImage1,bImage2,blendedImages] 
    SRImagesList = [SRImages_Entire,bImageSH,gaussianSprayKernel,
                    IQAMapSRHR,IQAMapSRHRInv,IQAMapSRHRIde,
                    IQAMapLRSR, IQAMapLRSRInv,IQAMapLRSRIde, 
                    IQAMapLRSROri, IQAMapLRSROriInv, IQAMapSRHROri, IQAMapSRHROriInv,
                    IQAMapTest, IQAMapTestInv, IQAMapTest2, IQAMapTest2Inv, IQAMapTest3, IQAMapTest3Inv
                    ]

    
    return lossList, SRImagesList
     


    ''' validationStep
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

    '''
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
