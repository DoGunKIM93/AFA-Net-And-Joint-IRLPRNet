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

version = 'Medical'
#subversion = '1-EDVR_CXR8_inference_256_x4down'
#subversion = '1-SPSR_CXR8_inference'
subversion = '1-SPSR_CXR8_AugmenatationTest'

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

        
        # EDVR        
        self.NET = predefined.EDVR(nf=128, nframes=1, groups=1, front_RBs=5, back_RBs=40)
        self.NET_pretrained = "EDVR-CXR8.pth"  # FaceModel: "EDVR-Face.pth" # satelliteModel : EDVR-DOTA.pth
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.0003)
        
        
        ########################################################################################################################
        # SPSR        
        
        self.netG = predefined.SPSR_Generator(scaleFactor = 4)
        self.netG_pretrained = "SPSR-netG-CXR8.pth"#"SPSR-RRDB_PSNR_x4.pth"   # FaceModel: "SPSR-netG-Face.pth" # satelliteModel : SPSR-netG-DOTA.pth # medicalModel : SPSR-netG-CXR8.pth
        self.netG_optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))
        self.netG_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.netG_optimizer, [5000,100000,200000,300000], 0.5)

        self.netD = predefined.SPSR_Discriminator(size = 64)
        self.netD_pretrained = "SPSR-netD-CXR8.pth"   # FaceModel: "SPSR-netD-Face.pth" # satelliteModel : SPSR-netD-DOTA.pth # medicalModel : SPSR-netD-CXR8.pth
        self.netD_optimizer = torch.optim.Adam(self.netD.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))
        self.netD_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.netD_optimizer, [5000,100000,200000,300000], 0.5)

        self.netDgrad = predefined.SPSR_Discriminator(size = 64)
        self.netDgrad_pretrained = "SPSR-netDgrad-CXR8.pth"   # FaceModel: "SPSR-netDgrad-Face.pth" # satelliteModel : "SPSR-netDgrad-DOTA.pth" # medicalModel : SPSR-netDgrad-CXR8.pth
        self.netDgrad_optimizer = torch.optim.Adam(self.netD.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))
        self.netDgrad_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.netDgrad_optimizer, [5000,100000,200000,300000], 0.5)

        self.netF = predefined.SPSR_FeatureExtractor()

        self.Get_gradient = predefined.SPSR_Get_gradient()
        self.Get_gradient_nopadding = predefined.SPSR_Get_gradient_nopadding()
        ########################################################################################################################
        
        self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()



#################################################################################
#                                     STEPS                                     #
#################################################################################

def trainStep(epoch, modelList, dataDict):

    
    #define loss function
    mse_criterion = nn.MSELoss()
    '''
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
    '''

    ####################################################################################################
    ## SPSR
    #define input data and label
    LRImages = dataDict['LR']
    HRImages = dataDict['HR']

    l_d_total_grad = 0
    l_g_total = 0

    pixel_criterion = "l1"
    pixel_weight = 2e-2
    feature_criterion = "l1"
    feature_weight = 1
    gan_type = "vanilla"
    gan_weight = 5e-3
    gradient_pixel_weight = 1e-2
    gradient_gan_weight = 5e-3
    pixel_branch_criterion = "l1"
    pixel_branch_weight = 5e-1
    Branch_pretrain = 1
    Branch_init_iters = 5000

    weight_decay_G = 0
    weight_decay_G_grad = 0
    weight_decay_D = 0

    D_update_ratio = 1
    D_init_iters = 0
    Branch_pretrain = 0
    Branch_init_iters = 1

    beta1_G = 0.9
    beta1_G_grad = 0.9
    beta1_D = 0.9

    # train
    modelList.netG.train()
    modelList.netD.train()
    modelList.netDgrad.train()
    
    # G pixel loss
    if pixel_weight > 0:
        l_pix_type = pixel_criterion
        if l_pix_type == 'l1':
            cri_pix = nn.L1Loss()
        elif l_pix_type == 'l2':
            cri_pix = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
        l_pix_w = pixel_weight
    else:
        cri_pix = None

    # G feature loss
    if feature_weight > 0:
        l_fea_type = feature_criterion
        if l_fea_type == 'l1':
            cri_fea = nn.L1Loss()
        elif l_fea_type == 'l2':
            cri_fea = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
        l_fea_w = feature_weight
    else:
        cri_fea = None

    # GD gan loss
    cri_gan = GANLoss(gan_type, 1.0, 0.0)
    l_gan_w = gan_weight

    # gradient_pixel_loss
    if gradient_pixel_weight > 0:
        cri_pix_grad = nn.MSELoss()
        l_pix_grad_w = gradient_pixel_weight
    else:
        cri_pix_grad = None

    # gradient_gan_loss
    if gradient_gan_weight > 0:
        cri_grad_gan = GANLoss(gan_type, 1.0, 0.0)
        l_gan_grad_w = gradient_gan_weight
    else:
        cri_grad_gan = None

    # G_grad pixel loss
    if pixel_branch_weight > 0:
        l_pix_type = pixel_branch_criterion
        if l_pix_type == 'l1':
            cri_pix_branch = nn.L1Loss()
        elif l_pix_type == 'l2':
            cri_pix_branch = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
        l_pix_branch_w = pixel_branch_weight
    else:
        cri_pix_branch = None

    log_dict = OrderedDict()
    get_grad = modelList.Get_gradient
    get_grad_nopadding = modelList.Get_gradient_nopadding


    # Optimizing
    # netG_optimizer
    modelList.netG_optimizer.zero_grad()

    fake_H_branch, fake_H, grad_LR = modelList.netG(LRImages)
    
    fake_H_grad = get_grad(fake_H)
    var_H_grad = get_grad(HRImages)
    var_ref_grad = get_grad(HRImages)
    var_H_grad_nopadding = get_grad_nopadding(HRImages)
    
    #print(fake_H.size(), HRImages.size())

    l_g_total = 0
    if epoch % D_update_ratio == 0 and epoch > D_init_iters:
        if cri_pix:  # pixel loss
            l_g_pix = l_pix_w * cri_pix(fake_H, HRImages)
            l_g_total += l_g_pix
        if cri_fea:  # feature loss
            real_fea = modelList.netF(HRImages)
            fake_fea = modelList.netF(fake_H)
            l_g_fea = l_fea_w * cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea
        
        if cri_pix_grad: #gradient pixel loss
            l_g_pix_grad = l_pix_grad_w * cri_pix_grad(fake_H_grad, var_H_grad)
            l_g_total += l_g_pix_grad


        if cri_pix_branch: #branch pixel loss
            l_g_pix_grad_branch = l_pix_branch_w * cri_pix_branch(fake_H_branch, var_H_grad_nopadding)
            l_g_total += l_g_pix_grad_branch


        # G gan + cls loss
        pred_g_fake = modelList.netD(fake_H)
        pred_d_real = modelList.netD(HRImages).detach()
        
        l_g_gan = l_gan_w * (cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_g_total += l_g_gan

        # grad G gan + cls loss
        pred_g_fake_grad = modelList.netDgrad(fake_H_grad)
        pred_d_real_grad = modelList.netDgrad(var_ref_grad).detach()

        l_g_gan_grad = l_gan_grad_w * (cri_grad_gan(pred_d_real_grad - torch.mean(pred_g_fake_grad), False) + 
                                            cri_grad_gan(pred_g_fake_grad - torch.mean(pred_d_real_grad), True)) /2
        l_g_total += l_g_gan_grad

        l_g_total.backward()
        modelList.netG_optimizer.step()


    # D
    modelList.netD_optimizer.zero_grad()
    l_d_total = 0

    pred_d_real = modelList.netD(HRImages)
    pred_d_fake = modelList.netD(fake_H.detach())  # detach to avoid BP to G

    l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
    l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

    l_d_total = (l_d_real + l_d_fake) / 2

    if gan_type == 'wgan-gp':
        if random_pt.size(0) != batch:
            random_pt.resize_(batch, 1, 1, 1)
        random_pt.uniform_()  # Draw random interpolation points
        interp = random_pt * fake_H.detach() + (1 - random_pt) * HRImages
        interp.requires_grad = True
        interp_crit, _ = modelList.netD(interp)
        l_d_gp = l_gp_w * cri_gp(interp, interp_crit) 
        l_d_total += l_d_gp

    l_d_total.backward()

    modelList.netD_optimizer.step()

    
    # D_grad
    pred_d_real_grad = modelList.netDgrad(var_ref_grad)
    pred_d_fake_grad = modelList.netDgrad(fake_H_grad.detach())  # detach to avoid BP to G
    
    l_d_real_grad = cri_grad_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True)
    l_d_fake_grad = cri_grad_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False)

    l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2

    l_d_total_grad.backward()

    modelList.netDgrad_optimizer.step()

    SRImages = fake_H

    l_g_total = torch.as_tensor(l_g_total)
    l_d_total = torch.as_tensor(l_d_total)
    l_d_total_grad = torch.as_tensor(l_d_total_grad)

    lossDict = {'g_total': l_g_total, 'd_total': l_d_total, 'd_total_grad': l_d_total_grad}
    SRImagesDict = {'SR' : SRImages}
    ##############################################################################################################
    

    return lossDict, SRImagesDict
     


def validationStep(epoch, modelList, dataDict):

    
    #define loss function
    mse_criterion = nn.MSELoss()
    '''
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
    '''

    ##############################################################################################################
    LRImages = dataDict['LR']
    HRImages = dataDict['HR']

    fake_H_branch = None
    SRImages = None
    grad_LR = None

    #gan_type = "vanilla"
    #cri_gan = GANLoss(gan_type, 1.0, 0.0)
    bce_criterion = nn.BCEWithLogitsLoss()

    modelList.netG.eval()
    with torch.no_grad():
        fake_H_branch, SRImages, grad_LR = modelList.netG(LRImages)
    
    loss = bce_criterion(SRImages, HRImages)
    loss = torch.as_tensor(loss)

    lossDict = {'bce_criterion': loss}
    SRImagesDict = {'SR' : SRImages}
    ##############################################################################################################
    

    return lossDict, SRImagesDict

def inferenceStep(epoch, modelList, dataDict):

    #eval mode
    modelList.NET.eval()
    #modelList.netG.eval()

    #define input data
    LRImages = dataDict['LR']

    #no grad for inference
    with torch.no_grad():
        
        #inference
        SRImages = modelList.NET(LRImages)
        #fake_H_branch, SRImages, grad_LR = modelList.netG(LRImages)
        #return values
        SRImagesDict = {'SR': SRImages} 

    return {}, SRImagesDict



#################################################
###############  EDIT THIS AREA  ################
#################################################
