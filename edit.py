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

#Gradually warm-up(increasing) learning rate for pytorch's optimizer.
from warmup_scheduler import GradualWarmupScheduler

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
from backbone.module.module import CharbonnierLoss, EdgeLoss
from backbone.structure import Epoch
from dataLoader import DataLoader

from backbone.augmentation import _getSize, _resize
#from backbone.PULSE.stylegan import G_synthesis,G_mapping
#from backbone.PULSE.SphericalOptimizer import SphericalOptimizer
#from backbone.PULSE.loss import LossBuilder


################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)

#version = 'SISR-Implementation2'
#subversion = '1-DRLN_192_train'
#subversion = '1-DeFiAN_inference'

#version = 'NTIRE2021-Track1'
#subversion = '1-High_Deblur(DRLN_DMPHN_Lv4_176)_300_L1_64'
#subversion = '1-High_Deblur(DRLN_DMPHN_Lv4_176)_300_L1_192'
#subversion = '1-High_Deblur(DRLN_DMPHN_Lv4_176)_300_L2_128_end2end'
#subversion = '1-High_Deblur(GFN)_300_L2_256_RDB'
#subversion = '2-NTIRE2021_AFA-Net' #2-NTIRE2021_AFA-Net #2-NTIRE2021_AFA-Net_Deblur #2-NTIRE2021_AFA-Net_SR
#subversion = '2-NTIRE2021_AFA-Net_MPRNet+DeFiAN_all_train'
#subversion = '3-SISR_DeFiAN_REDS_sharp'

# version = 'Deblur-Implementation2'
# subversion = '1-MPRNet_train'


version = 'New_DataLoader_Test'
subversion = '1-SR'


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

        '''
        # SR 1) EDVR        
        self.NET = predefined.EDVR(nf=128, nframes=1, groups=1, front_RBs=5, back_RBs=40)
        self.NET_pretrained = "EDVR-CXR8.pth"  # FaceModel: "EDVR-Face.pth" # satelliteModel : EDVR-DOTA.pth
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.0003)
        '''
        
        # SR 2) DeFiAN
        # L : n_channels = 64, n_blocks = 20, n_modules = 10 / S : n_channels = 32, n_blocks = 10, n_modules = 5
        self.NET = predefined.DeFiAN(n_channels = 64, n_blocks = 20, n_modules = 10, scale = 4, normalize = 1)
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.00003)
        self.NET_pretrained = "DeFiAN_L_x4.pth" # DeFiAN_L_x4.pth # DeFiAN_S_x4.pth
        
        '''
        # SR 3) DRLN
        self.NET = predefined.DRLN(scale=4)
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.0001)
        self.NET_pretrained = "DRLN_BIX4.pth"
        '''


        '''
        # SR 4) SPSR      
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
        '''


        '''
        # Deblur 1) DMPHN
        self.DMPHN_encoder_lv1 = predefined.DMPHN_Encoder()
        self.DMPHN_encoder_lv1_pretrained = "encoder_lv1_4level.pkl"
        self.DMPHN_encoder_lv1_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv1.parameters(), lr=0.0003)
        self.DMPHN_encoder_lv2 = predefined.DMPHN_Encoder()
        self.DMPHN_encoder_lv2_pretrained = "encoder_lv2_4level.pkl"
        self.DMPHN_encoder_lv2_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv2.parameters(), lr=0.0003)
        self.DMPHN_encoder_lv3 = predefined.DMPHN_Encoder()
        self.DMPHN_encoder_lv3_pretrained = "encoder_lv3_4level.pkl"
        self.DMPHN_encoder_lv3_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv3.parameters(), lr=0.0003)
        self.DMPHN_encoder_lv4 = predefined.DMPHN_Encoder() 
        self.DMPHN_encoder_lv4_pretrained = "encoder_lv4_4level.pkl"
        self.DMPHN_encoder_lv4_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv4.parameters(), lr=0.0003)

        self.DMPHN_decoder_lv1 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv1_pretrained = "decoder_lv1_4level.pkl"
        self.DMPHN_decoder_lv1_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv1.parameters(), lr=0.0003)
        self.DMPHN_decoder_lv2 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv2_pretrained = "decoder_lv2_4level.pkl"
        self.DMPHN_decoder_lv2_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv2.parameters(), lr=0.0003)
        self.DMPHN_decoder_lv3 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv3_pretrained = "decoder_lv3_4level.pkl" 
        self.DMPHN_decoder_lv3_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv3.parameters(), lr=0.0003)
        self.DMPHN_decoder_lv4 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv4_pretrained = "decoder_lv4_4level.pkl" 
        self.DMPHN_decoder_lv4_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv4.parameters(), lr=0.0003)
        '''


        '''        
        # Deblur 2) MPRNet
        self.NET = predefined.MPRNet()
        self.NET_pretrained = "MPRNet_pretrained.pth"
        # self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=0.00003)

        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
        num_epochs = 3000
        warmup_epochs = 3
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.NET_optimizer, num_epochs-warmup_epochs, eta_min=1e-6)
        self.NET_scheduler = GradualWarmupScheduler(self.NET_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        '''
        

        '''
        # Deblur+SR 1) GFN
        self.NET = model.GFN()
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=1e-4) # 1e-4 0.0003
        #self.NET_pretrained = "GFN_epoch_55.pkl"
        '''


        
        # Deblur+SR 2) AFA-Net
        # SR
        '''
        # 1) EDVR
        self.SR = predefined.EDVR(128, 1, 1, 5, 40)
        self.SR_pretrained = (
            "EDVR-General.pth"
        )
        self.SR_optimizer = torch.optim.Adam(self.SR.parameters(), lr=0.0003)
        '''

        '''
        # 2) DeFiAN
        self.SR = predefined.DeFiAN(n_channels = 64, n_blocks = 20, n_modules = 10, scale = 4, normalize = 0)
        self.SR_optimizer = torch.optim.Adam(self.SR.parameters(), lr=0.0001)
        self.SR_pretrained = "DeFiAN_L_x4_air.pth"
        self.SR_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.SR_optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        #self.SR_pretrained = "DeFiAN_REDS_61.pth"
        #self.SR_pretrained = "Fusion_SR.pth"
        '''


        '''
        # 3) DRLN
        self.NET = predefined.DRLN(scale=4)
        self.NET_pretrained = (
            "DRLN_BIX4.pt"
        )
        self.NET_optimizer = torch.optim.Adam(self.NET.parameters(), lr=1e-4)
        #self.NET_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.NET_optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        self.NET_scheduler = torch.optim.lr_scheduler.StepLR(self.NET_optimizer, step_size=45, gamma=0.5, last_epoch=-1)
        '''


        # Deblur
        '''
        # 1) DMPHN
        self.DMPHN_encoder_lv1 = predefined.DMPHN_Encoder()
        self.DMPHN_encoder_lv1_pretrained = "DMPHN_encoder_lv1-176_41.pth"
        self.DMPHN_encoder_lv1_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv1.parameters(), lr=0.0003)
        self.DMPHN_encoder_lv2 = predefined.DMPHN_Encoder()
        self.DMPHN_encoder_lv2_pretrained = "DMPHN_encoder_lv2-176_41.pth"
        self.DMPHN_encoder_lv2_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv2.parameters(), lr=0.0003)
        self.DMPHN_encoder_lv3 = predefined.DMPHN_Encoder()
        self.DMPHN_encoder_lv3_pretrained = "DMPHN_encoder_lv3-176_41.pth"
        self.DMPHN_encoder_lv3_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv3.parameters(), lr=0.0003)
        self.DMPHN_encoder_lv4 = predefined.DMPHN_Encoder() 
        self.DMPHN_encoder_lv4_pretrained = "DMPHN_encoder_lv4-176_41.pth"
        self.DMPHN_encoder_lv4_optimizer = torch.optim.Adam(self.DMPHN_encoder_lv4.parameters(), lr=0.0003)

        self.DMPHN_decoder_lv1 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv1_pretrained = "DMPHN_decoder_lv1-176_41.pth"
        self.DMPHN_decoder_lv1_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv1.parameters(), lr=0.0003)
        self.DMPHN_decoder_lv2 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv2_pretrained = "DMPHN_decoder_lv2-176_41.pth"
        self.DMPHN_decoder_lv2_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv2.parameters(), lr=0.0003)
        self.DMPHN_decoder_lv3 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv3_pretrained = "DMPHN_decoder_lv3-176_41.pth" 
        self.DMPHN_decoder_lv3_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv3.parameters(), lr=0.0003)
        self.DMPHN_decoder_lv4 = predefined.DMPHN_Decoder()
        self.DMPHN_decoder_lv4_pretrained = "DMPHN_decoder_lv4-176_41.pth" 
        self.DMPHN_decoder_lv4_optimizer = torch.optim.Adam(self.DMPHN_decoder_lv4.parameters(), lr=0.0003)
        '''

        '''
        # 2) MPRNet
        self.DB = predefined.MPRNet()
        self.DB_pretrained = "MPRNet_pretrained.pth"
        #self.DB_pretrained = "MPRNet_REDS_81.pth"
        self.DB_optimizer = torch.optim.Adam(self.DB.parameters(), lr=2e-4) # torch.optim.Adam(self.DBNET.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8)
        self.DB_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.DB_optimizer, T_0=10, T_mult=1, eta_min=1e-6)
        #self.DB_pretrained = "Fusion_DB.pth"
        '''
        
        # # Fusion
        # self.Fusion = model.GFN_MPRNetDeFiAN()
        # self.Fusion_optimizer = torch.optim.Adam(self.Fusion.parameters(), lr=5e-5) # 1e-4 0.0003
        # #self.Fusion_pretrained = "GFN_epoch_55.pkl"
        # self.Fusion_pretrained = "Fusion_MPR_DeFiAN.pth"
        

        '''
        # AFA
        self.BLENDER_FE = predefined.ResNeSt("269", mode="feature_extractor")
        self.BLENDER_FE_optimizer = torch.optim.Adam(self.BLENDER_FE.parameters(), lr=0.0001)
        self.BLENDER_FE_pretrained = "resnest269.pth"#"resnest269.pth" #BLENDER_FE_ResNeSt200.pth

        self.BLENDER_RES_f4 = model.DeNIQuA_ResDense(None, CW=128, Blocks=9, inFeature=2, outCW=2048, featureCW=2048)
        self.BLENDER_RES_f4_optimizer = torch.optim.Adam(self.BLENDER_RES_f4.parameters(), lr=0.0003)
        #self.BLENDER_RES_f4_pretrained = "BLENDER_RES_f4.pth"
        self.BLENDER_RES_f3 = model.DeNIQuA_ResDense(None, CW=128, Blocks=9, inFeature=2, outCW=1024, featureCW=1024)
        self.BLENDER_RES_f3_optimizer = torch.optim.Adam(self.BLENDER_RES_f3.parameters(), lr=0.0003)
        #self.BLENDER_RES_f3_pretrained = "BLENDER_RES_f3.pth"
        self.BLENDER_RES_f2 = model.DeNIQuA_ResDense(None, CW=128, Blocks=9, inFeature=2, outCW=512, featureCW=512)
        self.BLENDER_RES_f2_optimizer = torch.optim.Adam(self.BLENDER_RES_f2.parameters(), lr=0.0003)
        #self.BLENDER_RES_f2_pretrained = "BLENDER_RES_f2.pth"
        self.BLENDER_RES_f1 = model.DeNIQuA_ResDense(None, CW=128, Blocks=9, inFeature=2, outCW=256, featureCW=256)
        self.BLENDER_RES_f1_optimizer = torch.optim.Adam(self.BLENDER_RES_f1.parameters(), lr=0.0003)
        #self.BLENDER_RES_f1_pretrained = "BLENDER_RES_f1.pth"
        self.BLENDER_DECO = model.tSeNseR_Shuffle(CW=2048)
        self.BLENDER_DECO_optimizer = torch.optim.Adam(self.BLENDER_DECO.parameters(), lr=0.0001)
        #self.BLENDER_DECO_pretrained = "BLENDER_DECO_ResNeSt200.pth" #"BLENDER_DECO.pth" #BLENDER_DECO_ResNeSt200.pth
        '''

        self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()





#################################################################################
#                                     STEPS                                     #
#################################################################################

############ AFA-Net ############
'''
def _deblur(modelList, LRImages):
    images_lv1 = LRImages - 0.5

    H = images_lv1.size(2)
    W = images_lv1.size(3)

    images_lv2_1 = images_lv1[:, :, 0 : int(H / 2), :]
    images_lv2_2 = images_lv1[:, :, int(H / 2) : H, :]
    images_lv3_1 = images_lv2_1[:, :, :, 0 : int(W / 2)]
    images_lv3_2 = images_lv2_1[:, :, :, int(W / 2) : W]
    images_lv3_3 = images_lv2_2[:, :, :, 0 : int(W / 2)]
    images_lv3_4 = images_lv2_2[:, :, :, int(W / 2) : W]

    feature_lv3_1 = modelList.DMPHN_encoder_lv3(images_lv3_1)
    feature_lv3_2 = modelList.DMPHN_encoder_lv3(images_lv3_2)
    feature_lv3_3 = modelList.DMPHN_encoder_lv3(images_lv3_3)
    feature_lv3_4 = modelList.DMPHN_encoder_lv3(images_lv3_4)
    feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
    feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
    feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
    residual_lv3_top = modelList.DMPHN_decoder_lv3(feature_lv3_top)
    residual_lv3_bot = modelList.DMPHN_decoder_lv3(feature_lv3_bot)
    feature_lv2_1 = modelList.DMPHN_encoder_lv2(images_lv2_1 + residual_lv3_top)
    feature_lv2_2 = modelList.DMPHN_encoder_lv2(images_lv2_2 + residual_lv3_bot)
    feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
    residual_lv2 = modelList.DMPHN_decoder_lv2(feature_lv2)
    feature_lv1 = modelList.DMPHN_encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
    return modelList.DMPHN_decoder_lv1(feature_lv1) + 0.5

def _deblurLevel4(modelList, LRImages):
    images_lv1 = LRImages - 0.5

    H = images_lv1.size(2)
    W = images_lv1.size(3)

    
    images_lv2_1 = images_lv1[:, :, 0 : int(H / 2), :]
    images_lv2_2 = images_lv1[:, :, int(H / 2) : H, :]
    images_lv3_1 = images_lv2_1[:, :, :, 0 : int(W / 2)]
    images_lv3_2 = images_lv2_1[:, :, :, int(W / 2) : W]
    images_lv3_3 = images_lv2_2[:, :, :, 0 : int(W / 2)]
    images_lv3_4 = images_lv2_2[:, :, :, int(W / 2) : W]
    images_lv4_1 = images_lv3_1[:, :, 0 : int(H / 4), :]
    images_lv4_2 = images_lv3_1[:, :, int(H/4) : int(H/2), :]
    images_lv4_3 = images_lv3_2[:, : , 0 : int(H/4), :]
    images_lv4_4 = images_lv3_2[:, :, int(H/4) : int(H/2), :]
    images_lv4_5 = images_lv3_3[:, :, 0 : int(H/4), :]
    images_lv4_6 = images_lv3_3[:, :, int(H/4) : int(H/2), :]
    images_lv4_7 = images_lv3_4[:, :, 0 : int(H/4), :]
    images_lv4_8 = images_lv3_4[:, :, int(H/4) : int(H/2), :]
    

    
    #LR for inference
    ## 180 * 320 (H * W) ori
    ## 192 * 320 (H * W) re
    H_2 = int(H / 2) + 6 # for 90 -> 96
    H_4 = int(H_2/2)
    images_lv2_1 = images_lv1[:, :, 0 : H_2, :] # 90 / 96 
    images_lv2_2 = images_lv1[:, :, H_2 : H, :] # 90 180 / 96 192
    images_lv3_1 = images_lv2_1[:, :, :, 0 : int(W / 2)]
    images_lv3_2 = images_lv2_1[:, :, :, int(W / 2) : W]
    images_lv3_3 = images_lv2_2[:, :, :, 0 : int(W / 2)]
    images_lv3_4 = images_lv2_2[:, :, :, int(W / 2) : W]
    images_lv4_1 = images_lv3_1[:, :, 0 : int(H / 4), :] # 45 / 48
    images_lv4_2 = images_lv3_1[:, :, H_4 : H_2, :] # 45 90 / 48 96 
    images_lv4_3 = images_lv3_2[:, : , 0 : H_4, :] # 45 / 48
    images_lv4_4 = images_lv3_2[:, :, H_4 : H_2, :] # 45 90 / 48 96
    images_lv4_5 = images_lv3_3[:, :, 0 : H_4, :] # 45 / 48
    images_lv4_6 = images_lv3_3[:, :, H_4 : H_2, :] # 45 90 / 48 96
    images_lv4_7 = images_lv3_4[:, :, 0 : H_4, :] # 45 / 48
    images_lv4_8 = images_lv3_4[:, :, H_4 : H_2, :] # 45 90 / 48 96
    
    
    feature_lv4_1 = modelList.DMPHN_encoder_lv4(images_lv4_1)
    feature_lv4_2 = modelList.DMPHN_encoder_lv4(images_lv4_2)
    feature_lv4_3 = modelList.DMPHN_encoder_lv4(images_lv4_3)
    feature_lv4_4 = modelList.DMPHN_encoder_lv4(images_lv4_4)
    feature_lv4_5 = modelList.DMPHN_encoder_lv4(images_lv4_5)
    feature_lv4_6 = modelList.DMPHN_encoder_lv4(images_lv4_6)
    feature_lv4_7 = modelList.DMPHN_encoder_lv4(images_lv4_7)
    feature_lv4_8 = modelList.DMPHN_encoder_lv4(images_lv4_8)
    feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
    feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
    feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
    feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
    feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
    feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
    feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
    residual_lv4_top_left = modelList.DMPHN_decoder_lv4(feature_lv4_top_left)
    residual_lv4_top_right = modelList.DMPHN_decoder_lv4(feature_lv4_top_right)
    residual_lv4_bot_left = modelList.DMPHN_decoder_lv4(feature_lv4_bot_left)
    residual_lv4_bot_right = modelList.DMPHN_decoder_lv4(feature_lv4_bot_right)

    feature_lv3_1 = modelList.DMPHN_encoder_lv3(images_lv3_1 + residual_lv4_top_left)
    feature_lv3_2 = modelList.DMPHN_encoder_lv3(images_lv3_2 + residual_lv4_top_right)
    feature_lv3_3 = modelList.DMPHN_encoder_lv3(images_lv3_3 + residual_lv4_bot_left)
    feature_lv3_4 = modelList.DMPHN_encoder_lv3(images_lv3_4 + residual_lv4_bot_right)
    feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
    feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot
    feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
    residual_lv3_top = modelList.DMPHN_decoder_lv3(feature_lv3_top)
    residual_lv3_bot = modelList.DMPHN_decoder_lv3(feature_lv3_bot)

    feature_lv2_1 = modelList.DMPHN_encoder_lv2(images_lv2_1 + residual_lv3_top)
    feature_lv2_2 = modelList.DMPHN_encoder_lv2(images_lv2_2 + residual_lv3_bot)
    feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
    residual_lv2 = modelList.DMPHN_decoder_lv2(feature_lv2)
    feature_lv1 = modelList.DMPHN_encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
    
    return modelList.DMPHN_decoder_lv1(feature_lv1) + 0.5
'''

'''
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

    return modelList.BLENDER_DECO(f1, f2, f3, f4), [f1, f2, f3, f4]
'''

def trainStep(epoch, modelList, dataDict):
    #define loss function
    mse_criterion = nn.MSELoss()   
    l1_criterion = nn.L1Loss()
    Charbonnier_criterion = CharbonnierLoss()
    Edge_criterion = EdgeLoss()
    dists_criterion = DISTS(channels=3).cuda()
    #define input data and label
    LRImages = dataDict['LR']
    HRImages = dataDict['GT']

    '''
    #GPN
    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    #LR_Images_Deblur = F.interpolate(HRImages, size=(cH, cW), mode='bicubic', align_corners=False)
    LRImages_Deblur = LRImages_Deblur.cuda()
    
    modelList.NET.train()
        
    #train
    gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()
    test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()
    
    #inference
    [lr_deblur, SRImages] = modelList.NET(LRImages, gated_Tensor, test_Tensor)

    loss_Deblur = mse_criterion(lr_deblur, LRImages_Deblur)
    loss_SR = mse_criterion(SRImages, HRImages)

    lambda_db = 0.5
    #calculate loss and backpropagation
    loss = loss_SR + lambda_db * loss_Deblur
    backproagateAndWeightUpdate(modelList, loss)

    #return values
    lossDict = {'loss_SR': loss_SR, 'loss_Deblur': loss_Deblur, 'loss': loss}
    SRImagesDict = {'LRImages_Deblur': LRImages_Deblur, 'Deblur' : lr_deblur, 'SR' : SRImages}
    '''

    '''
    # MPRNet
    #train mode
    modelList.NET.train()
    
    # zero_grad
    for param in modelList.NET.parameters():
        param.grad = None
    
    
    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    #LR_Images_Deblur = F.interpolate(HRImages, size=(cH, cW), mode='bicubic', align_corners=False)
    LRImages_Deblur = LRImages_Deblur.cuda()
     

    #inference
    DBImages = modelList.NET(LRImages)
    # Compute loss at each stage
    loss_char = np.sum([Charbonnier_criterion(DBImages[j],LRImages_Deblur) for j in range(3)]) #len(DBImages)
    loss_edge = np.sum([Edge_criterion(DBImages[j],LRImages_Deblur) for j in range(3)]) #len(DBImages)
    loss = (loss_char) + (0.05*loss_edge)
    backproagateAndWeightUpdate(modelList, loss)

    #return values
    lossDict = {'loss_char': loss_char, 'loss_edge': loss_edge, 'loss': loss}
    SRImagesDict = {'Deblur' : DBImages[0], 'LRImages_Deblur' : LRImages_Deblur}
    '''


    
    # EDVR, DeFiAN, DRLN
    #train mode
    modelList.NET.train()
    
    #inference
    SRImages = modelList.NET(LRImages)

    #calculate loss and backpropagation
    loss = mse_criterion(SRImages, HRImages)
    backproagateAndWeightUpdate(modelList, loss)

    #return values
    lossDict = {'MSE': loss}
    SRImagesDict = {'SR' : SRImages}
    

    '''
    #DMPHN
    modelList.DMPHN_encoder_lv1.train()
    modelList.DMPHN_encoder_lv2.train()
    modelList.DMPHN_encoder_lv3.train()
    modelList.DMPHN_encoder_lv4.train()
    modelList.DMPHN_decoder_lv1.train()
    modelList.DMPHN_decoder_lv2.train()
    modelList.DMPHN_decoder_lv3.train()
    modelList.DMPHN_decoder_lv4.train()

    deblurredImages = _deblurLevel4(modelList, LRImages)
    loss = mse_criterion(deblurredImages, HRImages)
    backproagateAndWeightUpdate(modelList, loss)

    #return values
    lossDict = {'MSE': loss}
    SRImagesDict = {'Deblur' : deblurredImages}
    '''



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
    '''

     
    #AFA-Net
    '''
    modelList.SR.train()
    modelList.DB.eval()
    


    modelList.DMPHN_encoder_lv1.train()
    modelList.DMPHN_encoder_lv2.train()
    modelList.DMPHN_encoder_lv3.train()
    modelList.DMPHN_encoder_lv4.train()
    modelList.DMPHN_decoder_lv1.train()
    modelList.DMPHN_decoder_lv2.train()
    modelList.DMPHN_decoder_lv3.train()
    modelList.DMPHN_decoder_lv4.train()
    '''
    
    # modelList.Fusion.train()
    
    '''
    modelList.BLENDER_RES_f1.train()
    modelList.BLENDER_RES_f2.train()
    modelList.BLENDER_RES_f3.train()
    modelList.BLENDER_RES_f4.train()

    modelList.BLENDER_FE.train()
    modelList.BLENDER_DECO.train()
    

    factor = 8
    h,w = LRImages.shape[2], LRImages.shape[3]
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    LRImages = F.pad(LRImages, (0,padw,0,padh), 'reflect')


    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    LRImages_Deblur = LRImages_Deblur.cuda()

    with torch.no_grad():
        # Deblur
        deblurredImages, _, _ = modelList.DB(LRImages)
    
    # SR
    SRImages, SRFeatures = modelList.SR(deblurredImages)
    

    #train
    gated_Tensor = True#torch.cuda.FloatTensor().resize_(1).zero_() + 1
    test_Tensor = False#torch.cuda.FloatTensor().resize_(1).zero_()
    
    #print(deblurredFeatures.shape)
    #print(SRFeatures.shape)
    #print(LRImages.shape)
    #print(deblurredImages.shape)
    #inference
    # [lr_deblur, FusionImages] = modelList.Fusion(LRImages, deblurredFeatures, deblurredImages, SRFeatures, gated_Tensor, test_Tensor)

    #loss_SR = mse_criterion(SRImages, HRImages)
    lossFinal = dists_criterion(SRImages, HRImages)
    # loss_Deblur = mse_criterion(lr_deblur, LRImages_Deblur)
    # loss_Fusion = mse_criterion(FusionImages, HRImages)

    lambda_db = 0.5
    #calculate loss and backpropagation
    # loss = loss_Fusion + lambda_db * loss_Deblur + lambda_db * loss_SR
    loss = lossFinal
    backproagateAndWeightUpdate(modelList, loss)

    #return values
    # lossDict = {'loss_SR': loss_SR, 'loss_Deblur': loss_Deblur, 'loss_Fusion' : loss_Fusion, 'loss': loss}
    # SRImagesDict = {'LRImages_Deblur': LRImages_Deblur, 'Deblur' : lr_deblur, 'SR' : SRImages, 'fusion' : FusionImages}
    lossDict = {'loss': loss}
    SRImagesDict = {'SR' : SRImages}
    '''



    '''
    #with torch.no_grad():
    # SR
    SRImages = modelList.SR(LRImages)
    # Deblur
    #deblurredImages = _deblurLevel4(modelList, LRImages)
    deblurredImages = modelList.DB(LRImages)
    
    #deblurredImages = F.interpolate(deblurredImages, size=SRImages.size()[-2:])  # .data + 0.5
    deblurredSRImages = modelList.SR(deblurredImages[0])  # .data + 0.5    
    blendedImages, _ = _blend(modelList, SRImages, deblurredSRImages)

    #lossDeblur = mse_criterion(deblurredImages, LRImages_Deblur)
    loss_char = np.sum([Charbonnier_criterion(deblurredImages[j],LRImages_Deblur) for j in range(len(deblurredImages))])
    loss_edge = np.sum([Edge_criterion(deblurredImages[j],LRImages_Deblur) for j in range(len(deblurredImages))])
    lossDeblur = (loss_char) + (0.05*loss_edge)
    lossSR = l1_criterion(deblurredSRImages, HRImages)
    lossBlend_MSE = mse_criterion(blendedImages, HRImages)
    #print(lossDeblur.shape)
    #print(lossSR.shape)
    #print(lossBlend_MSE.shape)

    loss = lossDeblur*0.2 + lossSR*0.2  + lossBlend_MSE
    
    #backproagateAndWeightUpdate(modelList, loss)


    backproagateAndWeightUpdate(
        modelList,
        loss,
        modelNames=["BLENDER_RES_f1", "BLENDER_RES_f2", "BLENDER_RES_f3", "BLENDER_RES_f4", "BLENDER_FE", "BLENDER_DECO"]
    )
    
    # return values
    lossDict = {
        "train_SR_MSE": lossSR,
        "train_Deblur_MSE": lossDeblur,
        #"train_Blend_MSE": lossBlend_MSE,
        #"train_Blend_DISTS": lossBlend_DISTS,
        #"train_BlendFeature_MSE": lossBlendFeature_MSE,
        "train_Blend_Total": loss,
    }
    SRImagesDict = {        
        "SR": deblurredSRImages,
        "Deblur": deblurredImages[0],
        "LR_Deblur": LRImages_Deblur,
        "Blend": blendedImages,
    }
    
    '''

    return lossDict, SRImagesDict
     


def validationStep(epoch, modelList, dataDict):


    #define loss function
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    Charbonnier_criterion = CharbonnierLoss()
    Edge_criterion = EdgeLoss()
    dists_criterion = DISTS(channels=3).cuda()
    
    #define input data and label
    LRImages = dataDict['LR']
    HRImages = dataDict['GT']

    '''
    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    #LR_Images_Deblur = F.interpolate(HRImages, size=(cH, cW), mode='bicubic', align_corners=False)
    LRImages_Deblur = LRImages_Deblur.cuda()
    
    modelList.NET.eval()
        
    #train
    #gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
    #test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_() + 1
    gated_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()
    test_Tensor = torch.cuda.FloatTensor().resize_(1).zero_()

    #no grad for validation
    with torch.no_grad():    
        #inference
        [lr_deblur, SRImages] = modelList.NET(LRImages, gated_Tensor, test_Tensor)

        loss_Deblur = mse_criterion(lr_deblur, LRImages_Deblur)
        loss_SR = mse_criterion(SRImages, HRImages)

        lambda_db = 0.5
        #calculate loss and backpropagation
        loss = loss_SR + lambda_db * loss_Deblur

        #return values
        lossDict = {'loss_SR': loss_SR, 'loss_Deblur': loss_Deblur, 'loss': loss}
        SRImagesDict = {'Deblur' : lr_deblur, 'SR' : SRImages}
    '''

    '''
    # MPRNet
    #eval mode
    modelList.NET.eval()
    h, w = LRImages.shape[2], LRImages.shape[3]
    
    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    LRImages_Deblur = LRImages_Deblur.cuda()
    
    #no grad for validation
    with torch.no_grad():    
        #inference
        DBImages = modelList.NET(LRImages)
        
        #DBImages[0] = DBImages[0][:,:,:h,:w]
        #DBImages[1] = DBImages[1][:,:,:h,:w]
        #DBImages[2] = DBImages[2][:,:,:h,:w]
        #DBImages[3] = F.interpolate(DBImages[3], (180,320))

        #LRImages_Deblur = LRImages_Deblur[:,:,:h,:w]
        #calculate loss
        loss_char = np.sum([Charbonnier_criterion(DBImages[j],LRImages_Deblur) for j in range(3)]) # len(DBImages)
        loss_edge = np.sum([Edge_criterion(DBImages[j], LRImages_Deblur) for j in range(3)]) # len(DBImages)
        loss = (loss_char) + (0.05*loss_edge)


        #return values
        lossDict = {'loss_char': loss_char, 'loss_edge': loss_edge, 'loss': loss}
        SRImagesDict = {'Deblur' : DBImages[0], 'LRImages_Deblur' : LRImages_Deblur}

    print(LRImages.shape)
    print(DBImages[0].shape)
    print(LRImages_Deblur.shape)
    '''

    
    # EDVR, DeFiAN, DRLN
    #eval mode
    modelList.NET.eval()

    #no grad for validation
    with torch.no_grad():    
        #inference
        SRImages = modelList.NET(LRImages)

        #calculate loss
        loss = mse_criterion(SRImages, HRImages)

        #return values
        lossDict = {'MSE': loss}
        SRImagesDict = {'SR' : SRImages}
    

    '''
    # DMPHN
    modelList.DMPHN_encoder_lv1.eval()
    modelList.DMPHN_encoder_lv2.eval()
    modelList.DMPHN_encoder_lv3.eval()
    modelList.DMPHN_encoder_lv4.eval()
    modelList.DMPHN_decoder_lv1.eval()
    modelList.DMPHN_decoder_lv2.eval()
    modelList.DMPHN_decoder_lv3.eval()
    modelList.DMPHN_decoder_lv4.eval()

    #no grad for validation
    with torch.no_grad():   
        deblurredImages = _deblurLevel4(modelList, LRImages)
        loss = mse_criterion(deblurredImages, HRImages)

        #return values
        lossDict = {'MSE': loss}
        SRImagesDict = {'Deblur' : deblurredImages}
    '''

    '''
    #SPSR
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
    '''


    '''
    #AFA-Net
    modelList.SR.eval()
    modelList.DB.eval()
    
    
    modelList.DMPHN_encoder_lv1.train()
    modelList.DMPHN_encoder_lv2.train()
    modelList.DMPHN_encoder_lv3.train()
    modelList.DMPHN_encoder_lv4.train()
    modelList.DMPHN_decoder_lv1.train()
    modelList.DMPHN_decoder_lv2.train()
    modelList.DMPHN_decoder_lv3.train()
    modelList.DMPHN_decoder_lv4.train()
    
    
    # modelList.Fusion.eval()


    factor = 8
    h,w = LRImages.shape[2], LRImages.shape[3]
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    LRImages2 = F.pad(LRImages, (0,padw,0,padh), 'reflect')


    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    LRImages_Deblur = LRImages_Deblur.cuda()




    #train
    gated_Tensor = True#torch.cuda.FloatTensor().resize_(1).zero_() + 1
    test_Tensor = False#torch.cuda.FloatTensor().resize_(1).zero_()
    


    #lossFinal = dists_criterion(SRImages_Blended_Final, HRImages)

    #no grad for validation
    with torch.no_grad():    
        # Deblur
        deblurredImages, _, _ = modelList.DB(LRImages2)
        deblurredImages = deblurredImages[:,:,:h,:w]

        SRImages, SRFeatures = modelList.SR(deblurredImages)
        
        #inference
        # [lr_deblur, FusionImages] = modelList.Fusion(LRImages, deblurredFeatures, deblurredImages,  SRFeatures, gated_Tensor, test_Tensor)

        #loss_SR = mse_criterion(SRImages, HRImages)
        lossFinal = dists_criterion(SRImages, HRImages)
        # loss_Deblur = mse_criterion(lr_deblur, LRImages_Deblur)
        # loss_Fusion = mse_criterion(FusionImages, HRImages)

        lambda_db = 0.5
        #calculate loss and backpropagation
        #loss = loss_Fusion + lambda_db * loss_Deblur + lambda_db * loss_Deblur
        loss = lossFinal
        
        #return values
        # lossDict = {'loss_SR': loss_SR, 'loss_Deblur': loss_Deblur, 'loss_Fusion' : loss_Fusion, 'loss': loss}
        # SRImagesDict = {'Deblur' : lr_deblur, 'SR' : SRImages, 'FusionImages' : FusionImages}

        lossDict = {'loss': loss}
        SRImagesDict = {'SR' : SRImages}
    '''
    
    '''
    modelList.SR.eval()
    modelList.DB.eval()

    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()
    
    ####################################################### Preproc. #######################################################
    factor = 8
    h,w = LRImages.shape[2], LRImages.shape[3]
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    LRImages = F.pad(LRImages, (0,padw,0,padh), 'reflect')

    interpolation = 3
    HRImagesTemp = HRImages.cpu()
    _, cH, cW = _getSize(LRImages)
    LRImages_Deblur = torch.cat([_resize(HRImagesTemp[i, :, :, :], cH, cW, interpolation).unsqueeze(0) for i in range(HRImagesTemp.size(0))], 0)
    #LR_Images_Deblur = F.interpolate(HRImages, size=(cH, cW), mode='bicubic', align_corners=False)
    LRImages_Deblur = LRImages_Deblur.cuda()


    with torch.no_grad():
        # SR
        SRImages = modelList.SR(LRImages)

        # DeBlur
        #deblurredImages_small = _deblurLevel4(modelList, LRImages)
        deblurredImages_small = modelList.DB(LRImages)
        loss_char = np.sum([Charbonnier_criterion(deblurredImages_small[j],LRImages_Deblur) for j in range(len(deblurredImages_small))])
        loss_edge = np.sum([Edge_criterion(deblurredImages_small[j],LRImages_Deblur) for j in range(len(deblurredImages_small))])

        
        #deblurredImages = F.interpolate(deblurredImages_small, size=SRImages.size()[-2:])  # .data + 0.5
        #deblurredImages = modelList.SR(deblurredImages_small)  # .data + 0.5
        
        # SR-DEBLUR
        #SR_deblurredImages = _deblurLevel4(modelList, SRImages)

        # DEBLUR-SR
        deblurred_SRImages = modelList.SR(deblurredImages_small[0])

        # BLEND
        blendedImages, _ = _blend(modelList, SRImages, deblurred_SRImages)
    
    lossDeblur = (loss_char) + (0.05*loss_edge)
    loss_SR = l1_criterion(deblurred_SRImages, HRImages)
    #loss_Deblur_MSE = mse_criterion(deblurredImages_small, LRImages_Deblur)
    #loss_SR_Deblur_MSE = mse_criterion(SR_deblurredImages, HRImages)
    #loss_Deblur_SR_MSE = mse_criterion(deblurred_SRImages, HRImages)
    loss_Blend_MSE = mse_criterion(blendedImages, HRImages)

    # return values
    lossDict = {
        "valid_SR": loss_SR,
        "valid_Deblur": lossDeblur,
        #"valid_SR_Deblur_MSE": loss_SR_Deblur_MSE,
        #"valid_Deblur_SR_MSE": loss_Deblur_SR_MSE,
        "valid_Blend_MSE": loss_Blend_MSE,
    }
    SRImagesDict = {
        "SR": deblurred_SRImages,
        "Deblur": deblurredImages_small[0],
        "LR_Deblur": LRImages_Deblur,
        #"SR_Deblur": SR_deblurredImages,
        #"Deblur_SR": deblurred_SRImages,
        "Blend": blendedImages,
    }
    '''
    

    return lossDict, SRImagesDict

def inferenceStep(epoch, modelList, dataDict):

    #define input data
    LRImages = dataDict['LR']
    
    
    modelList.NET.eval()
    #h,w = LRImages.shape[2], LRImages.shape[3]

    with torch.no_grad():
        #SR
        SRImages = modelList.NET(dataDict['LR'])
        # Deblur
        deblurredImages, _, _ = modelList.NET(LRImages)
        #deblurredImages = deblurredImages[:,:,:h,:w]

        SRImagesDict = {'Deblur': deblurredImages}

    '''
    # MPRNet
    factor = 8
    h,w = LRImages.shape[2], LRImages.shape[3]
    H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
    padh = H-h if h%factor!=0 else 0
    padw = W-w if w%factor!=0 else 0
    LRImages2 = F.pad(LRImages, (0,padw,0,padh), 'reflect')
    
    
    # Fusion
    # modelList.SR.eval()
    modelList.DB.eval()
    # modelList.Fusion.eval()

    gated_Tensor = True#torch.cuda.FloatTensor().resize_(1).zero_() + 1
    test_Tensor = False#torch.cuda.FloatTensor().resize_(1).zero_()

    # Deblur -> SR
    with torch.no_grad():
        # Deblur
        deblurredImages, _, _ = modelList.DB(LRImages2)
        deblurredImages = deblurredImages[:,:,:h,:w]
        deblurredImages = F.interpolate(deblurredImages, (180,320))

        #SRImages, SRFeatures = modelList.SR(deblurredImages)

        SRImagesDict = {'Only_Deblur': deblurredImages}

    # with torch.no_grad():
    #     SRImages, SRFeatures = modelList.SR(LRImages)
    #     # Deblur
    #     deblurredImages, _, _, deblurredFeatures = modelList.DB(LRImages2)
    #     deblurredImages = deblurredImages[:,:,:h,:w]
    #     deblurredFeatures = F.interpolate(deblurredFeatures, (180,320))
    #     #inference
    #     [lr_deblur, FusionImages] = modelList.Fusion(LRImages, deblurredFeatures, deblurredImages, SRFeatures, gated_Tensor, test_Tensor)
      
    #     #return values
    #     SRImages = FusionImages

    #     SRImagesDict = {'SR': SRImages}
    '''

    '''
    # EDVR, DeFiAN, DRLN
    #eval mode
    modelList.NET.eval()
    #modelList.netG.eval()

    #no grad for inference
    with torch.no_grad():
        
        #inference
        SRImages = modelList.NET(LRImages)
        # fake_H_branch, SRImages, grad_LR = modelList.netG(LRImages)

        
        # #return values
        # # MPRNet
        # SRImages = SRImages[0]
        # SRImages = SRImages[:,:,:h,:w]
        
        SRImagesDict = {'SR': SRImages}
    '''

    '''
    # model
    modelList.SR.eval()

    modelList.DMPHN_encoder_lv1.eval()
    modelList.DMPHN_encoder_lv2.eval()
    modelList.DMPHN_encoder_lv3.eval()
    modelList.DMPHN_decoder_lv1.eval()
    modelList.DMPHN_decoder_lv2.eval()
    modelList.DMPHN_decoder_lv3.eval()
    
    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()
    
    ####################################################### Preproc. #######################################################

    with torch.no_grad():
        # SR
        SRImages = modelList.SR(LRImages)

        # DeBlur
        deblurredImages_small = _deblur(modelList, LRImages)
        #deblurredImages = F.interpolate(deblurredImages_small, size=SRImages.size()[-2:])  # .data + 0.5
        deblurredImages = modelList.SR(deblurredImages_small)  # .data + 0.5

        # SR-DEBLUR
        SR_deblurredImages = _deblur(modelList, SRImages)

        # DEBLUR-SR
        deblurred_SRImages = modelList.SR(deblurredImages_small)

        # BLEND
        #blendedImages, _ = _blend(modelList, SRImages, deblurredImages)
    
    # return values
    SRImagesDict = {
        "SR": SRImages,
        "Deblur": deblurredImages_small,
        "SR_Deblur": SR_deblurredImages,
        "Deblur_SR": deblurred_SRImages,
        #"Blend": blendedImages,
    }
    '''

    return {}, SRImagesDict
    

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