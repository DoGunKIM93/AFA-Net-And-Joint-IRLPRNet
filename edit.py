'''
edit.py
'''
editversion = "1.16.201014"


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
import backbone.module as module
from backbone.utils import loadModels, saveModels, backproagateAndWeightUpdate        
from backbone.config import Config
from backbone.SPSR.loss import GANLoss, GradientPenaltyLoss


#from backbone.PULSE.stylegan import G_synthesis,G_mapping
#from backbone.PULSE.SphericalOptimizer import SphericalOptimizer
#from backbone.PULSE.loss import LossBuilder



################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)

version = '42-OASIS'
subversion = '4-test'

# VERSION END (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)
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
    

        self.FE = model.SPSR_FeatureExtractor()
        
        self.OASIS = model.OASIS(featureExtractor = self.FE, Upscaler = model.Mirage, UpscalerArgs = [128, 4], patchSize = 17, scaleFactor = 4, modelNum = 64)
        self.OASIS_optimizer = torch.optim.Adam(self.OASIS.parameters(), lr=0.0003, weight_decay=0, betas=(0.9, 0.999))


        '''
        self.SR_A = model.SPSR_Generator(scaleFactor = 4)
        self.SR_A_pretrained = "SPSR-Robust-netG-General.pth"#"SPSR-baseline-netG.pth" #  # FaceModel: "EDVR-Face.pth"

        self.SR_B = model.SPSR_Generator(scaleFactor = 4)
        self.SR_B_pretrained = "SPSR-netG-Face.pth" 



        self.GRAND = model.SPSR_Generator(scaleFactor = 4)
        self.GRAND_optimizer = torch.optim.Adam(self.GRAND.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))

        self.GRAND_Disc = model.SPSR_Discriminator(size = 128)
        self.GRAND_Disc_optimizer = torch.optim.Adam(self.GRAND_Disc.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))

        self.GRAND_Grad_Disc = model.SPSR_Discriminator(size = 128)
        self.GRAND_Grad_Disc_optimizer = torch.optim.Adam(self.GRAND_Grad_Disc.parameters(), lr=0.0001, weight_decay=0, betas=(0.9, 0.999))

        self.GRAND_Feature = model.SPSR_FeatureExtractor()
        self.GRAND_Get_gradient = model.SPSR_Get_gradient()
        self.GRAND_Get_gradient_nopadding = model.SPSR_Get_gradient_nopadding()




        self.FEATUREMAPPER_A = model.DeNIQuA_Res(featureExtractor = None, CW=64, Blocks=9, inFeature=1, outCW=128, featureCW=128)
        self.FEATUREMAPPER_A_optimizer = torch.optim.Adam(self.FEATUREMAPPER_A.parameters(), lr=0.0003)

        self.FEATUREMAPPER_B = model.DeNIQuA_Res(featureExtractor = None, CW=64, Blocks=9, inFeature=1, outCW=128, featureCW=128)
        self.FEATUREMAPPER_B_optimizer = torch.optim.Adam(self.FEATUREMAPPER_B.parameters(), lr=0.0003)



        self.BLENDER = model.DeNIQuA_Res(featureExtractor = None, CW=64, Blocks=18, inFeature=2, outCW=128, featureCW=128)
        self.BLENDER_optimizer = torch.optim.Adam(self.BLENDER.parameters(), lr=0.0003)
        '''



        self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()


def trainStep(epoch, modelList, dataDict):

    LRImages = dataDict['LR']
    HRImages = dataDict['HR']

    #4. Blending
    SCALE_FACTOR = 4

    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = module.CharbonnierLoss(eps=1e-3)
    bce_criterion = nn.BCELoss()
    #ssim_criterion = MS_SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
    #dists_criterion = DISTS(channels=3).cuda()
 
    # model
    #odelList.SR_A.eval()
    #odelList.SR_B.eval()

    #odelList.GRAND.train()
    #odelList.GRAND_Disc.train()
    #odelList.GRAND_Grad_Disc.train()

    #odelList.FEATUREMAPPER_A.train()
    #odelList.FEATUREMAPPER_B.train()

    #odelList.BLENDER.train()


    # batch size
    batchSize = LRImages.size(0)

    SRImages = modelList.OASIS(LRImages, isTrain = False)
    #HRTrunc = HRImages[:,:,8*4:9*4,8*4:9*4]

    loss = mse_criterion(SRImages, HRImages)

    backproagateAndWeightUpdate(modelList, loss, "OASIS")

    lossDict = {'mse':loss}
    SRImagesDict = {'SR':SRImages}#, 'HRT':HRTrunc}
    '''
    ####################################################### Make Domain SR Features & Images #######################################################
    with torch.no_grad():
        feature_A = modelList.SR_A(LRImages, mode='encode')
        feature_B = modelList.SR_B(LRImages, mode='encode')
        fc = modelList.GRAND(HRImages, mode='encode')
        print(feature_A.size(), fc.size())

        SRImages_A = modelList.SR_A(feature_A, mode='decode')
        SRImages_B = modelList.SR_B(feature_B, mode='decode')






    ####################################################### Train to Construct GRAND LATENT SPACE #######################################################

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
    get_grad = modelList.GRAND_Get_gradient
    get_grad_nopadding = modelList.GRAND_Get_gradient_nopadding


    # Optimizing
    fake_H_branch, fake_H, grad_LR = modelList.GRAND(HRImages)
    
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
            real_fea = modelList.GRAND_Feature(HRImages)
            fake_fea = modelList.GRAND_Feature(fake_H)
            l_g_fea = l_fea_w * cri_fea(fake_fea, real_fea)
            l_g_total += l_g_fea
        
        if cri_pix_grad: #gradient pixel loss
            l_g_pix_grad = l_pix_grad_w * cri_pix_grad(fake_H_grad, var_H_grad)
            l_g_total += l_g_pix_grad


        if cri_pix_branch: #branch pixel loss
            l_g_pix_grad_branch = l_pix_branch_w * cri_pix_branch(fake_H_branch, var_H_grad_nopadding)
            l_g_total += l_g_pix_grad_branch


        # G gan + cls loss
        pred_g_fake = modelList.GRAND_Disc(fake_H)
        pred_d_real = modelList.GRAND_Disc(HRImages).detach()
        
        l_g_gan = l_gan_w * (cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
        l_g_total += l_g_gan

        # grad G gan + cls loss
        pred_g_fake_grad = modelList.GRAND_Grad_Disc(fake_H_grad)
        pred_d_real_grad = modelList.GRAND_Grad_Disc(var_ref_grad).detach()

        l_g_gan_grad = l_gan_grad_w * (cri_grad_gan(pred_d_real_grad - torch.mean(pred_g_fake_grad), False) + 
                                            cri_grad_gan(pred_g_fake_grad - torch.mean(pred_d_real_grad), True)) /2
        l_g_total += l_g_gan_grad

        backproagateAndWeightUpdate(modelList, l_g_total, "GRAND")


    # D
    l_d_total = 0

    pred_d_real = modelList.GRAND_Disc(HRImages)
    pred_d_fake = modelList.GRAND_Disc(fake_H.detach())  # detach to avoid BP to G

    l_d_real = cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
    l_d_fake = cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

    l_d_total = (l_d_real + l_d_fake) / 2

    if gan_type == 'wgan-gp':
        if random_pt.size(0) != batch:
            random_pt.resize_(batch, 1, 1, 1)
        random_pt.uniform_()  # Draw random interpolation points
        interp = random_pt * fake_H.detach() + (1 - random_pt) * HRImages
        interp.requires_grad = True
        interp_crit, _ = modelList.GRAND_Disc(interp)
        l_d_gp = l_gp_w * cri_gp(interp, interp_crit) 
        l_d_total += l_d_gp

    backproagateAndWeightUpdate(modelList, l_d_total, "GRAND_Disc")

    

    # D_grad
    pred_d_real_grad = modelList.GRAND_Grad_Disc(var_ref_grad)
    pred_d_fake_grad = modelList.GRAND_Grad_Disc(fake_H_grad.detach())  # detach to avoid BP to G
    
    l_d_real_grad = cri_grad_gan(pred_d_real_grad - torch.mean(pred_d_fake_grad), True)
    l_d_fake_grad = cri_grad_gan(pred_d_fake_grad - torch.mean(pred_d_real_grad), False)

    l_d_total_grad = (l_d_real_grad + l_d_fake_grad) / 2

    backproagateAndWeightUpdate(modelList, l_d_total_grad, "GRAND_Grad_Disc")




    SRImages = fake_H

    loss_GRAND = torch.as_tensor(l_g_total)
    loss_GRAND_Disc = torch.as_tensor(l_d_total)
    loss_GRAND_Grad_Disc = torch.as_tensor(l_d_total_grad)


    lossDict = {'G':loss_GRAND, 'D':loss_GRAND_Disc, 'D_grad':loss_GRAND_Grad_Disc}

    SRImagesDict = {'AutoEncoded' : SRImages}
    '''






















    '''

    IQAMap_E = modelList.BLENDER_E_DECO([feature_Entire, feature_Face])
    IQAMap_F = modelList.BLENDER_F_DECO([feature_Entire, feature_Face])

        
    #SPPXL_CNT = int(SRImages_Entire.size(-2) * SRImages_Entire.size(-1) / 256)
    #print(SPPXL_CNT)

    superPixelMap = None#vision.SLIC(SRImages_Entire, SPPXL_CNT, 10, True)

    blendedFeature_E, _ = vision.E2EBlending(feature_Entire, feature_Face, IQAMap_E, superPixelMap, softMode = True)
    blendedFeature_F, _ = vision.E2EBlending(feature_Entire, feature_Face, IQAMap_F, superPixelMap, softMode = True)

    SRImages_Blended_Entire = modelList.SR(blendedFeature_E, mode='decode')
    SRImages_Blended_Face = modelList.SR_FACE(blendedFeature_F, mode='decode')



    IQAMap = modelList.BLENDER_DECO([SRImages_Blended_Entire, SRImages_Blended_Face])
    SRImages_Blended_Final, sm = vision.E2EBlending(SRImages_Blended_Entire, SRImages_Blended_Face, IQAMap, superPixelMap, softMode = True)

    #with torch.no_grad():
    #    blendedHard, smHard = vision.E2EBlending(SRImages_Entire, SRImages_Face, IQAMap, superPixelMap, softMode = False)


    lossEntire = dists_criterion(SRImages_Blended_Entire, HRImages)
    lossFace = dists_criterion(SRImages_Blended_Face, HRImages)
    lossFinal = dists_criterion(SRImages_Blended_Final, HRImages)

    loss = lossEntire + lossFace + lossFinal




    backproagateAndWeightUpdate(modelList, loss, modelNames = ["BLENDER_E_DECO", "BLENDER_F_DECO", "BLENDER_DECO", "BLENDER_FE"])
    #backproagateAndWeightUpdate(modelList, lossFace, modelNames = ["BLENDER_F_DECO"])

    # return losses
    lossList = [lossEntire, lossFace, lossFinal]
    SRImagesList = [SRImages_Entire, SRImages_Face,
                SRImages_Blended_Entire, SRImages_Blended_Face, sm, SRImages_Blended_Final]

    # Update All model weights
    # if modelNames = None, this function updates all trainable model.
    # if modelNames is a String, updates one model
    # if modelNames is a List of string, updates those models.
    # backproagateAndWeightUpdate(modelList, loss, modelNames = "Ensemble")
    # backproagateAndWeightUpdate(modelList, loss, modelNames = "NET")

    # return losses
    # lossList = [srAdversarialLoss, hrAdversarialLoss, loss_disc, loss_pixelwise, loss_adversarial, loss]
    # return List of Result Images (also you can add some intermediate results).
    # SRImagesList = [gaussianSprayKernel,bImage1,bImage2,blendedImages]
    
    '''
    return lossDict, SRImagesDict
     


def validationStep(epoch, modelList, dataDict):

    LRImages = dataDict['LR']
    HRImages = dataDict['HR']
    
    batchSize = LRImages.size(0)

    # 1. ESPCN
    '''
    mse_criterion = nn.MSELoss()
    modelList.ESPCN.eval()
    SRImages = modelList.ESPCN(LRImages)
    loss = mse_criterion(SRImages, HRImages)  
    '''

    # 2. EDVR(S)
    '''
    cpl_criterion = utils.CharbonnierLoss(eps=1e-3)
    modelList.EDVR.eval()
    SRImages = modelList.EDVR(LRImages)
    loss = cpl_criterion(SRImages, HRImages)
    '''

    # 3. SPSR
    '''
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


    #SEQUENCE_LENGTH = Config.param.data.dataLoader.train.sequenceLength
    #lossList = [srAdversarialLoss, hrAdversarialLoss, loss_disc, loss_pixelwise, loss_adversarial, loss]
    # return List of Result Images (also you can add some intermediate results).
    #SRImagesList = [gaussianSprayKernel,bImage1,bImage2,blendedImages] 
    '''
    


    #4. Blending

    SCALE_FACTOR = 4

    # loss
    mse_criterion = nn.MSELoss()
    cpl_criterion = module.CharbonnierLoss(eps=1e-3)
    bce_criterion = nn.BCELoss()
    #ssim_criterion = MS_SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=False)
 
    # model
    modelList.SR.eval()
    modelList.SR_FACE.eval()

    #modelList.BLENDER_FE.eval()
    modelList.BLENDER_E_DECO.eval()
    modelList.BLENDER_F_DECO.eval()
    modelList.BLENDER_DECO.eval()
    modelList.BLENDER_FE.eval()


    # batch size
    batchSize = LRImages.size(0)
    


    ####################################################### Preproc. #######################################################
    # SR Processing
    

    

    

    

    with torch.no_grad():
        feature_Entire = modelList.SR(LRImages, 'encode')
        feature_Face = modelList.SR_FACE(LRImages, 'encode')

        SRImages_Entire = modelList.SR(feature_Entire, 'decode')
        SRImages_Face = modelList.SR_FACE(feature_Face, 'decode')
        bigLR = F.interpolate(LRImages, scale_factor=SCALE_FACTOR, mode='bicubic')

        IQAMap_E = modelList.BLENDER_E_DECO([feature_Entire, feature_Face])
        IQAMap_F = modelList.BLENDER_F_DECO([feature_Entire, feature_Face])
        
        #SPPXL_CNT = SRImages_Entire.size(-2) * SRImages_Entire.size(-1) / 2000
        superPixelMap = None#vision.SLIC(SRImages_Entire, SPPXL_CNT, 10, True)

        blendedFeature_E, _ = vision.E2EBlending(feature_Entire, feature_Face, IQAMap_E, superPixelMap, softMode = True)
        blendedFeature_F, _ = vision.E2EBlending(feature_Entire, feature_Face, IQAMap_F, superPixelMap, softMode = True)

        SRImages_Blended_Entire = modelList.SR(blendedFeature_E, 'decode')
        SRImages_Blended_Face = modelList.SR_FACE(blendedFeature_F, 'decode')

        IQAMap = modelList.BLENDER_DECO([SRImages_Blended_Entire, SRImages_Blended_Face])
        SRImages_Blended_Final, sm = vision.E2EBlending(SRImages_Blended_Entire, SRImages_Blended_Face, IQAMap, superPixelMap, softMode = True)

    lossEntire = mse_criterion(SRImages_Blended_Entire, HRImages)
    lossFace = mse_criterion(SRImages_Blended_Face, HRImages)
    lossFinal = mse_criterion(SRImages_Blended_Final, HRImages)


    loss = lossEntire + lossFace + lossFinal


    SRImagesList = [SRImages_Entire, SRImages_Face,
                    SRImages_Blended_Entire, SRImages_Blended_Face, sm, SRImages_Blended_Final]

    
    return loss, SRImagesList

def inferenceStep(epoch, modelList, dataDict):

    LRImages = dataDict['LR']

    # model
    # 1. ESPCN
    # modelList.ESPCN.eval()
    
    # 2. EDVR(S)
    '''
    modelList.EDVR.eval()
    SRImages = modelList.EDVR(LRImages)
    SRImagesList = [SRImages] 
    '''

    # 3. SPSR
    
    #gan_type = "vanilla"
    #cri_gan = GANLoss(gan_type, 1.0, 0.0)
    
    '''
    modelList.netG.eval()
    with torch.no_grad():
        _, SRImages, _ = modelList.netG(LRImages)
    '''
    modelList.EDVR.eval()
    SRImages = modelList.EDVR(LRImages)



    '''
    ####################################################### Preproc. #######################################################
    # SR Processing
    with torch.no_grad():
        #SRImages_ESPCN = modelList.ESPCN(LRImages)
        SRImages_EDVR = modelList.EDVR(LRImages)
        #fake_H_branch, SRImages_SPSR, grad_LR = modelList.netG(LRImages)
    modelList.netG.train()

    ####################################################### Blending. #######################################################
    if p.blendingMode != None:
        destination = SRImages_Entire
        source = SRImages_Face
        
        # test를 위한 roi (Detection 추가하면 roi 매핑 부분 수정 예정)
        roi = (1300, 700, 400, 400, 400, 900, 500, 300, 300, 400, 300, 500)
        blended_img = vision.BlendingMethod(p.blendingMode, source, destination, roi)
        SRImages_Ensembled = blended_imgzxcjh
    else:
        SRImages_Ensembled = SRImages_Entire

    # return List of Result Images (also you can add some intermediate results).
    '''
    SRImagesList = [SRImages] 
    
    
    return SRImagesList

#################################################
###############  EDIT THIS AREA  ################
#################################################
