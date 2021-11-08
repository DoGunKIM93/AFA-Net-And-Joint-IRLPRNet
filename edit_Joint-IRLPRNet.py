'''
edit.py
'''


#FROM Python LIBRARY
import os, sys
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
import predefined
from backbone.utils import backproagateAndWeightUpdate        
from backbone.config import Config
from backbone.structure import Epoch
from dataLoader import DataLoader

from backbone.augmentation import _getSize, _resize

from predefined.loss import CharbonnierLoss, EdgeLoss, Yolo_loss
from yolov3.models.yolo import Model
from yolov3.models.yolo_afa import Model_afa
from yolov3.utils.loss import ComputeLoss
from yolov3.utils.general import labels_to_class_weights, non_max_suppression, scale_coords, xyxy2xywh
from yolov3.utils.datasets import letterbox

################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)
version = 'Object_Detection'
subversion = '1-Joint-IRLPRNet'

# VERSION END (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)
###############################################


#################################################
###############  EDIT THIS AREA  ################
#################################################





class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss
        
#################################################################################
#                                     MODEL                                     #
#################################################################################

# Hyperparameters
import yaml
with open('yolov3/data/hyp.scratch.yaml') as f:
    hyp = yaml.safe_load(f)  # load hyps

device = torch.device('cuda')

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


        
        # AFA-Net
        # PR_SR
        # DeFiAN
        self.SR = predefined.model.SR.DeFiAN(n_channels = 64, n_blocks = 20, n_modules = 20, scale = 4, normalize = False)
        self.SR_pretrained = "DeFiAN-GOMTANG.pth" # DeFiAN_L_x4.pth # DeFiAN_S_x4.pth # DeFiAN_L_x4_air.pth # DeFiAN-GOMTANG.pth
        # self.SR_optimizer = torch.optim.Adam(self.SR.parameters(), lr=0.00001)
        # self.SR_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.SR_optimizer, T_0=10, T_mult=1, eta_min=0.000001)
            


        # PR_Deblur        
        # MPRNet
        self.DB = predefined.model.deblur.MPRNet()
        self.DB_pretrained = "MPRNet_pretrained.pth"        
        # self.DB_optimizer = torch.optim.Adam(self.DB.parameters(), lr=0.00001)
        # self.DB_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.DB_optimizer, T_0=10, T_mult=1, eta_min=0.000001)
                
        # num_epochs = 3000
        # warmup_epochs = 3
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.DB_optimizer, num_epochs-warmup_epochs, eta_min=1e-6)
        # self.DB_scheduler = GradualWarmupScheduler(self.DB_optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
        

        
        # FC
        self.BLENDER_FE = predefined.model.classifier.ResNeSt("200", mode="feature_extractor")
        self.BLENDER_FE_optimizer = torch.optim.Adam(self.BLENDER_FE.parameters(), lr=1E-3)
        self.BLENDER_FE_pretrained = "BLENDER_FE_ResNeSt200-2.pth"

        # IR
        self.BLENDER_RES_f4 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=2048, featureCW=2048)
        self.BLENDER_RES_f4_optimizer = torch.optim.Adam(self.BLENDER_RES_f4.parameters(), lr=1E-3)
        self.BLENDER_RES_f4_pretrained = "BLENDER_RES_f4-201.pth"
        
        self.BLENDER_RES_f3 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=1024, featureCW=1024)
        # self.BLENDER_RES_f3_optimizer = torch.optim.Adam(self.BLENDER_RES_f3.parameters(), lr=1E-3) #0.00001
        self.BLENDER_RES_f3_pretrained = "temp/1-End-to-end_AFA-Net_0_1:1_v2_with_out_syn_with_per_BLENDER_RES_f3-101.pth" # BLENDER_RES_f3-LBLP.pth BLENDER_RES_f3-201.pth

        self.BLENDER_RES_f2 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=512, featureCW=512)
        # self.BLENDER_RES_f2_optimizer = torch.optim.Adam(self.BLENDER_RES_f2.parameters(), lr=1E-3) #0.00001
        self.BLENDER_RES_f2_pretrained = "temp/1-End-to-end_AFA-Net_0_1:1_v2_with_out_syn_with_per_BLENDER_RES_f2-101.pth" # BLENDER_RES_f2-LBLP.pth BLENDER_RES_f2-201.pth

        self.BLENDER_RES_f1 = model.DeNIQuA_Res(None, CW=128, Blocks=9, inFeature=2, outCW=256, featureCW=256)
        # self.BLENDER_RES_f1_optimizer = torch.optim.Adam(self.BLENDER_RES_f1.parameters(), lr=1E-3) #0.00001
        self.BLENDER_RES_f1_pretrained = "temp/1-End-to-end_AFA-Net_0_1:1_v2_with_out_syn_with_per_BLENDER_RES_f1-101.pth" # BLENDER_RES_f1-LBLP.pth BLENDER_RES_f1-201.pth
        
        self.BLENDER_DECO = model.tSeNseR_AFA(CW=2048, inFeatures=1)
        # self.BLENDER_DECO_optimizer = torch.optim.Adam(self.BLENDER_DECO.parameters(), lr=1E-3) #0.00001
        self.BLENDER_DECO_pretrained = "temp/1-End-to-end_AFA-Net_0_1:1_v2_with_out_syn_with_per_BLENDER_DECO-101.pth" # BLENDER_DECO-LBLP.pth BLENDER_DECO_ResNeSt200-2.pth BLENDER_DECO-201.pth
        


        # Object Detection(LPR)
        self.LPR = Model('yolov3/models/yolov3.yaml', ch=3, nc=10, anchors=hyp.get('anchors')).to(device)
        self.LPR_pretrained = "temp/1-End-to-end_AFA-Net_0_1:1_v2_with_out_syn_with_per_LPR-101.pth" # OD-201.pth OD-191.pth utra_yolov3.pt
        # self.LPR_optimizer = torch.optim.Adam(self.LPR.parameters(), lr=1E-3)
        
        nl = self.LPR.model[-1].nl
        nc = 10
        imgsz = 320
        
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        names = [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        self.LPR.nc = nc  # attach number of classes to model
        self.LPR.hyp = hyp  # attach hyperparameters to model
        self.LPR.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.LPR.names = names


        self.initApexAMP() #TODO: migration to Pytorch Native AMP
        self.initDataparallel()





#################################################################################
#                                     STEPS                                     #
#################################################################################

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
    

def trainStep(epoch, modelList, dataDict, i):
    #define loss function
    # Charbonnier_criterion = CharbonnierLoss()
    # Edge_criterion = EdgeLoss()
    # l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    # yolo_criterion = Yolo_loss(device=None, batch=Config.param.data.dataLoader.train.batchSize, n_classes=10)
    # Perceptual_criterion = VGGPerceptualLoss().cuda()
    #define input data and label
    LRImages = dataDict['BLURRY_LR']
    HRImages = dataDict['GT']
    bboxes = dataDict['TXT']

    
    #model mode
    modelList.SR.eval()
    modelList.DB.eval()

    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()
    
    modelList.LPR.train()

    
    with torch.no_grad():
        #SR
        SRImages = modelList.SR(LRImages)
        # print(LRImages.shape)
        # print(SRImages.shape)
        #Deblur
        deblurredImages = modelList.DB(LRImages)
        # print("size=SRImages.size()[-2:] :", SRImages.size()[-2:])
        deblurredImages = F.interpolate(deblurredImages, size=SRImages.size()[-2:])
        blendedImages, blendedFeatures = _blend(modelList, SRImages, deblurredImages)
    
    #calculate loss and backpropagation
    # lossBlend_MSE = mse_criterion(blendedImages, HRImages)
    # lossBlend_Perceptual = Perceptual_criterion(blendedImages, HRImages)
    # loss_AFA_Net = lossBlend_MSE# + lossBlend_Perceptual*0.1
    
    
    # Object Detection(LPR) 
    # predict = modelList.LPR(blendedImages, blendedFeatures)
    predict = modelList.LPR(blendedImages) #blendedImages HRImages
    # print("predict shape : ", predict[0].shape)
    
    # 10 num 5 -> num * 10 6    
    new_data = []
        # 10 th
    for i, bbox in enumerate(bboxes):
        new_bbox  = [[0 for col in range(6)] for row in range(len(bbox))]
        # 4 ~ 6th
        # print(bbox)
        for k, bbo in enumerate(bbox):
            new_bbox[k][0] = i
            new_bbox[k][1:] = bbo
        # print("new_bbox : ", new_bbox)
        new_bbox = torch.Tensor(new_bbox).cuda()    
        new_data.append(new_bbox)
    
    labels_out = torch.cat(new_data, 0)
    # print(labels_out)
    # print("labels_out shape : ", labels_out.shape)

    loss_lpr, loss_items, lbox, lobj, lcls = compute_loss(predict, labels_out)
    loss = loss_lpr# * 0.1 + loss_AFA_Net
    
    # loss = loss_AFA_Net
    #calculate loss and backpropagation    
    backproagateAndWeightUpdate(modelList, loss)#, gradientClipping=1.0) 

    #return values
    lossDict = {'TRAIN_YOLOv3_LOSS' : loss_lpr, 'TRAIN_YOLOv3_lbox': lbox, 'TRAIN_YOLOv3_lobj': lobj, 'TRAIN_YOLOv3_lcls': lcls}
    # SRImagesDict = {
    #     "HR" : HRImages
    # }

    # lossDict = {'TRAIN_AFA_Net_LOSS' : loss, 'TRAIN_AFA_Net_LOSS_mse': lossBlend_MSE}#, 'TRAIN_AFA_Net_LOSS_perceptural': lossBlend_Perceptual*0.1}
    
    # lossDict = {'TRAIN_END2END_LOSS': loss, 'TRAIN_AFA_Net_LOSS' : loss_AFA_Net, 'TRAIN_YOLOv3_LOSS' : loss_lpr, 'TRAIN_YOLOv3_lbox': lbox, 'TRAIN_YOLOv3_lobj': lobj, 'TRAIN_YOLOv3_lcls': lcls}
    SRImagesDict = {
        "LR_BLURRY": LRImages,
        "SR": SRImages, 
        "Deblur": deblurredImages, 
        "Blend": blendedImages,
        "HR" : HRImages
    }

    return lossDict, SRImagesDict
     


def validationStep(epoch, modelList, dataDict, i):
    #define loss function
    mse_criterion = nn.MSELoss()
    # l1_criterion = nn.L1Loss()
    # Charbonnier_criterion = CharbonnierLoss()
    # Edge_criterion = EdgeLoss()
    # yolo_criterion = Yolo_loss(device=None, batch=Config.param.data.dataLoader.validation.batchSize, n_classes=10)
    # Perceptual_criterion = VGGPerceptualLoss().cuda()

    #define input data and label
    LRImages = dataDict['BLURRY_LR']
    HRImages = dataDict['GT']
    # bboxes = dataDict['TXT']

    
    #eval mode
    modelList.SR.eval()
    modelList.DB.eval()
    
    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()
    
    modelList.LPR.eval()

    #no grad for validation
    with torch.no_grad():    
        
        ###### PR_SR
        SRImages = modelList.SR(LRImages)

        ###### PR_DeBlur
        deblurredImages_small = modelList.DB(LRImages)
        deblurredImages = F.interpolate(deblurredImages_small, size=SRImages.size()[-2:])

        # # SR-DEBLUR
        # SR_deblurredImages = modelList.DB(SRImages)

        # # DEBLUR-SR
        # deblurred_SRImages = modelList.SR(deblurredImages_small)

        # BLEND
        blendedImages, blendedFeatures = _blend(modelList, SRImages, deblurredImages)
        
        # bboxes = torch.Tensor(bboxes).cuda()
        # bboxes_pred = modelList.LPR(SRImages, deblurredImages, blendedImages, inference=False)
        
        predict = modelList.LPR(blendedImages, augment = False)[0]
        # predict = modelList.LPR(blendedImages, blendedFeatures, augment = False)[0]
        
        conf_thres = 0.25
        iou_thres = 0.5
        classes = None
        agnostic_nms = False
        max_det = 1000
        save_conf = True
        gn = torch.tensor([320, 320])[[1, 0, 1, 0]]  # normalization gain whwh
        nb = 1

        # lb = []
        # targets = torch.zeros([0])
        # for i, bbox in enumerate(bboxes):
        #     new_bbox  = [[0 for col in range(6)] for row in range(len(bbox))]
        #     # 4 ~ 6th
        #     # print(bbox)
        #     for k, bbo in enumerate(bbox):
        #         new_bbox[k][0] = i
        #         new_bbox[k][1:] = bbo
        #     # print("new_bbox : ", new_bbox)
        
        #     targets = torch.Tensor(new_bbox).cuda() 
        # lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
        
        # Apply NMS
        predict = non_max_suppression(predict, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # print(predict)

        # Process detections
        for j, det in enumerate(predict):  # detections per image
            print("len(det) :", len(det))

            temp_list = []
            if len(det):    
                for *xyxy, conf, cls in reversed(det):
                    temp_string = str(int(cls.item())) + ','  + str(int(xyxy[0].item())) + ',' + str(int(xyxy[1].item())) +  ',' + str(int(xyxy[2].item())) + ',' + str(int(xyxy[3].item())) +  ',' + str(conf.item())
                    # print("temp_string : ", temp_string)
                    temp_list.append(temp_string)

            full_list = []
            for txtString in temp_list:
                full_list.append(txtString.strip().split(','))
            print("before full_list : ", full_list)
            full_list.sort(key=lambda x:int(x[1]))
            print("after full_list : ", full_list)

        '''
        # Process detections
        for j, det in enumerate(predict):  # detections per image
            print("len(det) :", len(det))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print("img.shape[2:], im0.shape : ", blendedImages.shape[2:], blendedImages.shape)
                det[:, :4] = scale_coords(HRImages.shape[2:], det[:, :4], HRImages.shape).round() # blendedImages

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    print(j, line)
                    # with open(txt_path + '.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
        '''
        

    #calculate loss
    lossBlend_MSE = mse_criterion(blendedImages, HRImages) # for run
    # lossBlend_Perceptual = Perceptual_criterion(blendedImages, HRImages)
    loss_AFA_Net = lossBlend_MSE# + lossBlend_Perceptual*0.1

    # loss_od, loss_items, lbox, lobj, lcls = compute_loss(pred, labels_out)
    # loss = loss_AFA_Net * 0.3 + loss_od * 0.7

    lossDict = {'TRAIN_AFA-Net_LOSS' : loss_AFA_Net}
    # lossDict = {'VAL_END2END_LOSS': loss, 'TRAIN_AFA_Net_LOSS' : loss_AFA_Net, 'TRAIN_YOLOv3_LOSS' : loss_od, 'TRAIN_YOLOv3_lbox': lbox, 'TRAIN_YOLOv3_lobj': lobj, 'TRAIN_YOLOv3_lcls': lcls}

    SRImagesDict = {
        "LR_BLURRY": LRImages,
        "SR": SRImages, 
        "Deblur": deblurredImages, 
        "Blend": blendedImages,
        "HR" : HRImages
    }


    return lossDict, SRImagesDict

def inferenceStep(epoch, modelList, dataDict, num):
    #define input data
    # LRImages = dataDict['LR']
    LRImages = dataDict['BLURRY_LR']
    HRImages = dataDict['GT']
    # bboxes = dataDict['TXT']

    
    modelList.SR.eval()
    modelList.DB.eval()
    
    modelList.BLENDER_RES_f1.eval()
    modelList.BLENDER_RES_f2.eval()
    modelList.BLENDER_RES_f3.eval()
    modelList.BLENDER_RES_f4.eval()

    modelList.BLENDER_FE.eval()
    modelList.BLENDER_DECO.eval()
    
    modelList.LPR.eval()
    #h,w = LRImages.shape[2], LRImages.shape[3] 

    with torch.no_grad():
        
        ###### PR_SR
        SRImages = modelList.SR(LRImages)

        ###### PR_DeBlur
        deblurredImages_small = modelList.DB(LRImages)
        deblurredImages = F.interpolate(deblurredImages_small, size=SRImages.size()[-2:])

        # BLEND
        blendedImages, blendedFeatures = _blend(modelList, SRImages, deblurredImages)
        
        predict = modelList.LPR(blendedImages, augment = False)[0]
        # predict = modelList.LPR(blendedImages, blendedFeatures, augment = False)[0]
        
        conf_thres = 0.25
        iou_thres = 0.5
        classes = None
        agnostic_nms = False
        max_det = 1000
        save_conf = True
        gn = torch.tensor([320, 320])[[1, 0, 1, 0]]  # normalization gain whwh
        nb = 1

        # Apply NMS
        predict = non_max_suppression(predict, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        print(predict)



        # Process detections
        for j, det in enumerate(predict):  # detections per image
            print("len(det) :", len(det))

            temp_list = []
            if len(det):    
                for *xyxy, conf, cls in reversed(det):
                    temp_string = str(int(cls.item())) + ','  + str(int(xyxy[0].item())) + ',' + str(int(xyxy[1].item())) +  ',' + str(int(xyxy[2].item())) + ',' + str(int(xyxy[3].item())) +  ',' + str(conf.item())
                    # print("temp_string : ", temp_string)
                    temp_list.append(temp_string)

            full_list = []
            for txtString in temp_list:
                full_list.append(txtString.strip().split(','))
            print("before full_list : ", full_list)
            full_list.sort(key=lambda x:int(x[1]))
            print("after full_list : ", full_list)
            predict_txt = ''
            for txtString in full_list:
                # print(txtString)
                predict = txtString[0]
                # print("predict : ", predict)
                predict_txt = predict_txt + predict
            print(num, predict_txt)

            savePath = f"./data/{version}/result/{subversion}/{'inference'}-{epoch}-{num}.txt"
            print(savePath)
            with open(savePath, 'w') as f:
                f.write(predict_txt)

        '''
        # Process detections
        for j, det in enumerate(predict):  # detections per image
            print("len(det) :", len(det))

            temp_list = []
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print("img.shape[2:], im0.shape : ", blendedImages.shape[2:], blendedImages.shape)
                det[:, :4] = scale_coords(blendedImages.shape[2:], det[:, :4], blendedImages.shape).round() # blendedImages
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # 0 (tensor(5., device='cuda:0'), 0.0062500000931322575, 0.0015625000232830644, 0.0, 0.0031250000465661287, tensor(0.61358, device='cuda:0'))

                    # print(line[0], line[1], line[2], line[3], line[4], line[5])
                    temp_string = str(int(line[0].item())) + ',' + str(line[1]) +  ',' + str(line[2]) + ',' + str(line[3]) +  ',' + str(line[4]) + ',' + str(line[5].item())
                    print("temp_string : ", temp_string)
                    temp_list.append(temp_string)
                    # print(i, line)
        

            full_list = []
            for txtString in temp_list:
                full_list.append(txtString.strip().split(','))
            print("full_list : ", full_list)
            full_list.sort(key=lambda x:x[1])
            predict_txt = ''
            for txtString in full_list:
                # print(txtString)
                predict = txtString[0]
                # print("predict : ", predict)
                predict_txt = predict_txt + predict
            print(num, predict_txt)

            savePath = f"./data/{version}/result/{subversion}/{'inference'}-{epoch}-{num}.txt"
            print(savePath)
            with open(savePath, 'w') as f:
                f.write(predict_txt)
        '''

        SRImagesDict = {
        "LR_BLURRY": LRImages,
        "SR": SRImages, 
        "Deblur": deblurredImages, 
        "Blend": blendedImages,
        "HR" : HRImages
        }   

    # print(predict)
    # modelList.NET.eval()
    # #h,w = LRImages.shape[2], LRImages.shape[3]

    # with torch.no_grad():
    #     #SR
    #     SRImages = modelList.NET(dataDict['LR'])
    #     # Deblur
    #     deblurredImages, _, _ = modelList.NET(LRImages)
    #     #deblurredImages = deblurredImages[:,:,:h,:w]

    #     SRImagesDict = {'Deblur': deblurredImages}


    return {}, SRImagesDict
    

#################################################################################
#                                     EPOCH                                     #
#################################################################################


modelList = ModelList()
compute_loss = ComputeLoss(modelList.LPR)

def initEpoch(train=True, validation=True, inference=True):

    EpochList = []

    if train is True:
        trainEpoch = Epoch( 
                            dataLoader = DataLoader('train'),
                            modelList = modelList,
                            step = trainStep,
                            researchVersion = version,
                            researchSubVersion = subversion,
                            writer = utils.initTensorboardWriter(version, subversion),
                            scoreMetricDict = { 
                                                'TRAIN_PSNR': 
                                                {
                                                    'function' : utils.calculateImagePSNR, 
                                                    'argDataNames' : ['Blend', 'GT'], 
                                                    'additionalArgs' : ['$RANGE'],
                                                },
                                            }, 
                            resultSaveData = ['LR_BLURRY', 'SR', 'Deblur', 'Blend', 'HR'] ,
                            # resultSaveData = ['HR'] ,
                            resultSaveFileName = 'train',
                            isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                            earlyStopIteration = Config.param.train.step.earlyStopStep,
                            name = 'TRAIN',
                            # do_resultSave= False,
                            )
        EpochList.append(trainEpoch)

    if validation is True:
        validationEpoch = Epoch( 
                            dataLoader = DataLoader('validation'),
                            modelList = modelList,
                            step = validationStep,
                            researchVersion = version,
                            researchSubVersion = subversion,
                            writer = utils.initTensorboardWriter(version, subversion),
                            scoreMetricDict = { 
                                                'VAL_PSNR': 
                                                {
                                                    'function' : utils.calculateImagePSNR, 
                                                    'argDataNames' : ['Blend', 'GT'], 
                                                    'additionalArgs' : ['$RANGE'],
                                                },
                                            }, 
                            resultSaveData = ['SR', 'Deblur', 'Blend'] ,
                            # resultSaveData = ['HR'] ,
                            resultSaveFileName = 'validation/validation',
                            isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                            earlyStopIteration = Config.param.train.step.earlyStopStep,
                            name = 'VALIDATION',

                            do_calculateScore = "DETAIL", 
                            do_modelSave = False, 
                            do_resultSave= 'EVERY',
                            )

        EpochList.append(validationEpoch)


    if inference is True:
        inferenceEpoch = Epoch( 
                            dataLoader = DataLoader('inference'),
                            modelList = modelList,
                            step = inferenceStep,
                            researchVersion = version,
                            researchSubVersion = subversion,
                            writer = utils.initTensorboardWriter(version, subversion),
                            scoreMetricDict = {}, 
                            resultSaveData = ['LR_BLURRY', 'SR', 'Deblur', 'Blend', 'HR'] ,
                            # resultSaveData = ['HR'] ,
                            resultSaveFileName = 'inference',
                            isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                            earlyStopIteration = -1, #Config.param.train.step.earlyStopStep,
                            name = 'INFERENCE',

                            do_calculateScore = False, 
                            do_modelSave = False, 
                            do_resultSave='EVERY',
                            )
        EpochList.append(inferenceEpoch)
    
    return tuple(EpochList)



#################################################
###############  EDIT THIS AREA  ################
#################################################
