'''
utils.py
'''
version = "1.33.200721"


#From Python
import argparse
import math
import numpy as np
import apex.amp as amp
import os
import subprocess
import psutil

from apex.parallel import DistributedDataParallel as DDP
from shutil import copyfile

#From Pytorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable


#From This Project
import param as p

from backbone.config import Config





######################################################################################################################################################################## 

# Logging

######################################################################################################################################################################## 


# 텐서보드 관련 초기화
def initTensorboard(ver, subversion):
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == "tensorboard":
            proc.kill()

    logdir = f'./data/{ ver }/log'
    subprocess.Popen(["tensorboard","--logdir=" + logdir,"--port=6006"])
    writer = SummaryWriter(f'{ logdir }/{ subversion }/')

    # if (p.testDataset == 'REDS' or p.testDataset == 'Vid4'):
    #     w4b = SummaryWriter(logdir + "/bicubic_X4_1Frames/")
    #     w4EDVR = SummaryWriter(logdir + "/EDVR_X4_7Frames/")
    # elif (p.testDataset == 'Set5'):
    #     w2 = SummaryWriter(logdir + "/bicubic_X2/")
    #     w3 = SummaryWriter(logdir + "/bicubic_X3/")
    #     w4 = SummaryWriter(logdir + "/bicubic_X4/")
    #     w8 = SummaryWriter(logdir + "/bicubic_X8/")

    return writer



def logValues(writer, valueTuple, iter):
    writer.add_scalar(valueTuple[0], valueTuple[1], iter)

def logImages(writer, imageTuple, iter):
    saveImages = torch.clamp(imageTuple[1], 0, 1)
    for i in range(imageTuple[1].size(0)):
        writer.add_image(imageTuple[0], imageTuple[1][i,:,:,:], iter)




######################################################################################################################################################################## 

# Saves & Loads

######################################################################################################################################################################## 



def loadModels(modelList, version, subversion, loadModelNum, isTest):
    startEpoch = 0
    lastLoss = torch.ones(1)*100
    bestPSNR = 0
    for mdlStr in modelList.getList():
        modelObj = getattr(modelList, mdlStr)
        optimizer = getattr(modelList, mdlStr + "_optimizer") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_optimizer")]) > 0 else None
        scheduler = getattr(modelList, mdlStr + "_scheduler") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_scheduler")]) > 0 else None
       

        modelObj.cuda()




        if (loadModelNum is not 'None' or len([attr for attr in vars(modelList) if attr == (mdlStr+"_pretrained")]) > 0 ): # 로드 할거야
            
            isPretrainedLoad = False
            if optimizer is None:
                isPretrainedLoad = True
            else:
                try:
                    if(loadModelNum == '-1'):
                        checkpoint = torch.load('./data/' + version + '/model/'+subversion+'/' + mdlStr + '.pth')
                    else:
                        checkpoint = torch.load('./data/' + version + '/model/'+subversion+'/' + mdlStr + '-' + loadModelNum+ '.pth')
                except:
                    print("utils.py :: Failed to load saved checkpoints.")
                    if modelList.getPretrainedPath(mdlStr) is not None:
                        isPretrainedLoad = True

            if isPretrainedLoad is True:
                print(f"utils.py :: load pretrained model... : {modelList.getPretrainedPath(mdlStr)}")
                loadPath = modelList.getPretrainedPath(mdlStr)
                checkpoint = torch.load(loadPath)
            
            
            # LOAD MODEL
            '''
            mk = list(modelObj.module.state_dict().keys())
            ck = list(checkpoint.keys())

            for i in range(len(mk)):
                if mk[i] != ck[i]:
                    print(mk[i], ck[i])
            
            '''
                

            try:
                modelObj.load_state_dict(checkpoint['model'],strict=True)
            except:
                try:
                    print("utils.py :: try another method... load model in GLOBAL STRUCTURE mode..")
                    modelObj.load_state_dict(checkpoint ,strict=True)
                except:
                    try:
                        print("utils.py :: try another method... load model in INNER MODEL GLOBAL STRUCTURE mode..")
                        modelObj.module.load_state_dict(checkpoint ,strict=True)
                    except:
                        try:
                            print("utils.py :: model load failed... load model in UNSTRICT mode.. (WARNING : load weights imperfectly)")
                            modelObj.load_state_dict(checkpoint['model'],strict=False)
                        except:
                            try:
                                print("utils.py :: model load failed... load model in GLOBAL STRUCTURE UNSTRICT mode.. (WARNING : load weights imperfectly)")
                                modelObj.load_state_dict(checkpoint ,strict=False)
                            except:
                                print("utils.py :: model load failed..... I'm sorry~")

            

            # LOAD OPTIMIZER
            if optimizer is not None:
                try:
                    optimizer.load_state_dict(checkpoint['optim'])
                    for param_group in optimizer.param_groups: param_group['lr'] = p.learningRate
                except:
                    optimDict = optimizer.state_dict()
                    preTrainedDict = {k: v for k, v in checkpoint.items() if k in optimDict}

                    optimDict.update(preTrainedDict)

            if len([attr for attr in vars(modelList) if attr == (mdlStr+"_pretrained")]) == 0:
            # LOAD VARs..
                try:
                    startEpoch = checkpoint['epoch']
                except:
                    pass#startEpoch = 0

                try:
                    lastLoss = checkpoint['lastLoss']
                except:
                    pass#lastLoss = torch.ones(1)*100
                
                try:
                    bestPSNR = checkpoint['bestPSNR']
                except:
                    pass#bestPSNR = 0
            
            
            if scheduler is not None:
                #scheduler.load_state_dict(checkpoint['scheduler'])
                scheduler.last_epoch = startEpoch
                scheduler.max_lr = p.learningRate
                scheduler.total_steps = p.schedulerPeriod

            try:
                if p.mixedPrecision is True:
                    amp.load_state_dict(checkpoint['amp'])
            except:
                pass

        #modelObj = nn.DataParallel(modelObj)  
        
        paramSize = 0
        for parameter in modelObj.parameters():
            paramSize = paramSize + np.prod(np.array(parameter.size()))
        print(mdlStr + ' : ' + str(paramSize))    

        if (isTest == True):
            modelObj.eval()
        else:
            modelObj.train()

    return startEpoch, lastLoss, bestPSNR
            
def saveModels(modelList, version, subversion, epoch, lastLoss, bestPSNR):

    for mdlStr in modelList.getList():
        modelObj = getattr(modelList, mdlStr)
        optimizer = getattr(modelList, mdlStr + "_optimizer") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_optimizer")]) > 0 else None
        scheduler = getattr(modelList, mdlStr + "_scheduler") if len([attr for attr in vars(modelList) if attr == (mdlStr+"_scheduler")]) > 0 else None

        if optimizer is not None:
            saveData = {}
            saveData.update({'model': modelObj.state_dict()})
            saveData.update({'optim': optimizer.state_dict()})
            if scheduler is not None:
                saveData.update({'scheduler': scheduler.state_dict()})
            saveData.update({'epoch': epoch + 1})
            saveData.update({'lastLoss': lastLoss})
            saveData.update({'bestPSNR': bestPSNR})
            if p.mixedPrecision is True:
                saveData.update({'amp': amp.state_dict()})
            saveData.update({'epoch': epoch + 1})


            torch.save(saveData, './data/'+version+'/model/'+subversion+'/' + mdlStr + '.pth')
            if epoch % p.archiveStep == 0:
                torch.save(saveData, './data/'+version+'/model/'+subversion+'/'+ mdlStr +'-%d.pth' % (epoch + 1))

def saveTensorToNPY(tnsr, fileName):
    #print(tnsr.cpu().numpy())
    np.save(fileName, tnsr.cpu().numpy())

def loadNPYToTensor(fileName):
    #print(np.load(fileName, mmap_mode='r+'))
    return torch.tensor(np.load(fileName, mmap_mode='r+'))







######################################################################################################################################################################## 

# Tensor Calculations

######################################################################################################################################################################## 

#local function
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = x
    if p.valueRangeType == '-1~1':
        out = (x + 1) / 2
    
    return out.clamp(0, 1)


def pSigmoid(input, c1):
    return (1 / (1 + torch.exp(-1 * c1 * input)))


def backproagateAndWeightUpdate(modelList, loss, modelNames = None):

    modelObjs = []
    optimizers = []
    if modelNames is None:
        modelObjs = modelList.getModels()
        optimizers = modelList.getOptimizers()
    elif isinstance(modelNames, (tuple, list)): 
        for mdlStr in modelList.getList():
            if mdlStr in modelNames:
                modelObj = getattr(modelList, mdlStr)
                optimizer = getattr(modelList, mdlStr + '_optimizer')
                modelObjs.append(modelObj)
                optimizers.append(optimizer)
    else:
        modelObjs.append(getattr(modelList, modelNames))
        optimizers.append(getattr(modelList, modelNames + '_optimizer'))


    #init model grad
    for modelObj in modelObjs:
        modelObj.zero_grad()

    #backprop and calculate weight diff
    if p.mixedPrecision == False:
        loss.backward()
    else:
        with amp.scale_loss(loss, optimizers) as scaled_loss:
            scaled_loss.backward()

    #weight update
    for optimizer in optimizers:
        optimizer.step()







######################################################################################################################################################################## 

# Etc.

######################################################################################################################################################################## 


def calculateImagePSNR(a, b):

    pred = a.cpu().data[0].numpy().astype(np.float32)
    gt = b.cpu().data[0].numpy().astype(np.float32)

    np.nan_to_num(pred, copy=False)
    np.nan_to_num(gt, copy=False)

    if p.valueRangeType == '-1~1':
        pred = (pred + 1)/2
        gt = (gt + 1)/2

    if p.colorMode == 'grayscale':
        pred = np.round(pred * 219.)
        pred[pred < 0] = 0
        pred[pred > 219.] = 219.
        pred = pred[0,:,:] + 16
            
        gt = np.round(gt * 219.)
        gt[gt < 0] = 0
        gt[gt > 219.] = 219.
        gt = gt[0,:,:] + 16
    elif p.colorMode == 'color':
        pred = 16 + 65.481*pred[0:1,:,:] + 128.553*pred[1:2,:,:] + 24.966*pred[2:3,:,:]
        pred[pred < 16.] = 16.
        pred[pred > 235.] = 235.

        gt = 16 + 65.481*gt[0:1,:,:] + 128.553*gt[1:2,:,:] + 24.966*gt[2:3,:,:]
        gt[gt < 16.] = 16.
        gt[gt > 235.] = 235.

    

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    #print(20 * math.log10(255.0/ rmse), cv2.PSNR(gt, pred), cv2.PSNR(cv2.imread('sr.png'), cv2.imread('hr.png')))
    return 20 * math.log10(255.0/ rmse)

# 시작시 폴더와 파일들을 지정된 경로로 복사 (백업)
def initFolderAndFiles(ver, subversion):

    subDirList = ['model', 'log', 'result', 'checkpoint']
    list(os.makedirs(f'./data/{ver}/{x}/{subversion}') for x in subDirList if not os.path.exists(f'./data/{ver}/{x}/{subversion}'))

    subDirUnderModelList = ['backbone']
    list(os.makedirs(f'./data/{ver}/model/{subversion}/{x}') for x in subDirUnderModelList if not os.path.exists(f'./data/{ver}/model/{subversion}/{x}'))

    list(copyfile(f'./{x}', f'./data/{ver}/model/{subversion}/{x}') for x in os.listdir('.') if x.endswith('.py'))
    list(copyfile(f'./backbone/{x}', f'./data/{ver}/model/{subversion}/backbone/{x}') for x in os.listdir('./backbone') if x.endswith('.py'))

def initArgParser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inferenceTest', '-it', action='store_true', help="Model Inference")
    parser.add_argument('--load', '-l', nargs='?', default='None', const='-1', help="load 여부")
    parser.add_argument('--irene', '-i', action='store_true', help="키지마세요")
    parser.add_argument('--nosave', '-n', action='store_true', help="epoch마다 validation 과정에 생기는 이미지를 가장 최근 이미지만 저장")
    parser.add_argument('--debug', '-d', action='store_true', help="VS코드 디버그 모드")

    args = parser.parse_args()

    return parser, args
    

def readConfigs():
    Config.readParam("param.yaml")
    Config.readDatasetConfig("datasetConfig.yaml")


def printirene():
    print("	@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%//**,,,,*,**,,*(#&&&&&&@@@@@@@@@@@@@@@@@@@@&&&&%")
    print("	@@@@@@@@@@@@@@@@@@@@@@@@@&(**,,,,,,,,,,,,,,,,,,,,,,,,,*%@@@@@@@@@@@@@@@@(@@&&&&%")
    print("	@@@@@@@@@@@@@@@@@@@@@@@(,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*,(@@@@@@@@@@@@@@&&&&%%")
    print("	@@@@@@@@@@@@@@@@@@@@(*,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*/@@@@@@@@@@@&&&&%%")
    print("	@@@@@@@@@@@@@@@@@#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*/#@@@@@@@@&&&&%#")
    print("	@@@@@@@@@@@@@&/,,,,,.,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,***#@@@@@@&&&&%#")
    print("	@@@@@@@@@@&/,,.,,,,,.,,,,,,,,,,,,,.,,,,,.,,,,,,,,,.,..,.,,**,*,,,,,,*&@@@@&&&%%#")
    print("	@@@@@@@&/,,,,,,,,.,,,,,.,,..,,.,,,.,,,,...        .. .....*(*,,,,,,,,*/@@@&&&%%#")
    print("	@@@@@&*,.,...,.,,.,..,,.,.....,,,...              ,. .....,/****,,,,*,**&@&&&%%#")
    print("	@@@@/,,....,,,........,..,,,.....               .,,***,*,,,,/. ,/**,,,**/&&&&%#(")
    print("	@@%,,..,.,...,.,,....,..,.......  .     ....  ../(/#%%%%%#(//*, .,,,,**,,/&&&%#(")
    print("	@#,,,.,,,,,,,...,,,..,,.,........      .....  .,/(/%&&&%%%%%%%#/. ,,*/****%&%%#/")
    print("	&*,,,,,,,,,......,.,............,.    . .. ....,(*/&&&&&&&&&&&%%%,.,///***%&%%#/")
    print("	*,.,,,,....,....... ,*.,.,...*.,..   .  .......*#(&&&&&&&&&&&&&&&&,,,****/%&%%(/")
    print("	,................ .,,,..,..,,.,..    . ...,....,*(%&&&&&&&&&&&&&&&%,..*,*/&&%#(*")
    print("	,,,............ ..,,,,.,*..*,/... . ...,,..,..**%%(*(/%%&&&&&&&&&@&, .,**/&&&#/,")
    print("	,,............ ,.,,,*,**.,,,..,... ......,.. .*##(%%%&%%%&&&&&&&%&&( .,,*&&&&@/,")
    print("	,,.....  ....,*,.*,,,*,*(,*,,,,..,....,,,.*. ,.. ,.,%&&&&&&&&%#(/**/ .,*#&&&&&&%")
    print("	..  .    ..*,/,./*,,*,,,,*.,,,,.,,..,.,.,,,,,/##(**(%&&&&&&&&&&%&&&%,,,/&&&&&&&&")
    print("	,,,.    ..(*/*,***,*.**,*,,,,**,*...*.*,,,.,*%&&&&&&&&&&&&&&%.  *./.,,(&&&&&&&&&")
    print("	,...   /.(/#*,,*/**,*.,*,,.,*,,,*,..,,,.,**,%&&&&&&&&&&&&&&&%#(/%..,,%&&&&&&&&&&")
    print("	@,.....*/#/*/../*/***/,,.,,**,/,,....,.*,,/(&&&&&&&&&&&%%&&&&&&&%%,(%%&&&&&&&&&&")
    print("	@@*...,#**/**/,/,**,*,,**,,,**/**..,.,,,,*(&&&&&&&&%%%&%&&&&&&&&&&%%%%%&&&&&&&&&")
    print("	@@@(%##*/(,,**,*#/,,,.,,,****/***,.*,,.,*/%&&&&&&&&#%#%###%%&&&&&&##%%%%&&&&&&&&")
    print("	@@@@@%/(,/*,.*,//*,,.   .,*******,,,,.,,/#%&&&&&&&&&&&&&&%&&&&&&&((##%%%&&&&&&&&")
    print("	@@@@@(/(( .***/*,.,        ,***,,.,.*.,*#&%%%&&&&&&&&&&&&&&&&&&&@###%%&%%#/,,.. ")
    print("	@@@&#/*(,,*.(,,  .          .,,,.....,((#%,/#%%&&.,(/,,//#&&&&@@@&&&(,*..,/**/,,")
    print("	@@&#(,**,,//..     .  . ,,(,,*,/,..,,**%%%&&&%&&%#(((//,,#&&&@@&&#  .,/*.....,/.")
    print("	@@##//,,/(.. .,,*,*,**,*//(/*(/,******,#%%%&&&&&&&%#(((#&&&&@&&&*  .  ....... ,.")
    print("	@#%/*(,//,.,*.****(/**#(/#**////#***,..,**(%&&&&&&&&&&&&&%&&&&&%..         .*(#%")
    print("	&&#**,(,/,,*(//(/,*//((,*,,... / ../,,,/((///#%%&&&&&&&%./#%&&#/*/    . .,#%%%%%")
    print("	&&#**(#(*,,,*/***.////,,.//....*...*,,.....,*,,/##%%%%.,.,//*,..*,  *,..,#%%%&%&")
    print("	&/*/(//(,,**,**.,*/,*#*..(.,,.(.,.(/(((###(/%(/,.. .,, .(//..... . ,,.,.*%%%%&&&")
    print("	#**//*.*,,*,./*,*,.,*..//..,#,**(/*,,**(#...     ,.  .**(,.**/////,,.,.*(%%&&&&&")
    print("	**//.,...,./**.,, ./..,,..,(,,*/////(,..*.   .**.  .,,,*.........,,.,*,.(%&%&&&%")
    print("	//*,..,,..**,*.,.  ..(,.,.(.,**,,*,   .*   .,,,  ./*,... .,,,..  ,*/(.,,#%%%%%%%")
    print("	/,,,..*..,**,.,.   .,,.,*.,,(,,,/.   . .  ,,*,... ,  ,*,,.....   ./,/*/. ,##%%%#")
    print("	/.,.,..,,,,,,,,,.. ......,,*,,....  .   .,*,..  ...,.,..        .*,........*(((*")
    print("	,,,,..* /.***,..   ,.....,,..  ..  .....  ... .,.  .,.             .     ......,")
    print("	..,...,,.,*,,.*     .........  ...,(  ..  .,,.*     ,..                      ...")
    print("	....* ,/,,**,   ,    .,.,,.,.. .**.   ,/.. ..  . ...,...                     ...")
    print("	.... ., .,,*...     .,,.,, ..*,  ..     .**,*,...  ..*,...           ,   .   ...")
    print("	. .. .....,...,.     .,*......   ,,     .....      ..........  ...... ,**,./*...")
    print("	.  .. ,/,.,... ...*...,.. (*. .,..    ...         ..   . ..,*..,* ..,,,,,..*(*,,")
    print("	.  .,..,..,...     ...# .....  . .     .        . .,,.,,..    .     #%,., ,..*#*")