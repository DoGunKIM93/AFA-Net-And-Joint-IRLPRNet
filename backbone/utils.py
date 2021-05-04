"""
utils.py
"""


# From Python
import argparse
import math
import numpy as np
import os
import subprocess
import psutil
import datetime
from typing import Union, List

try:
    import apex.amp as amp
except:
    pass

# from apex.parallel import DistributedDataParallel as DDP
from shutil import copyfile
from PIL import Image, ImageOps, ImageDraw, ImageFont

# From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from torch.autograd import Variable


# From This Project
from backbone.config import Config
from backbone.torchvision_injected import functional as vF


########################################################################################################################################################################

# Logging

########################################################################################################################################################################


# 텐서보드 관련 초기화
def initTensorboardWriter(ver, subversion):
    logdir = f"./data/{ ver }/log"
    #subprocess.Popen(["tensorboard", "--logdir=" + logdir, "--port=6006"])
    writer = SummaryWriter(f"{ logdir }/{ subversion }/")

    return writer

def initTensorboardProcess(ver):
    logdir = f"./data/{ ver }/"
    #subprocess.Popen(["tensorboard", "--logdir=" + logdir])
            


def logValues(writer, valueTuple, iter):
    writer.add_scalar(valueTuple[0], valueTuple[1], iter)


def logImages(writer, imageTuple, iter):
    saveImages = torch.clamp(imageTuple[1], 0, 1)
    for i in range(imageTuple[1].size(0)):
        writer.add_image(imageTuple[0], imageTuple[1][i, :, :, :], iter)


########################################################################################################################################################################

# Saves & Loads

########################################################################################################################################################################


def addCaptionToImageTensor(imageTensor, caption, valueRange="0~1", DEFAULT_TEXT_SIZE=24):

    assert imageTensor.dim() == 4

    IMAGE_HEIGHT, IMAGE_WIDTH = imageTensor.size()[-2:]

    # DEFAULT_TEXT_SIZE = 24
    TEXT_SIZE = max(12, int(DEFAULT_TEXT_SIZE * (IMAGE_HEIGHT / 256)))
    PAD_LEFT = int(DEFAULT_TEXT_SIZE * 0.5)
    PAD_TOP = int(DEFAULT_TEXT_SIZE * 0.25)
    PAD_BOTTOM = int(DEFAULT_TEXT_SIZE * 0.5)

    '''
    if valueRangeType == "-1~1":
        imageTensor = imageTensor.clamp(-1, 1)
    elif valueRangeType == "0~1":
        imageTensor = imageTensor.clamp(0, 1)
    '''
    imageTensor = imageTensor.clamp(*[float(x) for x in valueRange.replace(' ','').split('~')])

    imageTensor = denorm(imageTensor.cpu(), valueRange)

    ToPILImageFunc = ToPILImage()
    pilImageList = list([ToPILImageFunc(x).convert("RGB") for x in (imageTensor)])

    tmpPilImageList = []
    for pilImage in pilImageList:

        pilImage = ImageOps.expand(pilImage, (0, 0, 0, TEXT_SIZE + PAD_TOP + PAD_BOTTOM))  # L, T, R, B
        ImageDraw.Draw(pilImage).text(
            xy=(PAD_LEFT, IMAGE_HEIGHT + PAD_TOP),
            text=caption,
            fill=(255, 255, 255),
            font=ImageFont.truetype("." + Config.param.save.font.path, TEXT_SIZE),
        )

        tmpPilImageList.append(pilImage)

    return torch.stack(list(vF.to_tensor(x).clamp(0, 1) for x in tmpPilImageList))


def addFrameCaptionToImageSequenceTensor(imageSequenceTensor):
    """
    ADD Frame caption
    &&
    Cat horiziotal
    """
    assert imageSequenceTensor.dim() == 5

    return torch.cat(
        list(addCaptionToImageTensor(x.squeeze(1), f"Frame {i}") for i, x in enumerate(imageSequenceTensor.split(1, dim=1))),
        3,
    )


def saveImageTensorToFile(
    dataDict,
    fileName,
    saveDataDictKeys=None,
    valueRange="0~1",
    interpolation="bicubic",
    caption=True,
):

    rstDict = {}

    MAX_HEIGHT = max(list(x.size(-2) for x in dataDict.values()))
    MAX_WIDTH = max(list(x.size(-1) for x in dataDict.values()))

    if saveDataDictKeys is None or saveDataDictKeys == []:
        saveDataDictKeys = dataDict.keys()

    for k in saveDataDictKeys:
        assert k in dataDict.keys(), f"utils.py :: No key '{k}' in dataDict."

    for name in saveDataDictKeys:
        dim = dataDict[name].dim()

        rstElem = dataDict[name]

        # denorm
        rstElem = denorm(rstElem.cpu(), valueRange)

        # resize to MAX_SIZE both image and imageSequence Tensors
        if dim == 4:
            if rstElem.size(-2) != MAX_HEIGHT or rstElem.size(-1) != MAX_WIDTH:
                rstElem = F.interpolate(rstElem, size=(MAX_HEIGHT, MAX_WIDTH), mode=interpolation)
        elif dim == 5:
            rstElem = torch.cat(
                list(
                    F.interpolate(x.squeeze(1), size=(MAX_HEIGHT, MAX_WIDTH), mode=interpolation).unsqueeze(1)
                    for x in rstElem.split(1, dim=1)
                    if (x.size(-2) != MAX_HEIGHT or x.size(-1) != MAX_WIDTH)
                ),
                1,
            )
            if caption is True:
                rstElem = addFrameCaptionToImageSequenceTensor(rstElem)  # now all rstelems are dim 4

        rstDict.update({name: rstElem})

    # get max height after add frame caption
    MAX_HEIGHT = max(list(x.size(-2) for x in rstDict.values()))

    for name in saveDataDictKeys:
        dim = rstDict[name].dim()
        rstElem = rstDict[name]

        if dim == 4:
            rstElem = F.pad(rstElem, (0, 0, 0, MAX_HEIGHT - rstElem.size(-2)), "constant", 0)

        if caption is True:
            rstElem = addCaptionToImageTensor(rstElem, caption=name, valueRange='0~1')
        rstDict[name] = rstElem

    assert len(rstDict) != 0, "utils.py :: There is no save data...Make sure saveDataDictKeys in dataDict.keys()"

    rst = torch.cat(list(rstDict.values()), 3)

    save_image(rst, fileName, normalize=True)


def loadModels(modelList, version, subversion, loadModelNum):
    startEpoch = 0
    lastLoss = torch.ones(1) * 100
    bestPSNR = 0
    metaData = None
    for mdlStr in modelList.getList():
        modelObj = getattr(modelList, mdlStr)
        optimizer = (
            getattr(modelList, mdlStr + "_optimizer")
            if len([attr for attr in vars(modelList) if attr == (mdlStr + "_optimizer")]) > 0
            else None
        )
        scheduler = (
            getattr(modelList, mdlStr + "_scheduler")
            if len([attr for attr in vars(modelList) if attr == (mdlStr + "_scheduler")]) > 0
            else None
        )

        #if model has DO_NOT_FORCE_CUDA -> CPU
        if 'DO_NOT_FORCE_CUDA' in dir(modelObj) and modelObj.DO_NOT_FORCE_CUDA is True:
            modelObj.cpu()
        else:
            modelObj#.cuda()

        checkpoint = None
        if (
            loadModelNum != "None" or len([attr for attr in vars(modelList) if attr == (mdlStr + "_pretrained")]) > 0
        ):  # 로드 할거야

            isPretrainedLoad = False
            if optimizer is None:
                if modelList.getPretrainedPath(mdlStr) is not None:
                    isPretrainedLoad = True
            else:
                try:
                    if loadModelNum == "-1":
                        checkpoint = torch.load("./data/" + version + "/checkpoint/" + subversion + "/" + mdlStr + ".pth")
                    else:
                        checkpoint = torch.load(
                            "./data/" + version + "/checkpoint/" + subversion + "/" + mdlStr + "-" + loadModelNum + ".pth"
                        )
                except:
                    print("utils.py :: Failed to load saved checkpoints.")
                    if modelList.getPretrainedPath(mdlStr) is not None:
                        isPretrainedLoad = True

            if isPretrainedLoad is True:
                print(f"utils.py :: load pretrained model... : {modelList.getPretrainedPath(mdlStr)}")
                loadPath = modelList.getPretrainedPath(mdlStr)
                checkpoint = torch.load(loadPath)

            # LOAD MODEL
            """
            mk = list(modelObj.module.state_dict().keys())
            ck = list(checkpoint.keys())

            for i in range(len(mk)):
                if mk[i] != ck[i]:
                    print(mk[i], ck[i])
            
            """
            if checkpoint is not None:
                while(True):
                    try:
                        mthd = "NORMAL"
                        modelObj.load_state_dict(checkpoint["model"], strict=True)
                        break
                    except:
                        pass
                    try:
                        mthd = "GLOBAL STRUCTURE"
                        modelObj.load_state_dict(checkpoint, strict=True)
                        break
                    except:
                        pass
                    try:
                        mthd = "NORMAL state_dict"
                        modelObj.load_state_dict(checkpoint["state_dict"], strict=True)
                        break
                    except:
                        pass
                    try:
                        mthd = "INNER MODEL"
                        modelObj.module.load_state_dict(checkpoint["model"], strict=True)
                        break
                    except:
                        pass
                    try:
                        mthd = "INNER MODEL GLOBAL STRUCTURE"
                        modelObj.module.load_state_dict(checkpoint, strict=True)
                        break
                    except:
                        pass
                    try:
                        mthd = "INNER NORMAL state_dict"
                        modelObj.module.load_state_dict(checkpoint["state_dict"], strict=True)
                        break
                    except:
                        pass
                    try:
                        mthd = "UNSTRICT (WARNING : load weights imperfectly)"
                        modelObj.load_state_dict(checkpoint["model"], strict=False)
                        break
                    except:
                        pass
                    try:
                        mthd = "GLOBAL STRUCTURE UNSTRICT (WARNING : load weights imperfectly)"
                        modelObj.load_state_dict(checkpoint, strict=False)
                        break
                    except:
                        mthd = "FAILED"
                        print("utils.py :: model load failed..... I'm sorry~")
                        break

                print(f"{mdlStr} Loaded with {mthd} mode." if mthd != "FAILED" else f"{mdlStr} Load Failed.")

                # LOAD OPTIMIZER
                if optimizer is not None:
                    try:
                        lrList = [param_group["lr"] for param_group in optimizer.param_groups]
                        assert [lrList[0]] * len(
                            lrList
                        ) == lrList, (
                            "utils.py :: Error, optimizer has different values of learning rates. Tell me... I'll fix it."
                        )
                        lr = lrList[0]

                        optimizer.load_state_dict(checkpoint["optim"])
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    except:
                        optimDict = optimizer.state_dict()
                        preTrainedDict = {k: v for k, v in checkpoint.items() if k in optimDict}

                        optimDict.update(preTrainedDict)

                        if scheduler is not None:
                            # scheduler.load_state_dict(checkpoint['scheduler'])
                            scheduler.last_epoch = startEpoch
                            scheduler.max_lr = lr
                            # scheduler.total_steps = p.schedulerPeriod

                # if len([attr for attr in vars(modelList) if attr == (mdlStr+"_pretrained")]) == 0:
                # LOAD VARs..

                if isPretrainedLoad is False:
                    try:
                        metaData = checkpoint["metaData"]
                    except:
                        print(
                            "utils.py :: Failed to load meta data. It may caused by loading old models. (before main.py version 3.00) "
                        )

                try:
                    if isPretrainedLoad is False:
                        startEpoch = checkpoint["epoch"] if checkpoint["epoch"] > startEpoch else startEpoch
                except:
                    print("utils.py :: error to load last epoch")

                try:
                    if Config.param.train.method.mixedPrecision is True:
                        amp.load_state_dict(checkpoint["amp"])
                except:
                    print(
                        "utils.py :: WARNING : Failed to load APEX AMP checkpoint data. It may cause unexpected training behaviour."
                    )

        paramSize = 0
        for parameter in modelObj.parameters():
            paramSize = paramSize + np.prod(np.array(parameter.size()))
        print(mdlStr + " : " + str(paramSize))

    return startEpoch, metaData


def saveModels(modelList, version, subversion, epoch, metaData):

    for mdlStr in modelList.getList():
        modelObj = getattr(modelList, mdlStr)
        optimizer = (
            getattr(modelList, mdlStr + "_optimizer")
            if len([attr for attr in vars(modelList) if attr == (mdlStr + "_optimizer")]) > 0
            else None
        )
        scheduler = (
            getattr(modelList, mdlStr + "_scheduler")
            if len([attr for attr in vars(modelList) if attr == (mdlStr + "_scheduler")]) > 0
            else None
        )

        if optimizer is not None:
            saveData = {}
            saveData.update({"model": modelObj.state_dict()})
            saveData.update({"optim": optimizer.state_dict()})
            if scheduler is not None:
                saveData.update({"scheduler": scheduler.state_dict()})
            saveData.update({"metaData": metaData})
            if Config.param.train.method.mixedPrecision is True:
                saveData.update({"amp": amp.state_dict()})
            saveData.update({"epoch": epoch + 1})

            torch.save(saveData, "./data/" + version + "/checkpoint/" + subversion + "/" + mdlStr + ".pth")
            if epoch % Config.param.train.step.archiveStep == 0:
                torch.save(
                    saveData, "./data/" + version + "/checkpoint/" + subversion + "/" + mdlStr + "-%d.pth" % (epoch + 1)
                )


def saveTensorToNPY(tnsr, fileName):
    # print(tnsr.cpu().numpy())
    np.save(fileName, tnsr.cpu().numpy())


def loadNPYToTensor(fileName):
    # print(np.load(fileName, mmap_mode='r+'))
    return torch.tensor(np.load(fileName, mmap_mode="r+"))


def loadNPY(fileName):
    # print(np.load(fileName, mmap_mode='r+'))
    return np.load(fileName, mmap_mode="r+")


########################################################################################################################################################################

# Tensor Calculations

########################################################################################################################################################################

# local function
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def denorm(tnsr_or_nparray, valueRange):
    '''
    make tensor to 0~1 Float32 Tensor
    '''
    if isinstance(tnsr_or_nparray, torch.Tensor):
        rangeMin, rangeMax = [float(x) if type(tnsr_or_nparray).is_floating_point is True else int(x) for x in valueRange.replace(' ','').split('~')]
        out = tnsr_or_nparray.to(torch.float32)
        out = (out - rangeMin) / (rangeMax - rangeMin)
        out.clamp(0, 1)
    else:
        rangeMin, rangeMax = [float(x) for x in valueRange.replace(' ','').split('~')]
        out = tnsr_or_nparray
        out = (out - rangeMin) / (rangeMax - rangeMin)

    return out


def pSigmoid(input, c1):
    return 1 / (1 + torch.exp(-1 * c1 * input))


def backproagateAndWeightUpdate(modelList, loss, modelNames=None, gradientClipping=False, retainGraph=False):

    assert gradientClipping is False or str(gradientClipping).replace('.','').isdigit() or isinstance(gradientClipping, List), "utils.py :: gradientClipping argument must be 'False' or Number, or its list."

    modelObjs = []
    optimizers = []
    if modelNames is None:
        modelObjs = modelList.getModels()
        optimizers = modelList.getOptimizers()
    elif isinstance(modelNames, (tuple, list)):
        for mdlStr in modelList.getList():
            if mdlStr in modelNames:
                modelObj = getattr(modelList, mdlStr)
                optimizer = getattr(modelList, mdlStr + "_optimizer")
                modelObjs.append(modelObj)
                optimizers.append(optimizer)
    else:
        modelObjs.append(getattr(modelList, modelNames))
        optimizers.append(getattr(modelList, modelNames + "_optimizer"))

    # init model grad
    for modelObj in modelObjs:
        modelObj.zero_grad()

    # backprop and calculate weight diff
    if Config.param.train.method.mixedPrecision == False:
        loss.backward(retain_graph=retainGraph)
    else:
        with amp.scale_loss(loss, optimizers) as scaled_loss:
            scaled_loss.backward(retain_graph=retainGraph)

    # grad Clipping
    if isinstance(gradientClipping, List) is False:
        gradientClipping = [gradientClipping] * len(modelObjs)
    assert len(gradientClipping) == len(modelObjs), "utils.py :: If gradientClipping argument fed as a list, the length if gradientClipping list must be same length of the model list."
    for modelObj, gCElem in zip(modelObjs, gradientClipping):
        if gCElem is not False:
            nn.utils.clip_grad_norm_(modelObj.parameters(), gCElem)
    
    # weight update
    for optimizer in optimizers:
        optimizer.step()


########################################################################################################################################################################

# Etc.

########################################################################################################################################################################

def calculateImageRecoverPercentage(inp, outp, gt, lossFunction='MSE', interpolate='bicubic'):
    LOSSFUNC_DICT = {'MSE': nn.MSELoss,
                     'L1': nn.L1Loss}
    assert lossFunction in LOSSFUNC_DICT.keys()

    with torch.no_grad():
        criterion = LOSSFUNC_DICT[lossFunction]()

        if inp.size() != outp.size():
            inp = F.interpolate(inp, outp.size()[-2:], mode=interpolate)

        inpLoss = criterion(inp, gt)
        outpLoss = criterion(outp, gt)

        RP = inpLoss / outpLoss

    return RP



# TODO: BATCH
def calculateImagePSNR(a, b, valueRange, colorMode='color', colorSpace = 'YCbCr'):
    assert colorSpace in ['YCbCr', 'RGB'], 'calculateImagePSNR must be one of "YCbCr", "RGB"'

    if a.size(-3) == 1:
        colorMode = 'grayScale'
    elif a.size(-3) == 3:
        colorMode = 'color'

    pred = a.cpu().data[0].numpy().astype(np.float32)
    gt = b.cpu().data[0].numpy().astype(np.float32)

    np.nan_to_num(pred, copy=False)
    np.nan_to_num(gt, copy=False)

    '''
    if valueRangeType == "-1~1":
        pred = (pred + 1) / 2
        gt = (gt + 1) / 2
    '''

    pred = denorm(pred, valueRange)
    gt = denorm(gt, valueRange)

    if colorSpace == "YCbCr":
        if colorMode == "grayscale":
            pred = np.round(pred * 219.0)
            pred[pred < 0] = 0
            pred[pred > 219.0] = 219.0
            pred = pred[0, :, :] + 16

            gt = np.round(gt * 219.0)
            gt[gt < 0] = 0
            gt[gt > 219.0] = 219.0
            gt = gt[0, :, :] + 16
        elif colorMode == "color":
            pred = 16 + 65.481 * pred[0:1, :, :] + 128.553 * pred[1:2, :, :] + 24.966 * pred[2:3, :, :]
            pred[pred < 16.0] = 16.0
            pred[pred > 235.0] = 235.0

            gt = 16 + 65.481 * gt[0:1, :, :] + 128.553 * gt[1:2, :, :] + 24.966 * gt[2:3, :, :]
            gt[gt < 16.0] = 16.0
            gt[gt > 235.0] = 235.0 
    elif colorSpace == "RGB":
        pred = np.round(pred * 255.0)
        gt = np.round(gt * 255.0)

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    # print(20 * math.log10(255.0/ rmse), cv2.PSNR(gt, pred), cv2.PSNR(cv2.imread('sr.png'), cv2.imread('hr.png')))
    return 20 * math.log10(255.0 / rmse)


# 시작시 폴더와 파일들을 지정된 경로로 복사 (백업)
def initFolderAndFiles(ver, subversion):

    subDirList = ["model", "log", "result", "checkpoint"]

    list(
        os.makedirs(f"./data/{ver}/{x}/{subversion}")
        for x in subDirList
        if not os.path.exists(f"./data/{ver}/{x}/{subversion}")
    )

    EXT_LIST = ['.py','.yaml','.cu','.cpp','txt']
    EXCEPT_PATH = ['./data/']

    fileList = []
    folderList = set()
    # r=root, d=directories, f = files
    for r, d, f in os.walk("."):
        for file in f:
            if file.lower().endswith(tuple(EXT_LIST)) and not r.startswith(tuple(EXCEPT_PATH)):
                folderList.add(r)
                fileList.append(os.path.join(r, file))

    folderList = sorted(list(folderList))

    TIME_STR = str(datetime.datetime.today()).replace(' ','_').replace(':','-').replace('.','_')
    BACKUP_PATH = f'./data/{ver}/model/{subversion}/' + TIME_STR

    os.makedirs(BACKUP_PATH)
    [os.makedirs(BACKUP_PATH + x[1:], exist_ok=True) for x in folderList]

    [copyfile(x, BACKUP_PATH + x[1:]) for x in fileList]



def initArgParser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inferenceTest", "-it", action="store_true", help="Model Inference")
    parser.add_argument("--load", "-l", nargs="?", default="None", const="-1", help="load 여부")
    parser.add_argument("--debug", "-d", action="store_true", help="VS코드 디버그 모드")
    parser.add_argument("--tag", "-t", default="None", help="Tag")

    args = parser.parse_args()

    return parser, args
