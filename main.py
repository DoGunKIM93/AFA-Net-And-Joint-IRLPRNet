'''
main.py
'''
mainversion = "1.2.210604"

import inspect
import os

#FROM Python LIBRARY
import time
import warnings
from importlib import import_module

#FROM PyTorch
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from backbone.config import Config
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.param.general.GPUNum)

#from this project
import backbone.module.module as module
import backbone.structure as structure
import backbone.utils as utils
import backbone.vision as vision
import edit
import model
from backbone.structure import Epoch
from edit import ModelList





#Arg parser init
parser, args = utils.initArgParser()

#init Folder & Files
utils.initFolderAndFiles(edit.version, edit.subversion)

#turn on tensorboard process
utils.initTensorboardProcess(edit.version)


############################################
############################################
print("          Version : " + edit.version)
print("          sub Version : " + edit.subversion)
############################################
############################################


print("")
print("")
print("Load below models...")
startEpoch, metaData = utils.loadModels(edit.modelList, edit.version, edit.subversion, args.load)
print(f"All model loaded. Last Epoch: {startEpoch}")#", Loss: {lastLoss.item():.6f}, BEST Score: {bestScore:.2f} dB")


print("")
print("")
print("Init Epoch...")
#Define Epochs
if args.inferenceTest == False:
    trainEpoch = edit.trainEpoch
    validationEpoch = edit.validationEpoch

else:
    inferenceEpoch = edit.inferenceEpoch

    
print("")
print("")
print(f"Running...")



if args.inferenceTest == True :
    metaData = inferenceEpoch.run(currentEpoch = 0, metaData = metaData, do_calculateScore = False, do_modelSave = False, do_resultSave='EVERY')

else : 
    for e in range(startEpoch, Config.param.train.step.maxEpoch):
        metaData = trainEpoch.run(currentEpoch = e, metaData = metaData)
        if (e + 1) % Config.param.train.step.validationStep == 0:
            metaData = validationEpoch.run(currentEpoch = e, metaData = metaData, do_calculateScore = 'DETAIL', do_modelSave = False, do_resultSave='EVERY')
