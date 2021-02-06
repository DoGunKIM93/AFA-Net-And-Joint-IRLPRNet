'''
main.py
'''
mainversion = "3.02.201230"



#FROM Python LIBRARY
import time
import os
import inspect
from importlib import import_module


#FROM PyTorch
import torch
import torch.nn.functional as F

from torchvision.utils import save_image


#from this project
import data_loader as dl
import model
import edit
import backbone.vision as vision
import backbone.utils as utils
import backbone.module.module as module
import backbone.structure as structure

from edit import editversion, ModelList
from backbone.structure import Epoch
from backbone.config import Config












# GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.param.general.GPUNum)

#Arg parser init
parser, args = utils.initArgParser()

#Tensorboard
writer = utils.initTensorboard(edit.version, edit.subversion)

#init Folder & Files
utils.initFolderAndFiles(edit.version, edit.subversion)

#for multiprocessing gpu augmentation in dataloader
#if __name__ == '__main__':torch.multiprocessing.set_start_method('spawn')



#버전 체크
############################################
############################################
print("")
print("")
print("         ProjSR")
print("         Version : " + edit.version)
print("         sub Version : " + edit.subversion)
print("")
print("         -------- FRAMEWORK VERSIONs --------")



#print module version
exList = ['main.py', 'edit.py', 'data_loader_old.py', 'main_old.py', 'test.py', 'inference.py']
pyModuleStrList = list(x[:-3] for x in os.listdir('.') if x not in exList and x.endswith('.py')) + list(f'backbone.{x[:-3]}' for x in os.listdir('./backbone') if x.endswith('.py')) 
pyModuleObjList = list(map(import_module, pyModuleStrList))

versionDict = [['main', mainversion], ['edit', editversion]] + \
            list(map(
            lambda mdlStr, mdlObj: [mdlStr, mdlObj.version] if 'version' in dir(mdlObj) else [mdlStr, '-.--.------'], 
            pyModuleStrList, 
            pyModuleObjList )) + \
            [['config', Config.param.version]]



any(print(f'         {key.ljust(23)} : {val.split(".")[0]}.{val.split(".")[1].rjust(2)}.{val.split(".")[2]}') for key, val in versionDict)


print("") 
print("         ----------- SETTINGs DETAIL ----------")
############################################
############################################





#load DataLoader
if args.inferenceTest == True:
    print(f"load Test Dataset...")
    list(map( lambda k : print(f"    - {k}: {Config.paramDict['data']['dataLoader']['inference'][k]}"), Config.paramDict['data']['dataLoader']['inference']))
    inferenceDataLoader = dl.DataLoader('inference')
    print("")

else:

    print(f"load Train Dataset...")
    list(map( lambda k : print(f"    - {k}: {Config.paramDict['data']['dataLoader']['train'][k]}"), Config.paramDict['data']['dataLoader']['train']))
    trainDataLoader = dl.DataLoader('train')
    print("")

    print(f"load Valid Dataset...")
    list(map( lambda k : print(f"    - {k}: {Config.paramDict['data']['dataLoader']['validation'][k]}"), Config.paramDict['data']['dataLoader']['validation']))
    validationDataLoader = dl.DataLoader('validation')
    print("")
    
print("Dataset loaded.\n")




modelList = ModelList()




print("Load below models...")
startEpoch, metaData = utils.loadModels(modelList, edit.version, edit.subversion, args.load)
print(f"All model loaded. Last Epoch: {startEpoch}")#", Loss: {lastLoss.item():.6f}, BEST Score: {bestScore:.2f} dB")




print("Init Epoch...")
#Define Epochs
if args.inferenceTest == False:
    trainEpoch = Epoch(dataLoader = trainDataLoader,
                        modelList = modelList,
                             step = edit.trainStep,
                  researchVersion = edit.version,
               researchSubVersion = edit.subversion,
                           writer = writer,
                  scoreMetricDict = { 'AE_PSNR': {
                                            'function' : utils.calculateImagePSNR, 
                                        'argDataNames' : ['SR', 'HR'], 
                                        'additionalArgs' : ['$VALUE_RANGE_TYPE', '$COLOR_MODE'],
                                    }}, 
                resultSaveData = ['LR', 'SR', 'HR'] ,
            resultSaveFileName = 'train',
            isNoResultArchiving = args.nosave,
            earlyStopIteration = Config.param.train.step.earlyStopStep,
            name = 'TRAIN')

    validationEpoch = Epoch(dataLoader = validationDataLoader,
                            modelList = modelList,
                                step = edit.validationStep,
                    researchVersion = edit.version,
                    researchSubVersion = edit.subversion,
                                writer = writer,
                    scoreMetricDict = { 'PSNR': {
                                                    'function' : utils.calculateImagePSNR, 
                                                'argDataNames' : ['SR', 'HR'], 
                                            'additionalArgs' : ['$VALUE_RANGE_TYPE', '$COLOR_MODE'],
                                        }},
                        resultSaveData = ['LR', 'SR', 'HR'] ,
                    resultSaveFileName = 'valid/valid',
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name='VAILD')

else:
    inferenceEpoch = Epoch(dataLoader = inferenceDataLoader,
                            modelList = modelList,
                                step = edit.inferenceStep,
                    researchVersion = edit.version,
                researchSubVersion = edit.subversion,
                            writer = writer,
                    scoreMetricDict = { 'PSNR': {
                                                'function' : utils.calculateImagePSNR, 
                                            'argDataNames' : ['LR', 'HR'], 
                                            'additionalArgs' : ['$VALUE_RANGE_TYPE', '$COLOR_MODE'],
                                        }},
                    resultSaveData = ['LR_center', 'SR'] ,
                resultSaveFileName = 'inference',)

print(f"Running...")



if args.inferenceTest == True :
    metaData = inferenceEpoch.run(currentEpoch = 0, metaData = metaData, do_calculateScore = False, do_modelSave = False, do_resultSave='EVERY')

else : 
    for e in range(startEpoch, Config.param.train.step.maxEpoch):
        metaData = trainEpoch.run(currentEpoch = e, metaData = metaData)
        if (e + 1) % Config.param.train.step.validationStep == 0:
            metaData = validationEpoch.run(currentEpoch = e, metaData = metaData, do_calculateScore = 'DETAIL', do_modelSave = False, do_resultSave='EVERY')

        
            





