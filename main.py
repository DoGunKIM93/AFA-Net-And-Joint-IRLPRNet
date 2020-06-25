'''
main.py
'''
mainversion = "1.51.200624"


#FROM Python LIBRARY
import time
import csv
import os
import math
import numpy as np
import sys
import argparse
import subprocess
import psutil

from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter

#FROM PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image



#from this project
from data_loader import SRDataLoader
import data_loader as dl
import backbone.vision as vision
import model
import param as p
import backbone.utils as utils
from edit import editversion, version, subversion, trainStep, inferenceStep, ModelList
import backbone.module as module
import backbone.structure as structure



# 시작시 폴더와 파일들을 지정된 경로로 복사 (백업)
# 텐서보드 관련 초기화
def initFolderAndFiles():
    
    if not os.path.exists('data/' + version):
        os.makedirs('./data/' + version)

    if not os.path.exists('./data/' + version + '/model'):
        os.makedirs('./data/' + version + '/model')
    if not os.path.exists('./data/' + version + '/result'):
        os.makedirs('./data/' + version + '/result')
    if not os.path.exists('./data/' + version + '/log'):
        os.makedirs('./data/' + version + '/log')
    

    
    if not os.path.exists('./data/' + version + '/model/'+ subversion):
        os.makedirs('./data/' + version + '/model/'+ subversion)
    if not os.path.exists('./data/' + version + '/result/'+ subversion):
        os.makedirs('./data/' + version + '/result/'+ subversion)
    if not os.path.exists('./data/' + version + '/log/' + subversion):
        os.makedirs('./data/' + version + '/log/' + subversion)
    if not os.path.exists('./data/' + version + '/model/'+ subversion + '/backbone'):
        os.makedirs('./data/' + version + '/model/'+ subversion + '/backbone')

    if args.debug is True:
        copyfile('./' + 'main.py', './data/' + version + '/model/' + subversion+'/'+ 'main.py')
    else:
        copyfile('./' + sys.argv[0], './data/' + version + '/model/' + subversion+'/'+ sys.argv[0])

    copyfile('./data_loader.py', './data/' + version + '/model/'+ subversion +'/data_loader.py')
    copyfile('./backbone/vision.py', './data/' + version + '/model/' + subversion +'/backbone/vision.py')
    copyfile('./model.py', './data/' + version + '/model/' + subversion + '/model.py')
    copyfile('./param.py', './data/' + version + '/model/'+ subversion +'/param.py')
    copyfile('./backbone/utils.py', './data/' + version + '/model/'+ subversion +'/backbone/utils.py')
    copyfile('./backbone/structure.py', './data/' + version + '/model/'+ subversion +'/backbone/structure.py')
    copyfile('./edit.py', './data/' + version + '/model/'+ subversion +'/edit.py')













# GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"]=p.GPUNum

#Arg parser init
parser = argparse.ArgumentParser()

parser.add_argument('--test', '-t', action='store_true', help="test 여부")
parser.add_argument('--load', '-l', nargs='?', default='None', const='-1', help="load 여부")
parser.add_argument('--irene', '-i', action='store_true', help="키지마세요")
parser.add_argument('--nosave', '-n', action='store_true', help="epoch마다 validation 과정에 생기는 이미지를 가장 최근 이미지만 저장")
parser.add_argument('--debug', '-d', action='store_true', help="VS코드 디버그 모드")

args = parser.parse_args()





#Tensorboard
for proc in psutil.process_iter():
    # check whether the process name matches
    if proc.name() == "tensorboard":
        proc.kill()

logdir = './data/' + version + '/log'
subprocess.Popen(["tensorboard","--logdir=" + logdir,"--port=6006"])
writer = SummaryWriter(logdir + "/" + subversion + "/")

if (p.testDataset == 'REDS' or p.testDataset == 'Vid4'):
    w4b = SummaryWriter(logdir + "/bicubic_X4_1Frames/")
    w4EDVR = SummaryWriter(logdir + "/EDVR_X4_7Frames/")
elif (p.testDataset == 'Set5'):
    w2 = SummaryWriter(logdir + "/bicubic_X2/")
    w3 = SummaryWriter(logdir + "/bicubic_X3/")
    w4 = SummaryWriter(logdir + "/bicubic_X4/")
    w8 = SummaryWriter(logdir + "/bicubic_X8/")

#init Folder & Files
initFolderAndFiles()



#버전 체크
############################################
############################################
print("")
if args.irene == True : utils.printirene()
print("")
print("         ProjSR")
print("         Version : " + version)
print("         sub Version : " + subversion)
print("")
print("         ----FRAMEWORK VERSIONs----")
print("         main Version       : " + mainversion)
print("         edit Version       : " + editversion)
print("         dataloader Version : " + dl.version)
print("         vision Version     : " + vision.version)
print("         model Version      : " + model.version)
print("         module Version     : " + module.version)
print("         utils Version      : " + utils.version)
print("         structure Version  : " + structure.version)
print("         param Version      : " + p.version)
print("")
print("         ------SETTING DETAIL------")
############################################
############################################






#load DataLoader
if args.test == True:
    print(f"load Test Dataset... {p.testDataset} / {p.colorMode} X{p.scaleFactor} ({p.testScaleMethod})")
    testDataLoader = SRDataLoader(dataset   = p.testDataset,
                              datasetType   = p.testDatasetType,
                              dataPath      = p.dataPath,
                              scaleFactor   = p.scaleFactor,
                              scaleMethod   = p.testScaleMethod,
                              batchSize     = 1,
                              mode          = 'test',
                              sameOutputSize= p.sameOutputSize,
                              colorMode     = p.colorMode)
else:
    print(f"load Train Dataset... {p.trainDataset} / {p.colorMode} X{p.scaleFactor} ({p.trainScaleMethod})")
    trainDataLoader = SRDataLoader(dataset  = p.trainDataset,
                              datasetType   = p.trainDatasetType,
                              dataPath      = p.dataPath,
                              scaleFactor   = p.scaleFactor,
                              scaleMethod   = p.trainScaleMethod,
                              batchSize     = p.batchSize,
                              mode          = 'train',
                              cropSize      = p.cropSize,
                              sameOutputSize= p.sameOutputSize,
                              colorMode     = p.colorMode)

    print(f"load Valid Dataset... {p.testDataset} / {p.colorMode} X{p.scaleFactor} ({p.testScaleMethod})")
    validDataLoader = SRDataLoader(dataset   = p.testDataset,
                              datasetType   = p.testDatasetType,
                              dataPath      = p.dataPath,
                              scaleFactor   = p.scaleFactor,
                              scaleMethod   = p.testScaleMethod,
                              batchSize     = 1,
                              mode          = 'test',
                              sameOutputSize= p.sameOutputSize,
                              colorMode     = p.colorMode)
print("Dataset loaded.\n")





modelList = ModelList()

bestPSNR = 0

print("Load below models")
startEpoch, lastLoss, bestPSNR = utils.loadModels(modelList, version, subversion, args.load, args.test)
print(f"All model loaded. Last Epoch: {startEpoch}, Loss: {lastLoss.item():.6f}, BEST PSNR: {bestPSNR:.2f} dB")




for epoch in range(startEpoch, p.MaxEpoch):


    # ============= Train =============#
    # ============= Train =============#
    # ============= Train =============#
    # 1배치당 시간
    a = time.perf_counter()
    # 1에폭당 시간
    b = time.perf_counter()

    finali = 0
    PSNR = 0
    GlobalPSNRCount = 0

    Avgloss = [torch.zeros(1)]*256

    
    for i, Imagepairs in enumerate(trainDataLoader):

        #if i == 200: break

        LRImages = []
        HRImages = []
        for _LRi, _HRi in Imagepairs:
            LRImages.append(_LRi)
            HRImages.append(_HRi)
        LRImages = torch.cat(LRImages, 0)
        HRImages = torch.cat(HRImages, 0)
        LRImages = utils.to_var(LRImages)
        HRImages = utils.to_var(HRImages)
        batchSize = LRImages.size(0)
                
        ########### train STEP ############
        lossList, SRImagesList = trainStep(epoch, modelList, LRImages, HRImages)
        ###################################
        SRImages = SRImagesList[-1]
        if len(LRImages.size()) == 5:
            PSNR += utils.calculateImagePSNR(SRImages,HRImages[:,p.sequenceLength//2,:,:,:])
        else:
            PSNR += utils.calculateImagePSNR(SRImages,HRImages)
        
        GlobalPSNRCount += 1

        for lossIdx, loss in enumerate(lossList):
            Avgloss[lossIdx] = Avgloss[lossIdx] + torch.Tensor.item(loss.data)
            
        finali = i + 1

        if (i + 1) % 1 == 0:
            olda = a
            a = time.perf_counter()
            print('                      E[%d/%d][%.2f%%] NET:'
                    % (epoch, p.MaxEpoch, (i + 1) / (len(trainDataLoader.dataset) / p.batchSize / 100)),  end=" ")

            # print('loss:%.5f' % (Avgloss[0]/finali), end = " ")
            print('loss: [', end="")
            for lossIdx, _ in enumerate(lossList):
                print(f'{torch.Tensor.item(Avgloss[lossIdx].data)/finali:.5f}, ', end="")
            print('] ', end="")

            print('lr: [',  end="")

            for mdlStr in modelList.getList():
                if len([attr for attr in vars(modelList) if attr == (mdlStr+"_scheduler")]) > 0:
                    schd = getattr(modelList, mdlStr+"_scheduler")
                    print(f"{mdlStr}: {schd.get_lr()[0]:.6f} ",  end="")
                elif len([attr for attr in vars(modelList) if attr == (mdlStr+"_optimizer")]) > 0:
                    print(f"{mdlStr}: {p.learningRate:.6f} ",  end="")
            print(f"] time: {(a - olda):.2f} sec    ", end="\r")


    Avgloss[:] = [x / finali for x in Avgloss]
    
    oldb = b
    b = time.perf_counter()      

    print('E[%d/%d] NET:'
            % (epoch, p.MaxEpoch),  end=" ")

    #print('loss: %.5f PSNR: %.2f dB' % (Avgloss[0], PSNR/GlobalPSNRCount), end = " ")
    print('loss: [', end="")
    for lossIdx, _ in enumerate(lossList):
                print(f'{torch.Tensor.item(Avgloss[lossIdx].data):.5f}, ', end="")
    print(f'] PSNR: {PSNR/GlobalPSNRCount:.2f}dB', end=" ")
    

    print('lr: [ ',  end="")

    for mdlStr in modelList.getList():
        if len([attr for attr in vars(modelList) if attr == (mdlStr+"_scheduler")]) > 0:
            schd = getattr(modelList, mdlStr+"_scheduler")
            print(f"{mdlStr}: {schd.get_lr()[0]:.6f} ",  end="")
        elif len([attr for attr in vars(modelList) if attr == (mdlStr+"_optimizer")]) > 0:
            print(f"{mdlStr}: {p.learningRate:.6f} ",  end="")
    print(f"] time: {(b - oldb):.2f} sec                    ")



    print('saving model, ', end="")
    if (p.trainAccidentCoef is None or Avgloss[0] < lastLoss * p.trainAccidentCoef):
        utils.saveModels(modelList, version, subversion, epoch, Avgloss[0], bestPSNR)
        lastLoss = Avgloss[0]
    else:
        print('Something is Wrong... load last model and restart Epoch.')
        startEpoch, lastLoss, bestPSNR = utils.loadModels(modelList, version, subversion, '-1', args.test)
        continue

    print('log, ', end="")
    
    # Save loss log
    #utils.logValues(writer, ['train_loss', Avgloss[0].item()], epoch)
    for lossIdx, _ in enumerate(lossList):
        utils.logValues(writer, [f'train_loss_{lossIdx}', Avgloss[lossIdx].item()], epoch)

    utils.logValues(writer, ['train_PSNR', PSNR/GlobalPSNRCount], epoch)
    

    print('output images.')
    # Save sampled images
    copyCoff = 1
    if len(LRImages.size()) == 5:
        #LRImages = torch.cat(LRImages.cpu().split(1, dim=1),4).squeeze(1)
        LRImages = LRImages[:,p.sequenceLength//2,:,:,:]
        HRImages = HRImages[:,p.sequenceLength//2,:,:,:]
        copyCoff = 1#p.sequenceLength
    lr_images = utils.denorm(LRImages.cpu().view(LRImages.size(0), 1 if p.colorMode=='grayscale' else 3, LRImages.size(2), LRImages.size(3)))
    hr_images = utils.denorm(HRImages.cpu().view(HRImages.size(0), 1 if p.colorMode=='grayscale' else 3, HRImages.size(2), HRImages.size(3)))

    for i, si in enumerate(SRImagesList):
        if (si.size(2) != HRImages.size(2) or si.size(3) != HRImages.size(3)):
            SRImagesList[i] = F.interpolate(si, size=(HRImages.size(2),HRImages.size(3)), mode='bicubic')

    SRImages = torch.cat(SRImagesList, 3)

    sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if p.colorMode=='grayscale' else 3, SRImages.size(2), SRImages.size(3)))

    if p.sameOutputSize == False:
        cated_images = torch.cat((nn.functional.interpolate(lr_images.data, size=(HRImages.size(2),HRImages.size(3) * copyCoff), mode='bicubic'),
                            sr_images.data,
                            hr_images.data
                            ),3)    
    else:
        cated_images = torch.cat(( lr_images.data,
                            sr_images.data,
                            hr_images.data
                            ),3)

    if args.nosave :        
        savePath = './data/'+version+'/result/'+subversion+'/SRed_train_images.png'
    else :
        savePath = './data/'+version+'/result/'+subversion+'/SRed_train_images-' + str(epoch + 1) + '.png'
        utils.logImages(writer, ['train_images', cated_images], epoch)

    save_image(cated_images, savePath)

    

    

    if (epoch + 1) % p.validStep == 0:
        # ============= Valid =============#
        # ============= Valid =============#
        # ============= Valid =============#
        # 1배치당 시간
        a = time.perf_counter()
        # 1에폭당 시간
        b = time.perf_counter()

        finali = 0
        PSNR = 0
        GlobalPSNRCount = 0

        Avgloss = [torch.zeros(1)]*256

        
        for i, Imagepairs in enumerate(validDataLoader):

            with torch.no_grad():
                LRImages = []
                HRImages = []
                for _LRi, _HRi in Imagepairs:
                    LRImages.append(_LRi)
                    HRImages.append(_HRi)
                LRImages = torch.cat(LRImages, 0)
                HRImages = torch.cat(HRImages, 0)
                LRImages = utils.to_var(LRImages)
                HRImages = utils.to_var(HRImages)

                batchSize = LRImages.size(0)

                ########### Valid STEP ############
                loss, SRImagesList = inferenceStep(epoch, modelList, LRImages, HRImages)
                ###################################
                
                

                Avgloss[0] = Avgloss[0] + torch.Tensor.item(loss.data)
                    
                finali = i + 1

                if (i + 1) % 1 == 0:
                    olda = a
                    a = time.perf_counter()
                    print('                       Test : [%d/%d][%.2f%%]'
                            % (epoch, p.MaxEpoch, (i + 1) / (len(validDataLoader.dataset) / 1 / 100)),  end="\r")

                #print('saving output images...')
                # Save sampled images
                copyCoff = 1

                ###################################
                SRImages = SRImagesList[-1]
                if len(LRImages.size()) == 5:
                    PSNR += utils.calculateImagePSNR(SRImages,HRImages[:,p.sequenceLength//2,:,:,:])
                else:
                    PSNR += utils.calculateImagePSNR(SRImages,HRImages)
                GlobalPSNRCount += 1

                lr_images = utils.denorm(LRImages.cpu().view(LRImages.size(0), 1 if p.colorMode=='grayscale' else 3, LRImages.size(2), LRImages.size(3)))

                hr_images = utils.denorm(HRImages.cpu().view(HRImages.size(0), 1 if p.colorMode=='grayscale' else 3, HRImages.size(2), HRImages.size(3)))


                for ii, si in enumerate(SRImagesList):
                    if (si.size(2) != HRImages.size(2) or si.size(3) != HRImages.size(3)):
                        SRImagesList[ii] = F.interpolate(si, size=(HRImages.size(2),HRImages.size(3)), mode='bicubic')

                SRImages = torch.cat(SRImagesList, 3)

                sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if p.colorMode=='grayscale' else 3, SRImages.size(2), SRImages.size(3))) 
            
                if p.sameOutputSize == False:
                    cated_images = torch.cat((nn.functional.interpolate(lr_images.data, size=(HRImages.size(2),HRImages.size(3) * copyCoff), mode='bicubic'),
                                        sr_images.data,
                                        hr_images.data
                                        ),3)    
                else:
                    cated_images = torch.cat(( lr_images.data,
                                        sr_images.data,
                                        hr_images.data
                                        ),3)


                if args.nosave :        
                    savePath = './data/'+version+'/result/'+subversion+'/SRed_test_images-' + str(i) + '.png'
                else :
                    savePath = './data/'+version+'/result/'+subversion+'/SRed_test_images-' + str(epoch + 1) + '-' + str(i) + '.png'
                    utils.logImages(writer, ['test_images', cated_images], epoch)

                save_image(cated_images, savePath)

                
            

        
        Avgloss[:] = [x / finali for x in Avgloss]
        

        oldb = b
        b = time.perf_counter()      
        PSNR /= len(validDataLoader)

        if PSNR > bestPSNR:
            bestPSNR = PSNR

        print('Test : [%d/%d] PSNR: %.2f dB / BEST: %.2f dB                        '
                % (epoch, p.MaxEpoch, PSNR, bestPSNR))


        # Save loss log
        utils.logValues(writer, ['test_loss', Avgloss[0].item()], epoch)
        utils.logValues(writer, ['test_PSNR', PSNR], epoch)
        if (p.testDataset == 'Vid4'):
            utils.logValues(w4b, ['test_PSNR', 23.78], epoch)
            utils.logValues(w4EDVR, ['test_PSNR', 27.35], epoch)
        elif (p.testDataset == 'Set5'):
            utils.logValues(w2, ['test_PSNR', 33.59], epoch)
            utils.logValues(w3, ['test_PSNR', 30.42], epoch)
            utils.logValues(w4, ['test_PSNR', 28.47], epoch)
            utils.logValues(w8, ['test_PSNR', 24.57], epoch)

    for scheduler in modelList.getSchedulers():
        scheduler.step()
    
        





