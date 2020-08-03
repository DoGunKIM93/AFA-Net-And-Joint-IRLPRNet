'''
main.py
'''
mainversion = "1.61.200710"


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
import backbone.vision as vision
import model
import param as p
import backbone.utils as utils
import backbone.module as module
import backbone.structure as structure

import all_new_data_loader as andl

from data_loader import SRDataLoader
from edit import editversion, version, subversion, trainStep, validationStep, ModelList, inferenceStep
from backbone.config import Config













# GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"]=p.GPUNum

#Arg parser init
parser, args = utils.initArgParser()

#Tensorboard
writer = utils.initTensorboard(version, subversion)

#init Folder & Files
utils.initFolderAndFiles(version, subversion)

#read Configs
utils.readConfigs()





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
print("         -------- FRAMEWORK VERSIONs --------")

#print module version
exList = ['main.py', 'edit.py', 'all_new_data_loader.py', 'all_new_main.py', 'test.py']
pyModuleStrList = list(x[:-3] for x in os.listdir('.') if x not in exList and x != 'all_new_data_loader.py' and x.endswith('.py')) + list(f'backbone.{x[:-3]}' for x in os.listdir('./backbone') if x.endswith('.py'))
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
    print(f"load Test Dataset... {p.inferenceDataset} / {p.colorMode} X{p.scaleFactor} ({p.testScaleMethod})")
    inferenceDataset = SRDataLoader(dataset   = p.inferenceDataset,
                              datasetType   = p.inferenceDatasetType,
                              dataPath      = p.dataPath,
                              scaleFactor   = p.scaleFactor,
                              scaleMethod   = p.inferenceScaleMethod,
                              batchSize     = 1,
                              mode          = 'inference',
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
startEpoch, lastLoss, bestPSNR = utils.loadModels(modelList, version, subversion, args.load, args.inferenceTest)
print(f"All model loaded. Last Epoch: {startEpoch}, Loss: {lastLoss.item():.6f}, BEST PSNR: {bestPSNR:.2f} dB")

if args.inferenceTest == True :
    print("model_inference")

    # Duration of inference time
    start = time.perf_counter()

    for i, Imagepairs in enumerate(inferenceDataset):
        eachStart = time.perf_counter()

        ########### Inference using CUSTOM dataset ############
        if p.inferenceDataset == 'CUSTOM':
            with torch.no_grad():
                LRImages = []
                for _LRi in Imagepairs:
                    LRImages.append(_LRi)
                LRImages = torch.cat(LRImages, 0)
                LRImages = utils.to_var(LRImages)

                ########### Inference STEP without paired GT ############
                SRImagesList = inferenceStep(modelList, LRImages)
                ###################################

                SRImages = SRImagesList[-1]
                SRImages = torch.cat(SRImagesList, 3)

                if p.blendingMode == 'possionBlending':
                    sr_images = SRImages
                else:
                    sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if p.colorMode=='grayscale' else 3, SRImages.size(2), SRImages.size(3))) 
                
                savePath = './data/'+version+'/result/'+subversion+'/SRed_inference_images-' + str(i) + '.png'            
                save_image(sr_images, savePath)
        ########### Inference using Benchmark dataset ############
        else:
            epoch = 1
            PSNR = 0
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

                ########### Inference STEP with paired GT ############
                loss, SRImagesList = validationStep(epoch, modelList, LRImages, HRImages)
                ###################################

                SRImages = SRImagesList[-1]

                if len(LRImages.size()) == 5:
                    PSNR += utils.calculateImagePSNR(SRImages,HRImages[:,p.sequenceLength//2,:,:,:])
                else:
                    PSNR += utils.calculateImagePSNR(SRImages,HRImages)

                SRImages = torch.cat(SRImagesList, 3)
                
                if p.blendingMode == 'possionBlending':
                    sr_images = SRImages
                else:
                    sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if p.colorMode=='grayscale' else 3, SRImages.size(2), SRImages.size(3))) 
                
                print('/SRed_inference_images-' + str(i) + '.png' + ", PSNR : ", PSNR)
                savePath = './data/'+version+'/result/'+subversion+'/SRed_inference_images-' + str(i) + '.png'            
                save_image(sr_images, savePath)
        
        eachEnd = time.perf_counter()
        print('/SRed_inference_images-' + str(i) + '.png' + ', model inference completed : ', eachEnd - eachStart)
    
    end = time.perf_counter()
    print('All model inference completed : ', end - start)


else : 
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
            cated_images = torch.cat((F.interpolate(lr_images.data, size=(HRImages.size(2),HRImages.size(3) * copyCoff), mode='bicubic'),
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
                    loss, SRImagesList = validationStep(epoch, modelList, LRImages, HRImages)
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
                        cated_images = torch.cat((F.interpolate(lr_images.data, size=(HRImages.size(2),HRImages.size(3) * copyCoff), mode='bicubic'),
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
            
            # if (p.testDataset == 'Vid4'):
            #     utils.logValues(w4b, ['test_PSNR', 23.78], epoch)
            #     utils.logValues(w4EDVR, ['test_PSNR', 27.35], epoch)
            # elif (p.testDataset == 'Set5'):
            #     utils.logValues(w2, ['test_PSNR', 33.59], epoch)
            #     utils.logValues(w3, ['test_PSNR', 30.42], epoch)
            #     utils.logValues(w4, ['test_PSNR', 28.47], epoch)
            #     utils.logValues(w8, ['test_PSNR', 24.57], epoch)

        for scheduler in modelList.getSchedulers():
            scheduler.step()
        
            





