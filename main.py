'''
main.py
'''
mainversion = "2.07.201014"



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
from backbone.config import Config
import data_loader as dl
import backbone.vision as vision
import model
import backbone.utils as utils
import backbone.module as module
import backbone.structure as structure

from edit import editversion, version, subversion, trainStep, validationStep, ModelList, inferenceStep












# GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.param.general.GPUNum)

#Arg parser init
parser, args = utils.initArgParser()

#Tensorboard
writer = utils.initTensorboard(version, subversion)

#init Folder & Files
utils.initFolderAndFiles(version, subversion)

#for multiprocessing gpu augmentation in dataloader
#if __name__ == '__main__':torch.multiprocessing.set_start_method('spawn')



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
exList = ['main.py', 'edit.py', 'data_loader_old.py', 'main_old.py', 'test.py']
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
    inferenceDataset = dl.DataLoader('inference')
    print("")

else:

    print(f"load Train Dataset...")
    list(map( lambda k : print(f"    - {k}: {Config.paramDict['data']['dataLoader']['train'][k]}"), Config.paramDict['data']['dataLoader']['train']))
    trainDataLoader = dl.DataLoader('train')
    print("")

    print(f"load Valid Dataset...")
    list(map( lambda k : print(f"    - {k}: {Config.paramDict['data']['dataLoader']['validation'][k]}"), Config.paramDict['data']['dataLoader']['validation']))
    validDataLoader = dl.DataLoader('validation')
    print("")
    
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
        


        #TODO: 고치기
        '''
        ########### Inference using CUSTOM dataset ############
        if p.inferenceDataset == 'CUSTOM':
            with torch.no_grad():
                LRImages = []
                for _LRi in Imagepairs:
                    LRImages.append(_LRi0)
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
                    sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.inference.datasetComponent[0]]['colorMode'] =='grayscale' else 3, SRImages.size(2), SRImages.size(3))) 
                
                savePath = './data/'+version+'/result/'+subversion+'/SRed_inference_images-' + str(i) + '.png'            
                save_image(sr_images, savePath)

        ########### Inference using Benchmark dataset ############
        else:
        '''
        epoch = 0
        PSNR = 0
        with torch.no_grad():
            
            LRImages = Imagepairs['LR']

            HRImages = Imagepairs['HR']

            #HRImages = F.interpolate(HRImages, size=tuple(4*x for x in LRImages.size()[-2:]), mode='bicubic')
            epoch += 1
            batchSize = LRImages.size(0)

            ########### Inference STEP with paired GT ############
            loss, SRImagesList = validationStep(epoch, modelList, LRImages, HRImages)
            ###################################

            SRImages = SRImagesList[-1]
            '''
            if len(LRImages.size()) == 5:
                PSNR += utils.calculateImagePSNR(SRImages,HRImages[:,Config.param.data.dataLoader.inference.sequenceLength//2,:,:,:], Config.param.data.dataLoader.inference.valueRangeType, Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.inference.datasetComponent[0]]['colorMode'])
            else:
                PSNR += utils.calculateImagePSNR(SRImages,HRImages, Config.param.data.dataLoader.inference.valueRangeType, Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.inference.datasetComponent[0]]['colorMode'])
            '''
            SRImages = torch.cat(SRImagesList, 3)
            
            sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.inference.datasetComponent[0]]['colorMode'] =='grayscale' else 3, SRImages.size(2), SRImages.size(3)), Config.param.data.dataLoader.inference.valueRangeType) 
            
            #print('/SRed_inference_images-' + str(i) + '.png' + ", PSNR : ", PSNR)
            savePath = './data/'+version+'/result/'+subversion+'/SRed_inference_images-' + str(i) + '.png'            
            save_image(sr_images, savePath)
    
        eachEnd = time.perf_counter()
        print('/SRed_inference_images-' + str(i) + '.png' + ', model inference completed : ', eachEnd - eachStart)
    
    end = time.perf_counter()
    print('All model inference completed : ', end - start)
    #print(f'P S N R : {PSNR / epoch:.3f} dB')


else : 
    for epoch in range(startEpoch, Config.param.train.step.maxEpoch):

        
        # ============= Train =============#
        # ============= Train =============#
        # ============= Train =============#
        # 1배치당 시간
        a = time.perf_counter()
        # 1에폭당 시간
        b = time.perf_counter()

        finali = 0
        PSNR = 0

        PSNR_A = 0
        PSNR_B = 0

        GlobalPSNRCount = 0

        Avgloss = [torch.zeros(1)]*256

        
        for i, Imagepairs in enumerate(trainDataLoader):

            if i == Config.param.train.step.earlyStopStep : break

            LRImages = Imagepairs['LR']
            HRImages = Imagepairs['HR']
            batchSize = LRImages.size(0)
                    
            ########### train STEP ############
            lossList, SRImagesList = trainStep(epoch, modelList, LRImages, HRImages)
            ###################################
            SRImages = SRImagesList[-1]
            if len(LRImages.size()) == 5:
                PSNR += utils.calculateImagePSNR(SRImages,HRImages[:,Config.param.data.dataLoader.train.sequenceLength//2,:,:,:], Config.param.data.dataLoader.train.valueRangeType, Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.train.datasetComponent[0]]['colorMode'])
            else:
                PSNR += utils.calculateImagePSNR(SRImages,HRImages, Config.param.data.dataLoader.train.valueRangeType, Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.train.datasetComponent[0]]['colorMode'])

            GlobalPSNRCount += 1

            for lossIdx, loss in enumerate(lossList):
                Avgloss[lossIdx] = Avgloss[lossIdx] + torch.Tensor.item(loss.data)
                
            finali = i + 1

            if (i + 1) % 1 == 0:
                olda = a
                a = time.perf_counter()
                print('E[%d/%d][%.2f%%] NET:'
                        % (epoch, Config.param.train.step.maxEpoch, (i + 1) / (len(trainDataLoader.dataset) / Config.param.data.dataLoader.train.batchSize / 100)),  end=" ")

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
                        optimizer = getattr(modelList, mdlStr+"_optimizer")
                        lrList = [param_group['lr'] for param_group in optimizer.param_groups]
                        assert [lrList[0]] * len(lrList) == lrList, 'main.py :: Error, optimizer has different values of learning rates. Tell me... I\'ll fix it.'
                        lr = lrList[0]
                        print(f"{mdlStr}: {lr:.6f} ",  end="")
                print(f"] time: {(a - olda):.2f} sec    ", end="\r")


        Avgloss[:] = [x / finali for x in Avgloss]
        
        oldb = b
        b = time.perf_counter()      

        print('E[%d/%d] NET:'
                % (epoch, Config.param.train.step.maxEpoch),  end=" ")

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
                optimizer = getattr(modelList, mdlStr+"_optimizer")
                lrList = [param_group['lr'] for param_group in optimizer.param_groups]
                assert [lrList[0]] * len(lrList) == lrList, 'main.py :: Error, optimizer has different values of learning rates. Tell me... I\'ll fix it.'
                lr = lrList[0]
                print(f"{mdlStr}: {lr:.6f} ",  end="")
        print(f"] time: {(b - oldb):.2f} sec                    ")



        print('saving model, ', end="")
        utils.saveModels(modelList, version, subversion, epoch, Avgloss[0], bestPSNR)
        lastLoss = Avgloss[0]


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
            LRImages = LRImages[:,Config.param.data.dataLoader.train.sequenceLength//2,:,:,:]
            HRImages = HRImages[:,Config.param.data.dataLoader.train.sequenceLength//2,:,:,:]
            copyCoff = 1
        lr_images = utils.denorm(LRImages.cpu().view(LRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.train.datasetComponent[0]]['colorMode'] =='grayscale' else 3, LRImages.size(2), LRImages.size(3)), Config.param.data.dataLoader.train.valueRangeType)
        hr_images = utils.denorm(HRImages.cpu().view(HRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.train.datasetComponent[0]]['colorMode'] =='grayscale' else 3, HRImages.size(2), HRImages.size(3)), Config.param.data.dataLoader.train.valueRangeType)

        for i, si in enumerate(SRImagesList):
            if (si.size(2) != HRImages.size(2) or si.size(3) != HRImages.size(3)):
                SRImagesList[i] = F.interpolate(si, size=(HRImages.size(2),HRImages.size(3)), mode='bicubic')

        SRImages = torch.cat(SRImagesList, 3)

        sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.train.datasetComponent[0]]['colorMode'] == 'grayscale' else 3, SRImages.size(2), SRImages.size(3)), Config.param.data.dataLoader.train.valueRangeType)

        cated_images = torch.cat((F.interpolate(lr_images.data, size=(HRImages.size(2), HRImages.size(3) * copyCoff), mode='bicubic'),
                            sr_images.data,
                            hr_images.data
                            ),3)    
        '''
        else:
            cated_images = torch.cat(( lr_images.data,
                                sr_images.data,
                                hr_images.data
                                ),3)
        '''

        if args.nosave :        
            savePath = './data/'+version+'/result/'+subversion+'/SRed_train_images.png'
        else :
            savePath = './data/'+version+'/result/'+subversion+'/SRed_train_images-' + str(epoch + 1) + '.png'
            #utils.logImages(writer, ['train_images', cated_images], epoch)

        save_image(cated_images[:Config.param.save.maxSaveImageNumberTrain], savePath, padding=12)
        #save_image(utils.addCaptionToImageTensor(sr_images.data, 'Really Bad Boy 7월 7일 1234567890'), 'a.png')

        

        

        if (epoch + 1) % Config.param.train.step.validationStep == 0:
            # ============= Valid =============#
            # ============= Valid =============#
            # ============= Valid =============#
            # 1배치당 시간
            a = time.perf_counter()
            # 1에폭당 시간
            b = time.perf_counter()

            finali = 0

            PSNR = [0]*256
            t_PSNR = [0]*256

            GlobalPSNRCount = 0

            Avgloss = [torch.zeros(1)]*256

            
            for i, Imagepairs in enumerate(validDataLoader):

                with torch.no_grad():

                    '''
                    LRImages = []
                    HRImages = []
                    for _LRi, _HRi in Imagepairs:
                        LRImages.append(_LRi)
                        HRImages.append(_HRi)
                    LRImages = torch.cat(LRImages, 0)
                    HRImages = torch.cat(HRImages, 0)
                    LRImages = utils.to_var(LRImages)
                    HRImages = utils.to_var(HRImages)
                    '''
                    LRImages = Imagepairs['LR']
                    HRImages = Imagepairs['HR']

                    # (TMP) cut Alpha channel
                    LRImages = LRImages[:,0:3,:,:]
                    HRImages = HRImages[:,0:3,:,:]


                    #HRImages = F.interpolate(HRImages, size=tuple(4*x for x in LRImages.size()[-2:]), mode='bicubic')


                    batchSize = LRImages.size(0)

                    ########### Valid STEP ############
                    loss, SRImagesList = validationStep(epoch, modelList, LRImages, HRImages)
                    ###################################
                    
                    

                    Avgloss[0] = Avgloss[0] + torch.Tensor.item(loss.data)
                        
                    finali = i + 1

                    if (i + 1) % 1 == 0:
                        olda = a
                        a = time.perf_counter()
                        print('Test : [%d/%d][%.2f%%]'
                                % (epoch, Config.param.train.step.maxEpoch, (i + 1) / (len(validDataLoader.dataset) / 1 / 100)),  end="\r")

                    #print('saving output images...')
                    # Save sampled images
                    copyCoff = 1

                    ###################################
                    SRImages = SRImagesList[2]
                    if len(LRImages.size()) == 5:
                        t_PSNR = utils.calculateImagePSNR(SRImages,HRImages[:,Config.param.data.dataLoader.validation.sequenceLength//2,:,:,:], Config.param.data.dataLoader.validation.valueRangeType, Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.validation.datasetComponent[0]]['colorMode'])
                    else:
                        COLORMODE = Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.validation.datasetComponent[0]]['colorMode']
                        VALUERANGETYPE = Config.param.data.dataLoader.validation.valueRangeType
                        for si, SRImages in enumerate(SRImagesList):
                            t_PSNR[si] = utils.calculateImagePSNR(SRImages, HRImages, VALUERANGETYPE, COLORMODE)
                        

                    for pi in range(len(SRImagesList)):
                        PSNR[pi] += t_PSNR[pi]

                    GlobalPSNRCount += 1

                    print(f"PSNR: {GlobalPSNRCount-1} ({SRImages.size(2)}x{SRImages.size(3)}) : ", end="")
                    for pi in range(len(SRImagesList)):
                        print(f"{t_PSNR[pi]:.2f}dB ", end="")
                    print("")


                    lr_images = utils.denorm(LRImages.cpu().view(LRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.validation.datasetComponent[0]]['colorMode'] =='grayscale' else 3, LRImages.size(2), LRImages.size(3)), Config.param.data.dataLoader.validation.valueRangeType)

                    hr_images = utils.denorm(HRImages.cpu().view(HRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.validation.datasetComponent[0]]['colorMode'] =='grayscale' else 3, HRImages.size(2), HRImages.size(3)), Config.param.data.dataLoader.validation.valueRangeType)


                    for ii, si in enumerate(SRImagesList):
                        if (si.size(2) != HRImages.size(2) or si.size(3) != HRImages.size(3)):
                            SRImagesList[ii] = F.interpolate(si, size=(HRImages.size(2),HRImages.size(3)), mode='bicubic')

                    SRImages = torch.cat(SRImagesList, 3)

                    sr_images = utils.denorm(SRImages.cpu().view(SRImages.size(0), 1 if Config.paramDict['data']['datasetComponent'][Config.param.data.dataLoader.validation.datasetComponent[0]]['colorMode'] =='grayscale' else 3, SRImages.size(2), SRImages.size(3)), Config.param.data.dataLoader.validation.valueRangeType) 
                

                    cated_images = torch.cat((F.interpolate(lr_images.data, size=(HRImages.size(2),HRImages.size(3) * copyCoff), mode='bicubic'),
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

            for pi in range(len(SRImagesList)):
                PSNR[pi] /= len(validDataLoader)
            if PSNR[len(SRImagesList)-1] > bestPSNR:
                bestPSNR = PSNR[len(SRImagesList)-1]

            print(f'Test : [{epoch}/{Config.param.train.step.maxEpoch}] PSNR: ', end="")
            for pi in range(len(SRImagesList)):
                print(f'{PSNR[pi]:.2f} dB ', end="")
            print(f'/ Best : {bestPSNR:.2f} dB')


            # Save loss log
            utils.logValues(writer, ['test_loss', Avgloss[0].item()], epoch)
            utils.logValues(writer, ['test_PSNR', PSNR[len(SRImagesList)-1]], epoch)

        for scheduler in modelList.getSchedulers():
            scheduler.step()
        
            





