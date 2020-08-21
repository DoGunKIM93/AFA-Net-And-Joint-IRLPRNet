'''
structure.py
'''
version = "1.01.200820"

import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse

import apex.amp as amp
#from apex.parallel import DistributedDataParallel as DDP

from backbone.config import Config




class EpochBase():

    def __init__(self, 
                 dataLoader,
                 modelList,
                 researchVersion,
                 researchSubVersion,

                 scoreMetricDict = {}, # { (scoreName) -> STRING : { function:(func) -> FUNCTION , argDataNames:  ['LR','HR', ...] -> list of STRING, additionalArgs:[...]  } }
                 
                 earlyStopIteration = -1):

        self.dataLoader = dataLoader
        self.modelList = modelList

        self.researchVersion = researchVersion
        self.researchSubVersion = researchSubVersion

        self.scoreMetricDict = scoreMetricDict

        self.isPSNR = isPSNR
        self.PSNRDataNamePair = PSNRDataNamePair

        self.earlyStopIteration = earlyStopIteration

    def Epoch(self, currentEpoch):


        finali = 0
        globalScoreCount = 0


        ####################################
        #            Hyp. Params           
        ####################################
        DATALOADER_NAME = self.dataLoader.name
        DATASET_COMPONENT_NAME_LIST = Config.paramDict['data']['dataLoader'][DATALOADER_NAME]['datasetComponent']

        DATASET_LENGTH = len(self.dataLoader.dataset)
        PARAM_BATCH_SIZE = Config.paramDict['data']['dataLoader'][DATALOADER_NAME]['batchSize']
        MAX_EPOCH = Config.param.train.step.maxEpoch

        VALUE_RANGE_TYPE = Config.paramDict['data']['dataLoader'][DATALOADER_NAME]['valueRangeType']
        COLOR_MODE = Config.paramDict['data']['datasetComponent'][DATASET_COMPONENT_NAME_LIST[0]]['colorMode']
        SEQUENCE_LENGTH = Config.paramDict['data']['dataLoader'][DATALOADER_NAME]['sequenceLength'] if 'sequenceLength' in Config.paramDict['data']['dataLoader'][DATALOADER_NAME] else 1
        
        


        ####################################
        #            Rst. Vars           
        ####################################
        AvgScoreDict = {}
        bestScoreDict = {}
        AvgLossDict = {}
        timePerBatch = time.perf_counter() # 1배치당 시간
        timePerEpoch = time.perf_counter() # 1에폭당 시간





        ####################################
        #       in-Batch Instructions          
        ####################################
        for i, dataDict in enumerate(self.dataLoader):

            if i == self.earlyStopIteration: break

            batchSize = dataDict[list(dataDict.keys())[0]].size(0)
                    


            ####################################
            #             Train Step             
            ####################################
            lossDict, resultDict = trainStep(currentEpoch, modelList, dataDict)

            for key in resultDict:
                assert key not in dataDict.keys(), f"Some keys are duplicated in input data and result data of Step... : {key}"
                dataDict[key] = resultDict[key]



            ####################################
            #             SCORE CALC             
            ####################################
            for scoreMetricName in self.scoreMetricDict:
                scoreFunc = self.scoreMetricDict[scoreMetricName]['function']
                scoreFuncArgs = list( dataDict[name] for name in self.scoreMetricDict[scoreMetricName]['argDataNames'] )
                scoreFuncAdditionalArgs = self.scoreMetricDict[scoreMetricName]['additionalArgs']

                score = scoreFunc(*scoreFuncArgs, *scoreFuncAdditionalArgs)

                if scoreMetricName in AvgScoreDict:
                    AvgScoreDict[scoreMetricName] = AvgScoreDict[scoreMetricName] + score
                else:
                    AvgScoreDict[scoreMetricName] = score

            globalScoreCount += 1
 

            '''
            if self.isPSNR is True:
                inp = resultDict[self.PSNRDataNamePair[0]]
                tar = resultDict[self.PSNRDataNamePair[1]]

                if len(inp.size()) == 5:
                    tar = tar[:,Config.param.data.dataLoader.train.sequenceLength//2,:,:,:]
                    
                PSNR += utils.calculateImagePSNR(inp, tar, VALUE_RANGE_TYPE, COLOR_MODE)
            '''


            ####################################
            #             LOSS              
            ####################################
            for key in lossDict:
                if key in AvgLossDict:
                    AvgLossDict[key] = AvgLossDict[key] + torch.Tensor.item(lossDict[key].data)
                else:
                    AvgLossDict[key] = torch.Tensor.item(lossDict[key].data)
                
            finali = i + 1




            ####################################
            #        Printing & Logging              
            ####################################
            if (i + 1) % 1 == 0:

                #calc Time per Batch
                oldTimePerBatch = timePerBatch
                timePerBatch = time.perf_counter()

                #print Current status
                print('                      E[%d/%d][%.2f%%] NET:'
                        % (currentEpoch, MAX_EPOCH, (i + 1) / (DATASET_LENGTH / PARAM_BATCH_SIZE / 100)),  end=" ")

                #print Loss
                print('loss: [', end="")
                for key in enumerate(lossDict):
                    print(f'{torch.Tensor.item(AvgLossDict[key].data)/finali:.5f}, ', end="")
                print('] ', end="")


                #print Learning Rate
                print('lr: [',  end="")

                for mdlStr in self.modelList.getList():
                    if len([attr for attr in vars(self.modelList) if attr == (mdlStr+"_scheduler")]) > 0:
                        schd = getattr(self.modelList, mdlStr+"_scheduler")
                        print(f"{mdlStr}: {schd.get_lr()[0]:.6f} ",  end="")
                    elif len([attr for attr in vars(self.modelList) if attr == (mdlStr+"_optimizer")]) > 0:
                        optimizer = getattr(self.modelList, mdlStr+"_optimizer")
                        lrList = [param_group['lr'] for param_group in optimizer.param_groups]
                        assert [lrList[0]] * len(lrList) == lrList, 'main.py :: Error, optimizer has different values of learning rates. Tell me... I\'ll fix it.'
                        lr = lrList[0]
                        print(f"{mdlStr}: {lr:.6f} ",  end="")
                print(f"] time: {(timePerBatch - oldTimePerBatch):.2f} sec    ", end="\r")



        ####################################
        #      ENDing Calcs of Epochs              
        ####################################

        #Calc Avg. Losses & Scores
        for key in AvgLossDict:
            AvgLossDict[key] = AvgLossDict[key] / finali 
        for key in AvgScoreDict:
            AvgScoreDict[key] = AvgScoreDict[key] / finali 
            bestScoreDict[key] = bestScoreDict[key] if abs(bestScoreDict[key]) >= abs(AvgScoreDict[key]) else AvgScoreDict[key]
        
        #Calc Time per Epoch
        oldTimePerEpoch = timePerEpoch
        timePerEpoch = time.perf_counter()      




        ####################################
        #            Printing              
        ####################################

        #print Epoch Status
        print(f'E[{currentEpoch}/{Config.param.train.step.maxEpoch}] NET:',  end=" ")

        #print Epoch Loss & Score
        if len(AvgLossDict.keys()) > 0:
            print('loss: [', end="")
            for key in AvgLossDict:
                print(f'{torch.Tensor.item(AvgLossDict[key].data):.5f}, ', end="")
            print(']', end=" ")

        if len(AvgScoreDict.keys()) > 0:
            print('score: [', end="")
            for key in AvgScoreDict:
                print(f'{key}: {AvgScoreDict[key]:.2f}, ', end="")
            print(']', end=" ")
        
        #print LR
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

        #print Epoch Time
        print(f"] time: {(timePerEpoch - oldTimePerEpoch):.2f} sec                    ")




        ####################################    
        #          Model Saving              
        ####################################
        print('saving model, ', end="")
        utils.saveModels(modelList, self.researchVersion, self.researchSubVersion, currentEpoch, AvgLossDict, bestScoreDict)




        ####################################    
        #            LOGGING            
        ####################################
        print('log, ', end="")
        
        # Save loss log
        for key in AvgLossDict:
            utils.logValues(writer, [f'{key}', AvgLossDict[key].item()], currentEpoch)

        # Save score log
        for key in AvgScoreDict:
            utils.logValues(writer, [f'{key}', AvgScoreDict[key]], currentEpoch)



        
        ####################################        
        #            save Rst.            
        ####################################W
        #TODO:
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

        cated_images = torch.cat((F.interpolate(lr_images.data, size=(HRImages.size(2),HRImages.size(3) * copyCoff), mode='bicubic'),
                            sr_images.data,
                            hr_images.data
                            ),3)    


        if args.nosave :        
            savePath = './data/'+self.researchVersion+'/result/'+self.researchSubVersion+'/SRed_train_images.png'
        else :
            savePath = './data/'+self.researchVersion+'/result/'+self.researchSubVersion+'/SRed_train_images-' + str(epoch + 1) + '.png'
            utils.logImages(writer, ['train_images', cated_images], epoch)

        save_image(cated_images, savePath)




class DataLoaderVaultBase():
    def __init__(self):
        super(DataLoaderVaultBase, self).__init__()





class ModelListBase():
    def __init__(self):
        super(ModelListBase, self).__init__()

    def initDataparallel(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]

        for mdlStr in mdlStrLst:
            setattr(self, mdlStr, nn.DataParallel(getattr(self, mdlStr)))
    
    def initApexAMP(self):
        if Config.param.train.method.mixedPrecision is True:
            opt_level = 'O0' if Config.param.train.method.mixedPrecision is False else 'O1'
            mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
            for mdlStr in mdlStrLst:
                mdlObj = getattr(self, mdlStr)
                mdlOpt = getattr(self, mdlStr + "_optimizer") if len([attr for attr in vars(self) if attr == (mdlStr+"_optimizer")]) > 0 else None

                if mdlOpt is None:
                    mdlObj = amp.initialize(mdlObj.to('cuda'), opt_level = opt_level)
                    setattr(self, mdlStr, mdlObj)
                else:
                    mdlObj, mdlOpt = amp.initialize(mdlObj.to('cuda'), mdlOpt, opt_level = opt_level)
                    setattr(self, mdlStr, mdlObj)
                    setattr(self, mdlStr + "_optimizer", mdlOpt)
    

    def getList(self):
        return [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]

    def getModels(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlObjLst = []
        for mdlStr in mdlStrLst:
            mdlObjLst.append(getattr(self, mdlStr))
        return mdlObjLst
    
    def getOptimizers(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlOptLst = []
        for mdlStr in mdlStrLst:
            mdlOptLst.append(getattr(self, mdlStr))
        return mdlOptLst

    def getSchedulers(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlSchLst = []
        for mdlStr in mdlStrLst:
            mdlSchLst.append(getattr(self, mdlStr))
        return mdlSchLst

    def getPretrainedPaths(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and attr.endswith("_pretrained")]
        mdlPpaLst = []
        for mdlStr in mdlStrLst:
            mdlPpaLst.append(getattr(self, mdlStr))
        return mdlPpaLst

    def getPretrainedPath(self, mdlStr):
        pP = Config.param.data.path.pretrainedPath + getattr(self, mdlStr + "_pretrained")
        return pP

