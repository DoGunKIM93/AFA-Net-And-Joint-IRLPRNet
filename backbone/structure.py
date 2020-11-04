'''
structure.py
'''
version = "1.10.201015"

import torch.nn as nn
import torch
import apex.amp as amp

from torch.autograd import Variable


import argparse
import time


import backbone.utils as utils

from backbone.config import Config




class Epoch():

    def __init__(self, 
                 dataLoader,
                 modelList,
                 step,
                 researchVersion,
                 researchSubVersion,
                 writer,

                 scoreMetricDict = {}, # { (scoreName) -> STRING : { function:(func) -> FUNCTION , argDataNames:  ['LR','HR', ...] -> list of STRING, additionalArgs:[...], bestScore:(Score)->number } }
                 
                 resultSaveData = None,
                 resultSaveFileName = '',
                 isNoResultArchiving = False,
                 earlyStopIteration = -1):

        self.dataLoader = dataLoader
        self.modelList = modelList

        self.step = step

        self.researchVersion = researchVersion
        self.researchSubVersion = researchSubVersion

        self.writer = writer


        self.scoreMetricDict = scoreMetricDict

        self.resultSaveData = resultSaveData
        self.resultSaveFileName = resultSaveFileName
        self.isNoResultArchiving = isNoResultArchiving
        self.earlyStopIteration = earlyStopIteration

    def run(self, currentEpoch, metaData = None,
            do_resultSave = True, 
            do_modelSave = True,
            do_calculateScore = True):


        assert do_resultSave in [True, False]
        assert do_modelSave in [True, False]
        assert do_calculateScore in [True, 'DETAIL', False]



        ####################################
        #            Init MetaData           
        ####################################
        if metaData is None : 
            metaData = {}
        if 'lastLoss' not in metaData.keys() : 
            metaData.update({'lastLoss': {}})
        if 'bestScore' not in metaData.keys() : 
            metaData.update({'bestScore': {}})





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
        bestScoreDict = metaData['bestScore']
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
            #         Instruction Step             
            ####################################
            lossDict, resultDict = self.step(currentEpoch, self.modelList, dataDict)

            for key in dataDict:
                assert key not in resultDict.keys(), f"Some keys are duplicated in input data and result data of Step... : {key}"
                resultDict.update({key:dataDict[key]})



            ####################################
            #             SCORE CALC             
            ####################################
            if do_calculateScore is not False:

                if do_calculateScore is 'DETAIL':
                    print(f'CASE {i} :: ', end='')

                for scoreMetricName in self.scoreMetricDict:

                    scoreFunc = self.scoreMetricDict[scoreMetricName]['function']
                    scoreFuncArgs = list( resultDict[name] for name in self.scoreMetricDict[scoreMetricName]['argDataNames'] )
                    scoreFuncAdditionalArgs = self.scoreMetricDict[scoreMetricName]['additionalArgs']

                    for ss, sfaarg in enumerate(scoreFuncAdditionalArgs):
                        if sfaarg == '$VALUE_RANGE_TYPE':
                            scoreFuncAdditionalArgs[ss] = VALUE_RANGE_TYPE
                        elif sfaarg == '$COLOR_MODE':
                            scoreFuncAdditionalArgs[ss] = COLOR_MODE
                        elif sfaarg == '$SEQUENCE_LENGTH':
                            scoreFuncAdditionalArgs[ss] = SEQUENCE_LENGTH

                    score = scoreFunc(*scoreFuncArgs, *scoreFuncAdditionalArgs)

                    if do_calculateScore is 'DETAIL':
                        print(f"{scoreMetricName}: {score} ", end='')

                    if scoreMetricName in AvgScoreDict:
                        AvgScoreDict[scoreMetricName] = AvgScoreDict[scoreMetricName] + score
                    else:
                        AvgScoreDict.update({scoreMetricName: score})
                
                if do_calculateScore is 'DETAIL':
                    print('')

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
                    AvgLossDict.update({key: torch.Tensor.item(lossDict[key].data)})
                
            finali = i + 1




            ####################################
            #        Printing & Logging              
            ####################################
            if (i + 1) % 1 == 0:

                #calc Time per Batch
                oldTimePerBatch = timePerBatch
                timePerBatch = time.perf_counter()

                #print Current status
                print('E[%d/%d][%.2f%%] NET:'
                        % (currentEpoch, MAX_EPOCH, (i + 1) / (DATASET_LENGTH / PARAM_BATCH_SIZE / 100)),  end=" ")

                #print Loss
                print('loss: [', end="")
                for key in lossDict.keys():
                    print(f'{key}: {AvgLossDict[key]/finali:.5f}, ', end="")
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
            if key in bestScoreDict.keys():
                bestScoreDict[key] = bestScoreDict[key] if abs(bestScoreDict[key]) >= abs(AvgScoreDict[key]) else AvgScoreDict[key]
            else:
                bestScoreDict.update({key: AvgScoreDict[key]})
        #Update MetaData
        metaData['lastLoss'].update(AvgLossDict)
        metaData['bestScore'].update(bestScoreDict)

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
                print(f'{key}: {AvgLossDict[key]:.5f}, ', end="")
            print(']', end=" ")

        if len(AvgScoreDict.keys()) > 0:
            print('score: [', end="")
            for key in AvgScoreDict:
                print(f'{key}: {AvgScoreDict[key]:.2f}, ', end="")
            print(']', end=" ")
        
        #print LR
        print('lr: [ ',  end="")

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

        #print Epoch Time
        print(f"] time: {(timePerEpoch - oldTimePerEpoch):.2f} sec                    ")




        print('saving ', end="")

        ####################################    
        #          Model Saving              
        ####################################
        if do_modelSave is True:
            print('models, ', end="")
            utils.saveModels(self.modelList, self.researchVersion, self.researchSubVersion, currentEpoch, metaData)




        ####################################    
        #            LOGGING            
        ####################################
        print('log, ', end="")
        
        # Save loss log
        for key in AvgLossDict:
            utils.logValues(self.writer, [f'{key}', AvgLossDict[key]], currentEpoch)

        # Save score log
        for key in AvgScoreDict:
            utils.logValues(self.writer, [f'{key}', AvgScoreDict[key]], currentEpoch)


        
        
        ####################################        
        #            save Rst.            
        ####################################W
        if do_resultSave is True:

            '''
            for key in resultDict:
                assert key not in dataDict.keys(), f"Some keys are duplicated in input data and result data of Step... : {key}"
                dataDict.update({key:resultDict[key]})
            '''
                
            print('output images, ', end="")
            # Save sampled images
            if self.isNoResultArchiving :        
                savePath = './data/' + self.researchVersion + '/result/' + self.researchSubVersion + '/' + self.resultSaveFileName + '.png'
            else :
                savePath = './data/' + self.researchVersion + '/result/' + self.researchSubVersion + '/' + self.resultSaveFileName + '-' + str(currentEpoch + 1) + '.png'
                
            utils.saveImageTensorToFile(resultDict, savePath, saveDataDictKeys=self.resultSaveData, colorMode=COLOR_MODE, valueRangeType=VALUE_RANGE_TYPE, interpolation='nearest')

            #Image Logging is not Supported now
            #utils.logImages(self.writer, ['train_images', cated_images], currentEpoch)
        

        print('Finished.')



        ####################################        
        #         Step LR Scheduler            
        ####################################W
        for scheduler in self.modelList.getSchedulers():
            scheduler.step()
    



        return metaData




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
            try:
                mdlPpaLst.append(getattr(self, mdlStr))
            except AttributeError:
                mdlPpaLst.append(None)
        return mdlPpaLst

    def getPretrainedPath(self, mdlStr):
        try:
            pP = Config.param.data.path.pretrainedPath + getattr(self, mdlStr + "_pretrained")
        except AttributeError:
            pP = None
        return pP

