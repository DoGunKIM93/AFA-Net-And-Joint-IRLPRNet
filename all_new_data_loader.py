'''
all_new_data_loader.py
'''
version = "1.03.200717"


#FROM Python LIBRARY
import os
import random
import math
import numpy as np
import yaml 
import inspect
import itertools

from PIL import Image
from PIL import PngImagePlugin
from typing import List, Dict, Tuple, Union
from function import reduce


#FROM PyTorch
import torch

from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoaders
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder


#FROM This Project
import param as p
import backbone.preprocessing
import backbone.augmentation
import backbone.utils as utils
from backbone.config import Config



# Prevent memory Error in PIL
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# Read Preprocessings from backbone.preprocessing and Augmentations from backbone.augmentation automatically
PREPROCESSING_DICT = dict(x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith('__')) and (inspect.isclass(x[1])) )
AUGMENTATION_DICT = dict(x for x in inspect.getmembers(backbone.augmentation) if (not x[0].startswith('_')) and (inspect.isfunction(x[1])) )







class DatasetConfig():


    def __init__(self, 
                 name: str, 
                 origin : str = None, 
                 dataType : List[str] = None, 
                 labelType : List[str] = None, 
                 availableMode : List[str] = None, 
                 classes : List[str] = None, 
                 preprocessings : List[str] = None,
                 useDatasetConfig : bool = True):
                 
        self.name = name

        self.origin = origin
        self.dataType = dataType
        self.labelType = labelType
        self.availableMode = availableMode
        self.classes = classes
        self.preprocessings = preprocessings

        self.useDatasetConfig = useDatasetConfig

        if self.useDatasetConfig is True:
            self.getDatasetConfig()



    def getDatasetConfig(self):

        yamlData = Config.datasetConfigDict[f'{self.name}']

        self.origin = str(yamlData['origin'])

        self.dataType = yamlData['dataType']
        self.labelType = yamlData['labelType']


        self.availableMode = list(map(str, yamlData['availableMode']))
        self.classes = list(map(str, yamlData['classes']))
        self.preprocessings = list(map(str, yamlData['preprocessings']))

        





#Representation of single dataset property
class DatasetComponent():


    def __init__(self, datasetConfig, mode, classParameter):

        self.datasetConfig = datasetConfig
        self.name = datasetConfig.name

        self.mode = mode
        self.classParameter = classParameter


        self.dataFileList = None #same length (2-d List)
        self.labelFileList = None
        self.getDataFileList()
        

        self.preprocessingList = None
        self.makePreprocessingList()
    
    def getDataFileList(self):

        #TODO: LABEL
        #TODO: SEQUENCE

        # get origin path
        mainPath = Config.param.data.path.datasetPath
        path = f"{self.datasetConfig['origin']}/{self.mode}/"

        # dynamic construction of class path based on defined classes
        for i in range(len(self.datasetConfig['classes'])):
            classPathList = list(itertools.chain.from_iterable(list(map( lambda y : list(map(lambda x : str(x) if type(x) is int else x + '/' + str(y) if type(y) is int else y , classPathList)), self.classParameter[self.datasetConfig['classes'][i]])))) if i is not 0 else self.classParameter[self.datasetConfig['classes'][i]]
            # Sorry for who read this

        # add origin path in front of all elements of class path list
        pathList = list(map( lambda x : path + x , classPathList))
        
        # construct all of readable file lists in class path lists
        fileList = list(itertools.chain.from_iterable(list(map( lambda x :  list(map( lambda y : x + "/" + y, os.listdir(mainPath + x))) , pathList))))

        # set dataFileList without main path
        self.dataFileList = [ x for x in fileList if (x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp")) ]



    def makePreprocessingList(self):
        self.preprocessingList = list(map((lambda x: PREPROCESSING_DICT[x.split('(')[0]](*list(filter(lambda y : y != '', x.split('(')[1][:-1].replace(' ','').split(','))))), self.config.preprocessings))
        pass




    def __len__(self):
        return len(dataFileList)




##


#TODO: Distribution
#TODO: On-memory Supply 
class Dataset(torchDataset):


    def __init__(self, datasetComponentList:list[DatasetComponent], batchSize: int, samplingCount:int, sameOutputSize:bool, valueRangeType:str, shuffle:bool, augmentation:list[str], numWorkers:int):
        self.datasetComponentList = datasetComponentList
        self.datasetComponentListIntegrityTest()

        self.batchSize = batchSize
        self.samplingCount = samplingCount
        self.sameOutputSize = sameOutputSize
        self.valueRangeType = valueRangeType
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.numWorkers = numWorkers

        self.getDatasetRatio(datasetComponentList)

        self.globalFileList = None
        self.makeGlobalFileList(self.shuffle)



    def makeGlobalFileList(self, shuffle):
        gFL = [x.dataFileList for x in self.datasetComponentList]
        self.globalFileList = gFL



    def popItemInGlobalFileListByIndex(self, index):
        datasetComponentLengthList = list(map(len, datasetComponentList))

        indexComponent = index % len(datasetComponentLengthList)
        indexComponentFileList = ( index // len(datasetComponentLengthList) ) % datasetComponentLengthList[indexComponent]

        return Config.param.data.path.datasetPath + self.globalFileList[indexComponent][indexComponentFileList], self.datasetComponentList[indexComponent].preprocessingList




    def datasetComponentListIntegrityTest(self):
        # Test All dataComponents have same output type.
        # TODO: SAME SHAPE (B , C ..)
        datasetComponentOutputList = self.datasetComponentList.config.output
        assert not datasetComponentOutputList or [datasetComponentOutputList[0]]*len(datasetComponentOutputList) == datasetComponentOutputList, 'data_loader.py :: All datasets in dataloader must have same output type.'
       


    def getDatasetRatio(self, datasetComponentList:list):
        self.datasetLengthList = list(map(len, datasetComponentList))
        self.datasetRatio = list(map((lambda x : x / max(datasetLengthList)), datasetLengthList))



    def preprocessing(self, batchInput, preprocessings:list):
        rst = batchInput
        for preprocessingFunction in preprocessings:
            rst = preprocessingFunction(rst)

        return rst

    
    def calculateMemorySizePerTensor(self, dtype:torch.dtype, expectedShape:list[int]): #WARN: EXCEPT BATCH SIIZEEEEEE!!!!!!!!!!!!!!!!!!!!!!!
        sizeOfOneElementDict = {torch.float32 : 4,
                                torch.float   : 4,
                                torch.float64 : 8,
                                torch.double  : 8,
                                torch.float16 : 2,
                                torch.half    : 2,
                                torch.uint8   : 1,
                                torch.int8    : 1,
                                torch.int16   : 2,
                                torch.short   : 2,
                                torch.int32   : 4,
                                torch.int     : 4,
                                torch.int64   : 8,
                                torch.long    : 8,
                                torch.bool    : 2,
                                }
        elemSize = sizeOfOneElementDict(dtype)

        totalSize = reduce(lambda x, y: x * y, expectedShape) * elemSize

        return totalSize








    

    def torchvisionPreprocessing(self, pilImage, preProc):

        x = pilImage

        for preProcFunc in preProc:
            x = preProcFunc(x)

        return x

    
    def applyAugmentationFunction(self, tnsr, augmentation:str):

        assert augmentation in AUGMENTATION_DICT.keys(), "data_loader.py :: invalid Augmentation Function!! chcek param.yaml."

        augFunc = AUGMENTATION_DICT[augmentation.split['('][0]]
        args = list(filter(lambda y : y != '', augmentation.split['('][1][:-1].replace(' ','').split(',') ))

        tnsr = augFunc(tnsr, *args)

        return tnsr




    def dataAugmentation(self, tnsr, augmentations: list[str]):

        x = tnsr

        for augmentation in augmentations:
            x = applyAugmentationFunction(x, augmentation)

        return x

    def setTensorValueRange(self, tnsr, valueRangeType:str):

        if valueRangeType == '-1~1':
            tnsr = tnsr * 2 - 1

        return tnsr
    

    def loadPILImagesFromHDD(self, filePath):
        return Image.open(Config.param.data.path.datapath + filePath)

    def saveTensorsToHDD(self, tnsr, filePath):
        utils.saveTensorToNPY(tnsr, filePath)

    def loadTensorsFromHDD(self, filePath):
        return utils.loadNPYToTensor(filePath)

    def PIL2Tensor(self, pilImage):
        return transforms.ToTensor()(pilImage)







    def NPYMaker(self, filePath, preProc):
        
        PILImage = self.loadPILImagesFromHDD(filePath) #PIL
        PPedPILImage = self.torchvisionPreprocessing(pilImage, preProc)
        rstTensor = self.PIL2Tensor(PPedPILImage)

        self.saveTensorsToHDD(rstTensor, filePath)


    def methodNPYExists(self, filePath):

        tnsr = self.loadTensorsFromHDD(filePath).cuda()
        augedTensor = self.dataAugmentation(tnsr, self.augmentation)

        return augedTensor

    def methodNPYNotExists(self, filePath, preProc):

        PILImage = self.loadPILImagesFromHDD(filePath)
        PPedPILImage = self.torchvisionPreprocessing(PILImage, preProc)
        tnsr = self.PIL2Tensor(PPedPILImage).cuda()

        augedTensor = self.dataAugmentation(tnsr, self.augmentation)

        return augedTensor






    '''
    #Read File from HDD and store in memory
    #3*1024*1024 -> about 5000 pics in 64GB
    def H2M(self):

        #Calculate # of storable Batches
        sizePerTensor = self.calculateMemorySizePerTensor()
        cacheableMemorySize = Config.param.general.maxMemoryCachingSizeGB * 1024 * 1024 * 1024

        cacheBatchCount = cacheableMemorySize // sizePerTensor

        #Make Indexes
        indexes = []

        pass
    
    #Transfer Image / Tensor from memory to V-memory
    def M2V(self):
        pass
    '''


    def __getitem__(self, index):

        #popping File Path at GFL(self.globalFileList) by index
        filePath, preProc = self.popItemInGlobalFileListByIndex(index)

        if os.path.isfile(filePath + '.npy') is True:
            rst = self.methodNPYExists(filePath) #if .npy Exists, load preprocessed .npy File as Pytorch Tensor -> load to GPU directly -> Augmentation on GPU -> return
        else:
            rst = self.methodNPYNotExists(filePath, preProc) #if .npy doesn't Exists, load Image File as PIL Image -> Preprocess PIL Image on CPU -> convert to Tensor -> load to GPU -> Augmentation on GPU -> return

        rst = self.setTensorValueRange(rst)

        return rst

    def __len__(self):
        return max(list(map(len, datasetComponentList))) * len(datasetComponentList)
        
    


class DataLoader(torchDataLoader):
    def __init__(self, dataLoaderName: str, fromParam : bool = True):

        
        # INIT PARAMs #
        self.name = dataLoaderName
        self.datasetComponent = None
        self.batchSize = None
        self.samplingCount = None
        self.sameOutputSize = None
        self.valueRangeType = None
        self.shuffle = None
        self.augmentation = None
        self.numWorkers = None

        self.fromParam = fromParam

        if self.fromParam is True:
            self.getDataloaderParams()

        # CONSTRUCT DATASET #
        self.constructDataset()

        super().__init__(self, 
                         dataset = self.dataset,
                         batch_size = self.batchSize,
                         shuffle = self.shuffle,
                         num_workers = self.numWorkers)
    

    def getDataloaderParams(self):

        yamlData = Config.paramDict['data']['dataLoader'][f'{self.name}']

        self.datasetComponent = list(map(str, yamlData['datasetComponent']))

        self.batchSize = int(yamlData['batchSize'])
        self.samplingCount = int(yamlData['samplingCount'])
        self.sameOutputSize = bool(yamlData['sameOutputSize'])
        self.valueRangeType = str(yamlData['valueRangeType'])
        self.shuffle = str(yamlData['shuffle'])
        self.augmentation = yamlData['augmentation'] 
        self.numWorkers = Config.param.train.dataLoaderNumWorkers

    
    def constrctDataset(self):
        
        dataset = Dataset(self.datasetComponent)
        self.dataset = dataset




    

