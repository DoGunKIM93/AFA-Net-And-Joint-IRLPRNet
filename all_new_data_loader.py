'''
all_new_data_loader.py
'''
version = "1.01.200716.1"


#FROM Python LIBRARY
import os
import random
import math
import numpy as np
import yaml 
import inspect

from PIL import Image
from PIL import PngImagePlugin
from typing import List, Dict, Tuple, Union


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
from backbone.config import Config




# Prevent memory Error in PIL
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# Read Transforms from backbone.preprocessing automatically
PREPROCESSING_FUNCTION_DICT = dict(x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith('__')) and (inspect.isclass(x[1])) )




class DatasetConfig():


    def __init__(self, 
                 name: str, 
                 origin : str = None, 
                 dataType : List[str] = None, 
                 labelType : List[str] = None, 
                 availableMode : List[str] = None, 
                 classes : List[str] = None, 
                 datasetSpecificPreprocessings : List[str] = None,
                 useDatasetConfig : bool = True):
                 
        self.name = name

        self.origin = origin
        self.dataType = dataType
        self.labelType = labelType
        self.availableMode = availableMode
        self.classes = classes
        self.datasetSpecificPreprocessings = datasetSpecificPreprocessings

        self.useDatasetConfig = useDatasetConfig

        if self.useDatasetConfig is True:
            self.getDatasetConfig(Config.datasetConfigDict)


    @staticmethod
    def strToList(inp : str):
        return inp.replace(' ','').split(',')
    


    def getDatasetConfig(self, yamlDict):

        yamlData = yamlDict[f'{self.name}']\

        self.origin = str(yamlData['origin'])

        self.dataType = list(map(str, yamlData['dataType']))
        self.labelType = list(map(str, yamlData['labelType']))
        self.availableMode = list(map(str, yamlData['availableMode']))
        self.classes = list(map(str, yamlData['classes']))
        self.datasetSpecificPreprocessings = list(map(str, yamlData['datasetSpecificPreprocessings']))

        





#Representation of single dataset property
class DatasetComponent():


    def __init__(self, config):

        self.config = config
        self.name = config.name


        self.dataFileList = None #same length (2-d List)
        self.labelFileList = None
        self.dataClassCount = None
        self.getDataList()
        

        self.preprocessingList = None
        self.makePreprocessingList()
    
    def getDataList(self):
        #TODO: get datalist from config

        path = 

        self.dataClassCount = len(self.dataFileList)
        pass


    def makePreprocessingList(self):
        self.preprocessingList = list(map((lambda x: PREPROCESSING_FUNCTION_DICT[x]), self.config.preprocessings))
        pass


    def __len__(self):
        return len(dataFileList[0])







#TODO: Distribution
#TODO: On-memory Supply 
class Dataset(torchDataset):


    def __init__(self, datasetComponentList:list):
        self.datasetComponentList = datasetComponentList
        self.datasetComponentListIntegrityTest()
         
        self.getDatasetRatio(datasetComponentList)



    def datasetComponentListIntegrityTest(self):
        # Test All dataComponents have same output type.
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

    
    def getH2MSize(self):
        pass
    
    def getM2VSize(self):
        pass

    #Read File from HDD and store in memory
    def H2M(self):
        pass
    
    #Transfer Image / Tensor from memory to V-memory
    def M2V(self):
        pass



    #TODO:
    def __getitem__(self, index):
        
        # HDD to Memory (H2Msize 만큼)
        # STATIC P.P 
        # Memory to V-memory (M2Vsize 만큼)
        # Dinamic P.P
        # Return

        pass

    #TODO: define max length of Dataset
    def __len__(self):
        return max(self.datasetLengthList)
        
    


class DataLoader(torchDataLoader):
    def __init__(self, dataLoaderName: str, fromParam : bool = True):

        

        self.name = dataLoaderName
        self.datasetComponent = None
        self.batchSize = None
        self.samplingCount = None
        self.sameOutputSize = None
        self.valueRangeType = None
        self.shuffle = None
        self.preprocessing = None

        self.fromParam = fromParam

        if self.fromParam is True:
            self.getDataloaderParams(Config.paramDict)


        super().__init__(self, 
                            batch_size = batchSize,
                            shuffle = shuffle,
                            num_workers = 16)
    

    def getDataloaderParams(self, yamlDict):

        yamlData = yamlDict[f'{self.name}']

        self.datasetComponent = list(map(str, yamlData['datasetComponent']))

        self.batchSize = int(yamlData['batchSize'])
        self.samplingCount = int(yamlData['samplingCount'])
        self.sameOutputSize = bool(yamlData['sameOutputSize'])
        self.valueRangeType = str(yamlData['valueRangeType'])
        self.shuffle = str(yamlData['shuffle'])
        self.preprocessing = yamlData['preprocessing'] #TODO: IS IT WORKS?



    

