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
            self.getDatasetConfig()


    @staticmethod
    def strToList(inp : str):
        return inp.replace(' ','').split(',')
    


    def getDatasetConfig(self):

        yamlData = Config.datasetConfigDict[f'{self.name}']

        self.origin = str(yamlData['origin'])

        self.dataType = yamlData['dataType']
        self.labelType = yamlData['labelType']


        self.availableMode = list(map(str, yamlData['availableMode']))
        self.classes = list(map(str, yamlData['classes']))
        self.datasetSpecificPreprocessings = list(map(str, yamlData['datasetSpecificPreprocessings']))

        





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

        #TODO: Call datalists that have different subclasses Number

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

        
        # INIT PARAMs #
        self.name = dataLoaderName
        self.datasetComponent = None
        self.batchSize = None
        self.samplingCount = None
        self.sameOutputSize = None
        self.valueRangeType = None
        self.shuffle = None
        self.preprocessing = None
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
        self.preprocessing = yamlData['preprocessing'] #TODO: IS IT WORKS?
        self.numWorkers = Config.param.train.dataLoaderNumWorkers

    
    def constrctDataset(self):
        
        dataset = Dataset(self.datasetComponent)
        self.dataset = dataset




    

