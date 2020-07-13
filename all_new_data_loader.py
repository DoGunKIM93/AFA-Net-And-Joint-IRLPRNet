'''
all_new_data_loader.py
'''


#FROM Python LIBRARY
import os
import random
import math
import numpy as np
import yaml 
import inspect

from PIL import Image
from PIL import PngImagePlugin


#FROM PyTorch
import torch

from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder


#FROM This Project
import param as p
import backbone.preprocessing



'''
Standard Dataset Folder Structure

<Dataset Main Path>
├── <dataset name>
    ├── _data
│   └── members.yml
├── _drafts
│   ├── begin-with-the-crazy-ideas.md
│   └── on-simplicity-in-technology.md


'''


# Prevent memory Error in PIL
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

# Read Transforms from backbone.preprocessing automatically
PREPROCESSING_FUNCTION_DICT = dict(x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith('__')) and (inspect.isclass(x[1])) )




class DatasetConfig():


    def __init__(self, name, yamlDict):
        self.name = None
        self.output = None
        self.availableMode = None
        self.classes = None
        self.datasetSpecificPreprocessings = None

        self.getConfig(name, yamlDict)



    def strToList(self, str):
        return str.replace(' ','').split(',')
    


    def getConfig(self, name, yamlDict):

        yamlData = yamlDict[f'{name}']

        self.name = name

        self.output = self.strToList(str(yamlData['output']))

        self.availableMode = str(yamlData['availableMode'])

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
class DatasetBase(torchDataset):


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
    def __init__(self):
        pass

