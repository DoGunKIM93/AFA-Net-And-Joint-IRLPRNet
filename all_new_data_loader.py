'''
data_loader.py
'''
version = "2.10.200729"


#FROM Python LIBRARY
import os
import random
import math
import numpy as np
import yaml 
import inspect
import itertools
import glob

from PIL import Image
from PIL import PngImagePlugin
from typing import List, Dict, Tuple, Union, Optional


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
                 origin : Optional[str] = None, 
                 dataType : Optional[Dict[str, str]] = None, 
                 labelType : Optional[Dict[str, str]] = None, 
                 availableMode : Optional[List[str]] = None, 
                 classes : Optional[List[str]] = None, 
                 preprocessings : Optional[List[str]] = None,
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



    def splitDataType(self, dataType: Dict[str, str]):
        #return form of {'LR': {'dataType': 'text', 'tensorType': 'int'}, 'HR': {'dataType': 'imageSequence', 'tensorType': 'double'}}

        return dict([key, dict(zip(['dataName', 'dataType', 'tensorType'], [key] + dataType[key].split('-') ))] for key in dataType if (dataType[key].split('-')[0] in ['text', 'image', 'imageSequence'] and dataType[key].split('-')[1] in ['float', 'int', 'long', 'double']))


    def getDatasetConfig(self):

        yamlData = Config.datasetConfigDict[f'{self.name}']

        self.origin = str(yamlData['origin'])

        self.dataType = self.splitDataType(yamlData['dataType'])

        self.labelType = self.splitDataType(yamlData['labelType'])


        self.availableMode = list(map(str, yamlData['availableMode']))
        self.classes = list(map(str, yamlData['classes']))
        self.preprocessings = list(map(str, yamlData['preprocessings'])) if yamlData['preprocessings'] is not None else []

        





#Representation of single dataset property
class DatasetComponent():


    def __init__(self, name, mode, classParameter):

        self.name = name

        self.datasetConfig = None
        self.getDatasetConfigByComponentName()
        

        self.mode = mode
        self.classParameter = classParameter


        self.dataFileList = None #same length (2-d List)
        self.labelFileList = None
        self.getDataFileList()
        

        self.preprocessingList = None
        self.makePreprocessingList()
    
    def getDatasetConfigByComponentName(self):
        self.datasetConfig = DatasetConfig(Config.paramDict['data']['datasetComponent'][self.name]['dataConfig'])

    def getDataFileList(self):

        #TODO: LABEL
        #TODO: SEQUENCE

        # get origin path
        mainPath = Config.param.data.path.datasetPath
        path = f"{self.datasetConfig.origin}/{self.mode}/"

        #TODO: PLZ DEBUG
        '''
        # add label file path that independents about each label files (Without duplicate label)
        self.labelFileList = [file for file in os.listdir(mainPath + path) if (file.endswith(".txt"))]

        if self.labelFileList is not None and self.classParameter[self.datasetConfig.classes] == '$ALL':
            datasetConfigClasses = list(filter(os.path.isdir, glob.glob(mainPath + path+"*"))) #TODO: TEST
        else:
            datasetConfigClasses = self.datasetConfig.classes
        '''

        datasetConfigClasses = self.datasetConfig.classes
        # dynamic construction of class path based on defined classes
        for i in range(len(datasetConfigClasses)):
            classPathList = list(itertools.chain.from_iterable(list(map( lambda y : list(map(lambda x : str(x) if type(x) is int else x + '/' + str(y) if type(y) is int else y , classPathList)), self.classParameter[datasetConfigClasses[i]])))) if i is not 0 else self.classParameter[datasetConfigClasses[i]]
            # Sorry for who read this

        # add origin path in front of all elements of class path list
        pathList = list(map( lambda x : path + x , classPathList))
        
        # construct all of readable file lists in class path lists
        dataFileLists = [ x for x in list(map( lambda x :  list(map( lambda y : x + "/" + y, os.listdir(mainPath + x))) , pathList)) if (x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp"))  ]
        assert [len(dataFileLists[0])] * len(dataFileLists) == list(map(len, dataFileLists)), f'data_loader.py :: ERROR! dataset {self.name} has NOT same count of data files for each classes.'

        
        #LABEL
        if isLabel:
            labelPath = f'{path}GT/'
            labelFiles = os.listdir(mainPath + x)

            if len(labelFiles) == 1:
                #Label case .txt file
                labelFiles = labelFiles * len(dataFileLists[0])
            else:
                #Label case for same count of image files (GT)
                assert [len(labelFiles)] * len(dataFileLists) == list(map(len, dataFileLists)), f'data_loader.py :: ERROR! label and data files should be had same count. (dataset {self.name})'

        else:
            labelFiles = [None] * len(dataFileLists[0])

        labelFiles = [labelFiles] * len(dataFileLists)

        dataFileDictList = list( map( lambda x: dict(zip(['dataFilePath','labelFilePath'], x)), list(zip(       itertools.chain.from_iterable(dataFileLists), itertools.chain.from_iterable(labelFiles)   )) ))

        # set dataFileList without main path
        self.dataFileList = dataFileDictList



    def makePreprocessingList(self):
        self.preprocessingList = list(map((lambda x: PREPROCESSING_DICT[x.split('(')[0]](*list(filter(lambda y : y != '', x.split('(')[1][:-1].replace(' ','').split(','))))), self.datasetConfig.preprocessings))
        pass




    def __len__(self):
        return len(self.dataFileList)







#TODO: Distribution
#TODO: On-memory Supply 
class Dataset(torchDataset):


    def __init__(self, 
                 datasetComponentParamList: List, 
                 batchSize: int, 
                 samplingCount:int, 
                 sameOutputSize:bool, 
                 valueRangeType:str, 
                 shuffle:bool, 
                 augmentation:List[str], 
                 numWorkers:int,
                 makePreprocessedFile:bool):
        self.datasetComponentParamList = datasetComponentParamList

        self.datasetComponentObjectList = None
        self.constructDatasetComponents()
        self.datasetComponentObjectListIntegrityTest()

        self.batchSize = batchSize
        self.samplingCount = samplingCount
        self.sameOutputSize = sameOutputSize
        self.valueRangeType = valueRangeType
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.numWorkers = numWorkers

        self.makePreprocessedFile = makePreprocessedFile

        #self.getDatasetRatio(self.datasetComponentObjectList)

        self.globalFileList = None
        self.makeGlobalFileList(self.shuffle)

        self.globalLabelList = None
        self.makeGlobalLabelList(self.shuffle)


    def makeGlobalFileList(self, shuffle):
        gFL = [x.dataFileList for x in self.datasetComponentObjectList]
        self.globalFileList = gFL

    def makeGlobalLabelList(self, shuffle):
        print("self.datasetComponentObjectList : ", self.datasetComponentObjectList)
        gLL = [x.labelFileList for x in self.datasetComponentObjectList]
        self.globalLabelList = gLL


    def popItemInGlobalFileListByIndex(self, index):
        datasetComponentLengthList = list(map(len, self.datasetComponentObjectList))

        indexComponent = index % len(datasetComponentLengthList)
        indexComponentFileList = ( index // len(datasetComponentLengthList) ) % datasetComponentLengthList[indexComponent]

        return {'dataFilePath':  Config.param.data.path.datasetPath + self.globalFileList[indexComponent][indexComponentFileList]['dataFilePath'], 
                'labelFilePath': Config.param.data.path.datasetPath + self.globalFileList[indexComponent][indexComponentFileList]['labelFilePath'], 
                'preprocessing': self.datasetComponentObjectList[indexComponent].preprocessingList, 
                'dataType':      self.datasetComponentObjectList[indexComponent].datasetConfig.dataType,
                'labelType':     self.datasetComponentObjectList[indexComponent].datasetConfig.labelType}

    # self.globalLabelList에 있는 내용 load 해서 dict 형식으로 만든 후 반환 --> 1 ## list [dict, dict, dict]
    # init에서 한번 호출하는 방식으로 list [dict, dict, dict] 만들어 놓기
    #def getItemInGlobalLabelDictByIndex(self, index):


    def constructDatasetComponents(self):
        self.datasetComponentObjectList = [DatasetComponent(*x) for x in self.datasetComponentParamList]

    def datasetComponentObjectListIntegrityTest(self):
        # Test All dataComponents have same 
        # dataType
        # TensorType
        # name
        datasetComponentdataTypeList =  [x.datasetConfig.dataType for x in self.datasetComponentObjectList]
        assert [datasetComponentdataTypeList[0]] * len(datasetComponentdataTypeList) == datasetComponentdataTypeList, 'data_loader.py :: All datasets in dataloader must have same dataType.'

        datasetComponentlabelTypeList =  [x.datasetConfig.labelType for x in self.datasetComponentObjectList]
        assert [datasetComponentlabelTypeList[0]] * len(datasetComponentlabelTypeList) == datasetComponentlabelTypeList, 'data_loader.py :: All datasets in dataloader must have same tensorType.'
       


    def getDatasetRatio(self, datasetComponentObjectList:list):
        self.datasetLengthList = list(map(len, datasetComponentObjectList))
        self.datasetRatio = list(map((lambda x : x / max(datasetLengthList)), datasetLengthList))



    def preprocessing(self, batchInput, preprocessings:list):
        rst = batchInput
        for preprocessingFunction in preprocessings:
            rst = preprocessingFunction(rst)

        return rst

    
    def calculateMemorySizePerTensor(self, dtype:torch.dtype, expectedShape:List[int]): #WARN: EXCEPT BATCH SIIZEEEEEE!!!!!!!!!!!!!!!!!!!!!!!
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




    def dataAugmentation(self, tnsr, augmentations: List[str]):

        x = tnsr

        for augmentation in augmentations:
            x = applyAugmentationFunction(x, augmentation)

        return x

    def setTensorValueRange(self, tnsr, valueRangeType:str):

        if valueRangeType == '-1~1':
            tnsr = tnsr * 2 - 1

        return tnsr
    

    def loadPILImagesFromHDD(self, filePath):
        return Image.open(filePath)

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

        #augedTensor = self.dataAugmentation(rstTensor.cuda(), self.augmentation)

        return rstTensor


    def methodNPYExists(self, filePath):

        tnsr = self.loadTensorsFromHDD(filePath)
        #augedTensor = self.dataAugmentation(tnsr, self.augmentation)

        return tnsr

    def methodNPYNotExists(self, filePath, preProc):

        PILImage = self.loadPILImagesFromHDD(filePath)
        PPedPILImage = self.torchvisionPreprocessing(PILImage, preProc)
        tnsr = self.PIL2Tensor(PPedPILImage)

        #augedTensor = self.dataAugmentation(tnsr, self.augmentation)

        return tnsr

    #def LabelProcess(self, ):





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
        popped = self.popItemInGlobalFileListByIndex(index)
        dataFilePath = popped['filePath']
        labelFilePath = popped['labelFilePath']
        preProc = popped['preprocessing']
        dataType = popped['dataType']
        labelType = popped['labelType']

        rst = {}

        #
        # ADD DATA
        #

        ## 2. Dataset에서 init시 호출해서 생성한 dic에 filePath 중 filename 또는 그 상위 path만을 key로 해당하는 value 매칭
        if os.path.isfile(filePath + '.npy') is True:
            rst = self.methodNPYExists(filePath) #if .npy Exists, load preprocessed .npy File as Pytorch Tensor -> load to GPU directly -> Augmentation on GPU -> return
        else:
            if self.makePreprocessedFile is True:
                rst = self.NPYMaker(filePath, preProc) # if .npy doesn't Exists and self.makePreprocessedFile is True, make .npy file and augmentating tensor and return
            else:
                rst = self.methodNPYNotExists(filePath, preProc) #if .npy doesn't Exists, load Image File as PIL Image -> Preprocess PIL Image on CPU -> convert to Tensor -> load to GPU -> Augmentation on GPU -> return

        rst[dataType['dataName']] = self.setTensorValueRange(rst)

        #
        # ADD LABEL
        #

        rst[labelType['dataName']] = SOMETHING #TODO:

        ## RETURN DICT OF 
        return rst

    def __len__(self):
        return max(list(map(len, self.datasetComponentObjectList))) * len(self.datasetComponentObjectList)
        
    


class DataLoader(torchDataLoaders):
    def __init__(self, 
                 dataLoaderName: str, 
                 fromParam : bool = True, 
                 datasetComponentParamList: Optional[List[str]] = None,
                 batchSize: Optional[int] = None,
                 samplingCount: Optional[int] = None,
                 sameOutputSize: Optional[bool] = None,
                 valueRangeType: Optional[str] = None,
                 shuffle: Optional[bool] = None,
                 augmentation: Optional[List[str]] = None,
                 numWorkers: Optional[int] = None,
                 makePreprocessedFile: Optional[bool] = None):

        
        # INIT PARAMs #
        self.name = dataLoaderName
        self.datasetComponentParamList = datasetComponentParamList
        self.batchSize = batchSize
        self.samplingCount = samplingCount
        self.sameOutputSize = sameOutputSize
        self.valueRangeType = valueRangeType
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.numWorkers = numWorkers
        self.makePreprocessedFile = makePreprocessedFile

        self.fromParam = fromParam

        if self.fromParam is True:
            self.getDataloaderParams()

        # CONSTRUCT DATASET #
        self.dataset = None
        self.constructDataset()

        super(DataLoader, self).__init__(
                         dataset = self.dataset,
                         batch_size = self.batchSize,
                         shuffle = self.shuffle,
                         num_workers = self.numWorkers,
                         collate_fn = self.GPUAugmentataionCollater)
    

    def GPUAugmentataionCollater(self, samples):
        rstDict = {}
        for key in samples:
            rstDict[key] = torch.stack([sample[key] for sample in samples])
        return rstDict


    def getDataloaderParams(self):

        yamlData = Config.paramDict['data']['dataLoader'][f'{self.name}']

        datasetNameList = list(map(str, yamlData['datasetComponent']))
        datasetModeList = list( Config.paramDict['data']['datasetComponent'][name]['mode'] for name in datasetNameList  )
        datasetClassParameterList = list( Config.paramDict['data']['datasetComponent'][name]['classParameter'] for name in datasetNameList  )

        self.datasetComponentParamList = zip(datasetNameList, datasetModeList, datasetClassParameterList)


        self.batchSize = int(yamlData['batchSize'])
        self.samplingCount = int(yamlData['samplingCount'])
        self.sameOutputSize = bool(yamlData['sameOutputSize'])
        self.valueRangeType = str(yamlData['valueRangeType'])
        self.shuffle = str(yamlData['shuffle'])
        self.augmentation = yamlData['augmentation'] 
        self.numWorkers = Config.param.train.dataLoaderNumWorkers
        self.makePreprocessedFile = yamlData['makePreprocessedFile'] 

    
    def constructDataset(self):
        
        self.dataset = Dataset(datasetComponentParamList = self.datasetComponentParamList, 
                               batchSize = self.batchSize, 
                               samplingCount = self.samplingCount, 
                               sameOutputSize = self.sameOutputSize, 
                               valueRangeType = self.valueRangeType, 
                               shuffle = self.shuffle, 
                               augmentation = self.augmentation, 
                               numWorkers = self.numWorkers,
                               makePreprocessedFile = self.makePreprocessedFile)




    

