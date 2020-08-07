'''
data_loader.py
'''
version = "2.20.200805"


#FROM Python LIBRARY
import os
import random
import math
import numpy as np
import yaml 
import inspect
import itertools
import glob
import time

from PIL import Image
from PIL import PngImagePlugin
from typing import List, Dict, Tuple, Union, Optional


#FROM PyTorch
import torch

from torch._utils import ExceptionWrapper
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter, _DatasetKind
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
        #return form of {'dataName': 'LR', dataType': 'text', 'tensorType': 'int'}
        
        dataTypeList = ['Text', 'Image', 'ImageSequence'] 
        tensorTypeList = ['Float', 'Int', 'Long', 'Double']



        if dataType is not None:
            for key in dataType:
                assert dataType[key].split('-')[0] in dataTypeList, f'data_loader.py :: dataset config {self.name} has invalid dataType.'
                assert dataType[key].split('-')[1] in tensorTypeList, f'data_loader.py :: dataset config {self.name} has invalid tensorType.'
            return dict(zip(['dataName', 'dataType', 'tensorType'], [key] + dataType[key].split('-') ))
        else:
            return {}

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



        datasetConfigClasses = self.datasetConfig.classes
        # dynamic construction of class path based on defined classes
        for i in range(len(datasetConfigClasses)):
            classPathList = list(itertools.chain.from_iterable(list(map( lambda y : list(map(lambda x : str(x) if type(x) is int else x + '/' + str(y) if type(y) is int else y , classPathList)), self.classParameter[datasetConfigClasses[i]])))) if i is not 0 else self.classParameter[datasetConfigClasses[i]]
            # Sorry for who read this

        # add origin path in front of all elements of class path list
        pathList = list(map( lambda x : path + x , classPathList))



        if self.datasetConfig.dataType['dataType'] == 'Text':
            # construct all of readable file lists in class path lists
            dataFileLists = [ x for x in list(map( lambda x :  list(filter( lambda x:(x.endswith(".txt")), list(map( lambda y : x + "/" + y, sorted(os.listdir(mainPath + x)))) )), pathList))  ]

        elif self.datasetConfig.dataType['dataType'] == 'Image':
            # construct all of readable file lists in class path lists
            dataFileLists = [ x for x in list(map( lambda x :  list(filter( lambda x:(x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp")), list(map( lambda y : x + "/" + y, sorted(os.listdir(mainPath + x)))) )), pathList))  ]

        elif self.datasetConfig.dataType['dataType'] == 'ImageSequence':
            # construct all of sequence folders in class path lists
            dataFileLists = [ x for x in list(map( lambda x : list(map( lambda y : x + "/" + y, sorted(os.listdir(mainPath + x)) )), pathList))  ]



        
        #if label Exists
        if len(self.datasetConfig.labelType) > 0:

            labelPath = f'{path}GT/'
            
            if self.datasetConfig.labelType['dataType'] == 'Text':
                # construct all of readable file lists in class path lists
                labelFiles = sorted(list(filter( lambda x:(x.endswith(".txt")), os.listdir(mainPath + labelPath))))

            elif self.datasetConfig.labelType['dataType'] == 'Image':
                # construct all of readable file lists in class path lists
                labelFiles = sorted(list(filter( lambda x:(x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp")), os.listdir(mainPath + labelPath))))

            elif self.datasetConfig.labelType['dataType'] == 'ImageSequence':
                # construct all of sequence folders in class path lists
                labelFiles = sorted(os.listdir(mainPath + labelPath))


            
            # add origin path in front of all elements of label file list
            labelFiles = list(map( lambda x : labelPath + x , labelFiles))

            if len(labelFiles) == 1:
                #Label case ex (.txt file)
                labelFiles = labelFiles * len(dataFileLists[0])
            else:
                #Label case for same count of image files (GT)
                assert [len(dataFileLists[0])] * len(dataFileLists) == list(map(len, dataFileLists)), f'data_loader.py :: ERROR! dataset {self.name} has NOT same count of data files for each classes.'
                assert [len(labelFiles)] * len(dataFileLists) == list(map(len, dataFileLists)), f'data_loader.py :: ERROR! label and data files should be had same count. (dataset {self.name})'
                
            
        else:
            labelFiles = [None] * len(dataFileLists[0])

        labelFiles = [labelFiles] * len(dataFileLists)

        dataFileDictList = list( map( lambda x: dict(zip(['dataFilePath','labelFilePath'], x)), list(zip(       itertools.chain.from_iterable(dataFileLists), itertools.chain.from_iterable(labelFiles)   )) ))

        # set dataFileList without main path
        self.dataFileList = dataFileDictList



    def makePreprocessingList(self):
        self.preprocessingList = list(map((lambda x: PREPROCESSING_DICT[x.split('(')[0]](*list(filter(lambda y : y != '', x.split('(')[1][:-1].replace(' ','').split(','))))), self.datasetConfig.preprocessings))



    def __len__(self):
        return len(self.dataFileList)







#TODO: Distribution
#TODO: On-memory Supply 
class Dataset(torchDataset):


    def __init__(self, 
                 datasetComponentParamList: List, 
                 batchSize: int, 
                 samplingCount:int, 
                 valueRangeType:str, 
                 shuffle:bool, 
                 augmentation:List[str], 
                 numWorkers:int,
                 makePreprocessedFile:bool):
        self.datasetComponentParamList = datasetComponentParamList

        self.datasetComponentObjectList = None
        self._constructDatasetComponents()
        self._datasetComponentObjectListIntegrityTest()

        self.batchSize = batchSize
        self.samplingCount = samplingCount
        self.valueRangeType = valueRangeType
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.numWorkers = numWorkers

        self.makePreprocessedFile = makePreprocessedFile

        self.globalFileList = None
        self._makeGlobalFileList(self.shuffle)

        self.PILImageToTensorFunction = transforms.ToTensor()


    def _makeGlobalFileList(self, shuffle):
        gFL = [x.dataFileList for x in self.datasetComponentObjectList]
        #print(gFL)
        self.globalFileList = gFL


    def _popItemInGlobalFileListByIndex(self, index):
        datasetComponentLengthList = list(map(len, self.datasetComponentObjectList))

        indexComponent = index % len(datasetComponentLengthList)
        indexComponentFileList = ( index // len(datasetComponentLengthList) ) % datasetComponentLengthList[indexComponent]

        return {'dataFilePath':  Config.param.data.path.datasetPath + self.globalFileList[indexComponent][indexComponentFileList]['dataFilePath'], 
                'labelFilePath': Config.param.data.path.datasetPath + self.globalFileList[indexComponent][indexComponentFileList]['labelFilePath'] if self.globalFileList[indexComponent][indexComponentFileList]['labelFilePath'] is not None else None, 
                }



    def _constructDatasetComponents(self):
        self.datasetComponentObjectList = [DatasetComponent(*x) for x in self.datasetComponentParamList]

    def _datasetComponentObjectListIntegrityTest(self):
        # Test All dataComponents have same 
        # dataType
        # TensorType
        # name
        datasetComponentdataTypeList =  [x.datasetConfig.dataType for x in self.datasetComponentObjectList]
        assert [datasetComponentdataTypeList[0]] * len(datasetComponentdataTypeList) == datasetComponentdataTypeList, 'data_loader.py :: All datasets in dataloader must have same dataType.'

        datasetComponentlabelTypeList =  [x.datasetConfig.labelType for x in self.datasetComponentObjectList]
        assert [datasetComponentlabelTypeList[0]] * len(datasetComponentlabelTypeList) == datasetComponentlabelTypeList, 'data_loader.py :: All datasets in dataloader must have same tensorType.'
       


    
    def _calculateMemorySizePerTensor(self, dtype:torch.dtype, expectedShape:List[int]): #WARN: EXCEPT BATCH SIIZEEEEEE!!!!!!!!!!!!!!!!!!!!!!!
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

    

    def _torchvisionPreprocessing(self, pilImage, preProc):

        x = pilImage
        for preProcFunc in preProc:
            x = preProcFunc(x)

        return x

    



    def _applyAugmentationFunction(self, tnsr, augmentationFuncStr:str):

        assert augmentationFuncStr.split('(')[0] in AUGMENTATION_DICT.keys(), "data_loader.py :: invalid Augmentation Function!! chcek param.yaml."

        augFunc = AUGMENTATION_DICT[augmentationFuncStr.split('(')[0]]
        args = list( int(x) for x in list(filter(lambda y : y != '', augmentationFuncStr.split('(')[1][:-1].replace(' ','').split(',') )) )

        tnsr = augFunc(tnsr, *args)

        return tnsr

    # DATA AUGMENTATION ON CPU - Multiprocessing with Forked Worker processes - 
    # Before toTensor() Augmentation
    # CROP OPERATION usually be here
    # Make Sure that ALL output Size (C, H, W) are same.
    def _dataAugmentation(self, tnsrList, augmentations: List[str]):

        # If input is a list of two tensors -> make it to list of list of two tensors
        # The standard Input type is list of list of two tensors -> [  [data_1, label_1], ... , [data_N, label_N]  ]
        if not isinstance(tnsrList[0], list):
            xList = [tnsrList]
        else:
            xList = tnsrList

        augedXList = []

        for x in xList:
            for augmentation in augmentations:
                x = self._applyAugmentationFunction(x, augmentation)
                #print(augmentation)
                if augmentation == 'toTensor()':
                    break
            augedXList.append(x)

        #TRANSPOSE
        augedXList = list(map(list, zip(*augedXList)))

        augedXTensorList = [torch.stack(x) for x in augedXList]

        return augedXTensorList


    def _setTensorValueRange(self, tnsr, valueRangeType:str):

        if valueRangeType == '-1~1':
            tnsr = tnsr * 2 - 1

        return tnsr
    

    def _loadPILImagesFromHDD(self, filePath):
        return Image.open(filePath)

    def _saveTensorsToHDD(self, tnsr, filePath):
        print(f'Write Tensor to {filePath}.npy...')
        utils.saveTensorToNPY(tnsr, filePath)

    def _loadNPArrayFromHDD(self, filePath):
        return utils.loadNPY(filePath)

    def _PIL2Tensor(self, pilImage):
        return self.PILImageToTensorFunction(pilImage)






    def _NPYMaker(self, filePath, preProc):
        PILImage = self._loadPILImagesFromHDD(filePath) #PIL
        PPedPILImage = self._torchvisionPreprocessing(PILImage, preProc)
        rstTensor = self._PIL2Tensor(PPedPILImage)

        self._saveTensorsToHDD(rstTensor, filePath)

        return rstTensor


    def _methodNPYExists(self, filePath):
        npa = self._loadNPArrayFromHDD(filePath)
        
        return npa


    def _methodNPYNotExists(self, filePath, preProc):
        PILImage = self._loadPILImagesFromHDD(filePath)
        PPedPILImage = self._torchvisionPreprocessing(PILImage, preProc)

        return PPedPILImage





    
    def _readItem(self, Type, FilePath, preProc):
        # data information defined in config
        if FilePath is not None:


            ###################################################################################
            # CASE TEXT
            ###################################################################################
            if Type['dataType'] == 'Text':
                pass


            ###################################################################################
            # CASE IMAGE
            ###################################################################################
            elif Type['dataType'] == 'Image':

                if os.path.isfile(FilePath + '.npy') is True:
                    rst = self._methodNPYExists(FilePath + '.npy') #if .npy Exists, load preprocessed .npy File as Pytorch Tensor -> load to GPU directly -> Augmentation on GPU -> return
                else:
                    if self.makePreprocessedFile is True:
                        rst = self._NPYMaker(FilePath, preProc) # if .npy doesn't Exists and self.makePreprocessedFile is True, make .npy file and augmentating tensor and return
                    else:
                        rst = self._methodNPYNotExists(FilePath, preProc) #if .npy doesn't Exists, load Image File as PIL Image -> Preprocess PIL Image on CPU -> convert to Tensor -> load to GPU -> Augmentation on GPU -> return


            ###################################################################################
            # CASE IMAGESEQUENCE
            ###################################################################################
            elif Type['dataType'] == 'ImageSequence':

                seqFileList = sorted(os.listdir(FilePath))
                rstList = []

                for seqFile in seqFileList:
                    seqFilePath = FilePath + '/' + seqFile
                    if os.path.isfile(seqFilePath + '.npy') is True:
                        rst = self._methodNPYExists(seqFilePath + '.npy') #if .npy Exists, load preprocessed .npy File as Pytorch Tensor -> load to GPU directly -> Augmentation on GPU -> return
                    else:
                        if self.makePreprocessedFile is True:
                            rst = self._NPYMaker(seqFilePath, preProc) # if .npy doesn't Exists and self.makePreprocessedFile is True, make .npy file and augmentating tensor and return
                        else:
                            rst = self._methodNPYNotExists(seqFilePath, preProc) #if .npy doesn't Exists, load Image File as PIL Image -> Preprocess PIL Image on CPU -> convert to Tensor -> load to GPU -> Augmentation on GPU -> return
                    rstList.append(rst)
                rst = rstList
            
        # data information not defined in config
        else:
            rst = None

        return rst





    def __getitem__(self, index):

        #popping File Path at GFL(self.globalFileList) by index
        popped = self._popItemInGlobalFileListByIndex(index)
        dataFilePath = popped['dataFilePath']
        labelFilePath = popped['labelFilePath']

        componentIndex = index % len(self.datasetComponentObjectList)

        preProc = self.datasetComponentObjectList[componentIndex].preprocessingList
        dataType = self.datasetComponentObjectList[componentIndex].datasetConfig.dataType
        labelType = self.datasetComponentObjectList[componentIndex].datasetConfig.labelType

        rstDict = {}


        filePathList = [dataFilePath, labelFilePath]
        typeList = [dataType, labelType]


        ###################################################################################
        # ADD DATA & LABEL
        ###################################################################################


        for Type, FilePath in zip(typeList, filePathList):
            rstDict[Type['dataName']] = self._readItem(Type, FilePath, preProc)






        ###################################################################################
        # Data Augmentation
        ###################################################################################

        rstDict[dataType['dataName']], rstDict[labelType['dataName']] = self._dataAugmentation( [rstDict[dataType['dataName']], rstDict[labelType['dataName']]], self.augmentation )
        
        rstDict[dataType['dataName']] = self._setTensorValueRange(rstDict[dataType['dataName']], self.valueRangeType)

        rstDict[labelType['dataName']] = self._setTensorValueRange(rstDict[labelType['dataName']], self.valueRangeType)






        ###################################################################################
        # Data Demension Align
        ###################################################################################


        if dataType['dataType'] == 'Text':
            pass
        elif dataType['dataType'] == 'Image':
            assert len(rstDict[dataType['dataName']].size()) == 4
            rstDict[dataType['dataName']] = rstDict[dataType['dataName']].squeeze(0)
        elif dataType['dataType'] == 'ImageSequence':
            assert len(rstDict[dataType['dataName']].size()) == 4


        if labelType['dataType'] == 'Text':
            pass
        elif labelType['dataType'] == 'Image':
            assert len(rstDict[labelType['dataName']].size()) == 4
            rstDict[labelType['dataName']] = rstDict[labelType['dataName']].squeeze(0)
        elif labelType['dataType'] == 'ImageSequence':
            assert len(rstDict[labelType['dataName']].size()) == 4






        ## RETURN DICT OF 
        return rstDict

    def __len__(self):
        return max(list(map(len, self.datasetComponentObjectList))) * len(self.datasetComponentObjectList)
        
    


class DataLoader(torchDataLoader):
    def __init__(self, 
                 dataLoaderName: str, 
                 fromParam : bool = True, 
                 datasetComponentParamList: Optional[List[str]] = None,
                 batchSize: Optional[int] = None,
                 samplingCount: Optional[int] = None,
                 valueRangeType: Optional[str] = None,
                 shuffle: Optional[bool] = None,
                 augmentation: Optional[List[str]] = None,
                 numWorkers: Optional[int] = None,
                 makePreprocessedFile: Optional[bool] = None):

        
        

        # INIT PARAMs #
        self.name = dataLoaderName
        print(f"Preparing Dataloader {self.name}... ")

        self.datasetComponentParamList = datasetComponentParamList
        self.batchSize = batchSize
        self.samplingCount = samplingCount
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
                         collate_fn = self.Collater)

        print(f"Data prepared : {len(self.dataset)} data")
        
    


    def Collater(self, samples):
        rstDict = {}
        
        for key in samples[0]:
            tnsrList = [sample[key] for sample in samples]
            rstDict[key] = tnsrList

        return rstDict


    def getDataloaderParams(self):

        yamlData = Config.paramDict['data']['dataLoader'][f'{self.name}']

        datasetNameList = list(map(str, yamlData['datasetComponent']))
        datasetModeList = list( Config.paramDict['data']['datasetComponent'][name]['mode'] for name in datasetNameList  )
        datasetClassParameterList = list( Config.paramDict['data']['datasetComponent'][name]['classParameter'] for name in datasetNameList  )

        self.datasetComponentParamList = zip(datasetNameList, datasetModeList, datasetClassParameterList)


        self.batchSize = int(yamlData['batchSize'])
        self.samplingCount = int(yamlData['samplingCount'])
        self.valueRangeType = str(yamlData['valueRangeType'])
        self.shuffle = str(yamlData['shuffle'])
        self.augmentation = yamlData['augmentation'] 
        self.numWorkers = Config.param.train.dataLoaderNumWorkers
        self.makePreprocessedFile = yamlData['makePreprocessedFile'] 

    
    def constructDataset(self):
        
        self.dataset = Dataset(datasetComponentParamList = self.datasetComponentParamList, 
                               batchSize = self.batchSize, 
                               samplingCount = self.samplingCount, 
                               valueRangeType = self.valueRangeType, 
                               shuffle = self.shuffle, 
                               augmentation = self.augmentation, 
                               numWorkers = self.numWorkers,
                               makePreprocessedFile = self.makePreprocessedFile)


    def __iter__(self) -> '_BaseDataLoaderIter':
        assert self.num_workers > 0, "data_loader.py :: Current Version of Data Loader Only Support more than one num_workers."
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIterWithDataAugmentation(self, self.augmentation)




    



class _MultiProcessingDataLoaderIterWithDataAugmentation(_MultiProcessingDataLoaderIter):

    def __init__(self, loader, augmentation):
        self.augmentation = augmentation
        super(_MultiProcessingDataLoaderIterWithDataAugmentation, self).__init__(loader)


    def _applyAugmentationFunction(self, tnsr, augmentationFuncStr:str):

        assert augmentationFuncStr.split('(')[0] in AUGMENTATION_DICT.keys(), "data_loader.py :: invalid Augmentation Function!! chcek param.yaml."

        augFunc = AUGMENTATION_DICT[augmentationFuncStr.split('(')[0]]
        args = list( int(x) for x in list(filter(lambda y : y != '', augmentationFuncStr.split('(')[1][:-1].replace(' ','').split(',') )) )

        tnsr = augFunc(tnsr, *args)

        return tnsr


    def _GPUDataAugmentation(self, tnsrList, augmentations: List[str]):

        x = tnsrList

        augmentationsAfterToTensor = augmentations[augmentations.index('toTensor()') + 1:]

        for augmentation in augmentationsAfterToTensor:
            x = self._applyAugmentationFunction(x, augmentation)

        return x



    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        
        #a = time.perf_counter()
        AugedTensor = {}
        AugedTList = self._GPUDataAugmentation( [ torch.stack(data[key]).cuda() for key in data ], self.augmentation )
        
        for i, key in enumerate(data):
            AugedTensor[key] = AugedTList[i]

        #print(time.perf_counter() - a)
        return AugedTensor

    def _next_data(self): 
        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1

            if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
                if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    self._shutdown_worker(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx] 
                return self._process_data(data) #CUDA MULTIPROC ERROR HERE!!