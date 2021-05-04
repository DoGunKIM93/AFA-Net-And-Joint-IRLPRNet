# FROM Python LIBRARY
import random
import numpy as np
import time
import multiprocessing
import random
import re

from PIL import Image
from PIL import PngImagePlugin
from PIL import ImageFile
from typing import List, Dict, Tuple, Union, Optional


# FROM PyTorch
import torch

from torch._utils import ExceptionWrapper
from torch.utils.data import Dataset as torchDataset
from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data._utils import pin_memory
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter, _DatasetKind, _SingleProcessDataLoaderIter
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder


# FROM This Project
import backbone.augmentation
import backbone.utils as utils
from backbone.config import Config


from .datasetComponent import DatasetComponent, PREPROCESSING_DICT, AUGMENTATION_DICT, METADATA_DICT_KEYS, EXT_DICT, IS_ITERABLE_DICT
from .datasetConfig import DatasetConfig


# Prevent memory Error in PIL
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: Distribution
class Dataset(torchDataset):
    def __init__(
        self,
        datasetComponentParamList: List,
        mainPath: str,

        batchSize: int,
        type: str,
        range: str,

        isEval: bool,
        isCaching: bool,

        outOrder: List[str],
        filter: Dict,
        augmentation: List[str],
    ):
        '''
        Dataset Settings
        '''
        self.batchSize = batchSize
        self.type = type
        self.range = range

        self.isEval = isEval
        self.isCaching = isCaching

        self.outOrder = outOrder
        self.filter = filter
        self.augmentation = augmentation

        self.mainPath = mainPath

        self.datasetComponentParamList = datasetComponentParamList
        self.datasetComponentObjectList = None
        self._constructDatasetComponents()
        self._datasetComponentObjectListIntegrityTest()

        self.PILImageToTensorFunction = transforms.ToTensor()

        

        '''
        Dataset data
        '''
        self.globalFileList = None

        self._makeGlobalFileList()
        self.mapper = None
        self._makeFileListIndexer()

        self.LabelDataDictList = None

        self.cache = None#{}#FILE_CACHE
        if self.isCaching is True:
            self._caching()


    def _caching(self):

        def _cachingFunc(zippedLists, isMultiProc = True):
            if isMultiProc is True:
                cache = {}
                for filePath, preProc in zippedLists:
                    cache.update({filePath: self._readImage(filePath, preProc)})
                    self.cache_cnt.value += 1
                    print(f"ETA : {(self.cache_MX_LEN - self.cache_cnt.value) / self.cache_cnt.value * (time.perf_counter() - self.cache_startTime):.0f} sec [{self.cache_cnt.value} / {self.cache_MX_LEN}]       ", end='\r')
            else:
                cache = {}
                cnt = 0
                for filePath, preProc in zippedLists:
                    cache.update({filePath: self._readImage(filePath, preProc)})
                    cnt += 1
                    print(f"ETA : {(self.cache_MX_LEN - cnt) / cnt * (time.perf_counter() - self.cache_startTime):.0f} sec [{cnt} / {self.cache_MX_LEN}]       ", end='\r')
            return cache
        
        datasetComponentType = self.datasetComponentObjectList[0].datasetConfig.dataType["dataType"]

        t = time.perf_counter()

        MULTIPROC = 0
        print("Make cache data list...")
        fileListDict = {}
        
        cnt = 0

        for dCO in self.datasetComponentObjectList:
            print(f"Caching Dataset '{dCO.datasetConfig.name}' ... ")
            preProc = dCO.preprocessingList
            for i, dfDict in enumerate(dCO.dataFileList):
                for dfDictKey in dfDict.keys():

                    assert datasetComponentType in ['Image', 'ImageSequence']
                    if datasetComponentType == 'Image':
                        filePath = dfDict[dfDictKey]
                        fileListDict.update({filePath:preProc})
                    elif datasetComponentType == 'ImageSequence':
                        for filePath in dfDict[dfDictKey]:
                            fileListDict.update({filePath:preProc})
                        
                    cnt += 1
                    print(f"Make list... {cnt}   ", end='\r')
        
        fileList = list( fileListDict.keys() )
        preProcList = list( fileListDict[x] for x in fileList )


        print(f"Start Caching with {MULTIPROC} processes...")


        self.cache = {}
        MX_LEN = len(fileList)
        self.cache_MX_LEN = MX_LEN
        self.cache_startTime = time.perf_counter()


        if MULTIPROC == 0:
            self.cache = self._cachingFunc(zip(fileList,preProcList), isMultiProc=False)
        else:
            CHNK_SIZE = MX_LEN // MULTIPROC
            cacheeList = []
            for i in range(MULTIPROC):
                stIdx = i * CHNK_SIZE
                edIdx = (i + 1) * CHNK_SIZE if i != MULTIPROC - 1 else MX_LEN
                cacheeList.append(list(zip(fileList[stIdx: edIdx],preProcList[stIdx: edIdx])))

            with multiprocessing.Manager() as manager:
                self.cache_cnt = manager.Value(int, 0)
                pool = multiprocessing.Pool(processes=MULTIPROC + 1)
                [self.cache.update(x) for x in pool.map(self._cachingFunc, cacheeList)]
                pool.close()
                pool.join()


        print(f"Cached in {time.perf_counter() - t:.1f} sec.                                  ")



    def _makeGlobalFileList(self):
        gFL = [x.dataFileList for x in self.datasetComponentObjectList]
        self.globalFileList = gFL

    def _makeLabelDataDict(self, labelPath):
        """
        Arguments:
            labelPath: (string) text file path of images and annotations
        Return:
            A dict containing:
                file path, label value
                1) string
                2) list of float

        intput : label path
        output : dict (file path(str), label value[list of list])
        """
        labels = []
        LabelDict = {}
        labels_copy = []
        fullPath = ""
        isFirst = True

        upperName = labelPath.split("/")[-2]  # GT
        modeName = labelPath.split("/")[-1]  # label_train

        f = open(labelPath, "r")
        lines = f.readlines()

        for line in lines:
            line = line.rstrip()
            if line.startswith("#"):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    LabelDict.update({fullPath: labels_copy})
                    labels.clear()
                path = line[2:]
                fullPath = labelPath.replace(upperName + "/" + modeName, "") + path
            else:
                line = line.split(" ")
                label = [float(x) for x in line]
                labels.append(label)

        labels_copy = labels.copy()
        LabelDict.update({fullPath: labels_copy})

        return LabelDict

    def _makeGlobalLabelDataDictList(self):
        """
        # Commponent Count 후 각 Commponent에 접근의 Label에 접근
        # output : list [dict, dict, ..., dict] # dict{filepath : values}
        # firstIndex -> Component에 대한 type는 동일하게만 들어옴 (txt)
        """
        if self.datasetComponentObjectList[0].datasetConfig.dataType["dataType"] == "Text":
            firstIndex = 0
            LabelDataPathList = []
            componentList = len(self.datasetComponentObjectList)

            LabelDataPathList = list(
                map(
                    lambda x: Config.param.data.path.datasetPath + self.globalFileList[x][firstIndex]["labelFilePath"],
                    range(componentList),
                )
            )
            LabelDataDictList = list(map(lambda x: self._makeLabelDataDict(x), LabelDataPathList))

            self.LabelDataDictList = LabelDataDictList
        else:
            self.LabelDataDictList = {}

    def _makeFileListIndexer(self):

        mapper = []

        dcCount = 0  # datasetComponent의 갯수
        dcLenFilesList = []  # datasetComponent 각각이 가지고 있는 총 파일의 갯 수 리스트
        dcLenSeqsList = []  # datasetComponent가 가지고 있는 각각의 시퀀스 갯수 리스트
        SeqLenFilesList = []  # 각 dc의 시퀀스 밑에 몇개의 파일이 있는지 리스트 (2-d)


        dataTypeList = [self.datasetComponentObjectList[0].outDict[_key]['type'] for _key in self.outOrder]

        for dcL in self.globalFileList:
            dcCount += 1
            dcLenFilesList.append(len(dcL))

        ####################################################################
        # EVAL MODE   :    len -> sum(len(datasetComponents))
        ####################################################################
        if self.isEval is True:

            for i in range(self.__len__()):

                for j in range(len(dcLenFilesList)):
                    if i < sum(dcLenFilesList[: j + 1]):
                        break
                dcIdx = j

                fileIdx = i - sum(dcLenFilesList[:j])

                mapper.append([dcIdx, fileIdx])

        ####################################################################
        # TRAIN MODE   :    len -> max(len(datasetComponents)) * N
        ####################################################################
        else:

            for i in range(self.__len__()):

                dcIdx = i % dcCount

                fileIdx = i // dcCount % dcLenFilesList[dcIdx]

                mapper.append([dcIdx, fileIdx]) 

        '''

        ####################################################################
        # SEQUENCE
        ####################################################################
        if self.datasetComponentObjectList[0].datasetConfig.dataType["dataType"] == "ImageSequence":

            for dcL in self.globalFileList:
                dcCount += 1
                dcLenSeqsList.append(len(dcL))

                tempLenFiles = 0
                tempSeqLenFiles = []
                for seqL in dcL:
                    ln = len(seqL[list(seqL.keys())[0]]) - (self.datasetComponentObjectList[0].sequenceLength - 1)
                    tempSeqLenFiles.append(ln)
                    tempLenFiles += ln
                SeqLenFilesList.append(tempSeqLenFiles)
                dcLenFilesList.append(tempLenFiles)

            ####################################################################
            # EVAL MODE   :    len -> sum(len(datasetComponents))
            ####################################################################
            if self.isEval is True:

                ######## SEQ         [     [dc [seq [file, file] ],[seq] ],[dc]    ]
                for i in range(self.__len__()):

                    for j in range(len(dcLenFilesList)):
                        if i < sum(dcLenFilesList[: j + 1]):
                            break
                    dcIdx = j

                    tMapperElem = []

                    tmp = i // dcCount % dcLenFilesList[dcIdx]
                    for j in range(len(SeqLenFilesList[dcIdx]) - 1):
                        if tmp < sum(SeqLenFilesList[dcIdx][: j + 1]):
                            break
                    seqIdx = j

                    fileIdx = tmp - sum(SeqLenFilesList[dcIdx][:j])

                    for s in range(self.datasetComponentObjectList[dcIdx].sequenceLength):
                        tMapperElem.append([dcIdx, seqIdx, fileIdx + s])

                    mapper.append(tMapperElem)

            ####################################################################
            # TRAIN MODE   :    len -> max(len(datasetComponents)) * 2
            ####################################################################
            else:

                for i in range(self.__len__()):

                    tMapperElem = []

                    dcIdx = i % dcCount

                    tmp = i // dcCount % dcLenFilesList[dcIdx]
                    for j in range(len(SeqLenFilesList[dcIdx])):
                        if tmp < sum(SeqLenFilesList[dcIdx][: j + 1]):
                            break
                    seqIdx = j

                    fileIdx = tmp - sum(SeqLenFilesList[dcIdx][:j])

                    for s in range(self.datasetComponentObjectList[dcIdx].sequenceLength):
                        tMapperElem.append([dcIdx, seqIdx, fileIdx + s])

                    mapper.append(tMapperElem)

        ####################################################################
        # Non-SEQUENCE
        ####################################################################
        else:

            for dcL in self.globalFileList:
                dcCount += 1
                dcLenFilesList.append(len(dcL))

            ####################################################################
            # EVAL MODE   :    len -> sum(len(datasetComponents))
            ####################################################################
            if self.isEval is True:

                for i in range(self.__len__()):

                    for j in range(len(dcLenFilesList)):
                        if i < sum(dcLenFilesList[: j + 1]):
                            break
                    dcIdx = j

                    fileIdx = i - sum(dcLenFilesList[:j])

                    mapper.append([dcIdx, fileIdx])

            ####################################################################
            # TRAIN MODE   :    len -> max(len(datasetComponents)) * 2
            ####################################################################
            else:

                for i in range(self.__len__()):

                    dcIdx = i % dcCount

                    fileIdx = i // dcCount % dcLenFilesList[dcIdx]

                    mapper.append([dcIdx, fileIdx])
        '''

        self.mapper = mapper
        # print(self.mapper)
        # print("str(len(self.mapper))" + str(len(self.mapper)))

    def _popItemInGlobalFileListByIndex(self, index):

        componentIndex, componentFileListIndex = self.mapper[index]

        rstDict = {}
        for _i, _key in enumerate(self.outOrder):
            filePath = self.globalFileList[componentIndex][componentFileListIndex][_i]
            rstDict[_key] = filePath

        return rstDict

        '''
        datasetComponentType = self.datasetComponentObjectList[0].datasetConfig.dataType["dataType"]


        if datasetComponentType != "ImageSequence":

            componentIndex, componentFileListIndex = self.mapper[index]

            dFP = self.globalFileList[componentIndex][componentFileListIndex]["dataFilePath"]
            dFP = (
                dFP
                if isinstance(dFP, str)
                else [x for x in dFP]
            )

            lFP = (
                self.globalFileList[componentIndex][componentFileListIndex]["labelFilePath"]
                if self.globalFileList[componentIndex][componentFileListIndex]["labelFilePath"] is not None
                else None
            )
            lFP = (
                (
                    lFP
                    if isinstance(lFP, str)
                    else [x for x in lFP]
                )
                if lFP is not None
                else None
            )

        else:

            dFPList = []
            lFPList = []

            for componentIndex, seqIdx, fileIdx in self.mapper[index]:

                # print(componentIndex, seqIdx, fileIdx)

                dFP = self.globalFileList[componentIndex][seqIdx]["dataFilePath"][fileIdx]
                dFP = (
                    dFP
                    if isinstance(dFP, str)
                    else [x for x in dFP]
                )

                lFP = (
                    self.globalFileList[componentIndex][seqIdx]["labelFilePath"][fileIdx]
                    if self.globalFileList[componentIndex][seqIdx]["labelFilePath"] is not None
                    else None
                )
                lFP = (
                    (
                        lFP
                        if isinstance(lFP, str)
                        else [x for x in lFP]
                    )
                    if lFP is not None
                    else None
                )

                dFPList.append(dFP)
                lFPList.append(lFP)

            dFP = dFPList
            lFP = lFPList
        

        return {"dataFilePath": dFP, "labelFilePath": lFP}
        '''



    def _constructDatasetComponents(self):
        self.datasetComponentObjectList = [DatasetComponent(*x) for x in self.datasetComponentParamList]

    def _datasetComponentObjectListIntegrityTest(self):
        # Test All dataComponents

        # out keys must be dataset outOrder's superset
        datasetComponentOutKeysList = [x.outDict.keys() for x in self.datasetComponentObjectList]
        for datasetComponentOutKey in datasetComponentOutKeysList:
            assert set(datasetComponentOutKey).issuperset(set(self.outOrder)) is True, "dataLoader.py :: datasetComponent out keys must be dataloader outOrder's superset"

        # same outdict keys have same type
        for key in self.outOrder:
            assert [self.datasetComponentObjectList[0].outDict[key]["type"]] * len(self.datasetComponentObjectList) \
                 == [x.outDict[key]["type"] for x in self.datasetComponentObjectList] \
                     , "dataLoader.py :: all datasetComponent's same outdict keys must have same type"

    def _calculateMemorySizePerTensor(
        self, dtype: torch.dtype, expectedShape: List[int]
        ):  # WARN: EXCEPT BATCH SIIZEEEEEE!!!!!!!!!!!!!!!!!!!!!!!
        sizeOfOneElementDict = {
            torch.float32: 4,
            torch.float: 4,
            torch.float64: 8,
            torch.double: 8,
            torch.float16: 2,
            torch.half: 2,
            torch.uint8: 1,
            torch.int8: 1,
            torch.int16: 2,
            torch.short: 2,
            torch.int32: 4,
            torch.int: 4,
            torch.int64: 8,
            torch.long: 8,
            torch.bool: 2,
        }
        elemSize = sizeOfOneElementDict(dtype)

        totalSize = reduce(lambda x, y: x * y, expectedShape) * elemSize

        return totalSize




    def _torchvisionPreprocessing(self, pilImage, preProc):

        x = pilImage
        for preProcFunc in preProc:
            x = preProcFunc(x)

        return x



    # DATA AUGMENTATION ON CPU - Multiprocessing with Forked Worker processes -
    # Before toTensor() Augmentation
    # CROP OPERATION usually be here
    # Make Sure that ALL output Size (C, H, W) are same.
    def _dataAugmentation(self, dataList, augmentations: List[str]):

        def _applyAugmentationFunction(tnsr, augmentationFuncStr: str):

            return _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr)

        def _dataModulation(dataList):
            rstList = []
            seqLenList = []
            #make non-seq data to 1-len seq data
            for _data in dataList:
                if not isinstance(_data, list):
                    rstList.append([_data])
                    seqLenList.append(0)
                else:
                    rstList.append(_data)
                    seqLenList.append(len(_data))
            
            #copy 1-len seq data by max length
            maxSeqLen = max(seqLenList)
            for _i, _data in enumerate(rstList):
                if len(_data) < maxSeqLen:
                    rstList[_i] = _data + [_data[-1]] * (maxSeqLen - len(_data))

            #M of N-seq data to N of M-seq data (zip)
            # case M = 2)
            # If input is a list of two tensors -> make it to list of list of two tensors
            # The standard Input type is list of list of two tensors -> [  [data_1, label_1], ... , [data_N, label_N]  ]
            return list(zip(*rstList)), seqLenList

        
        def _dataDemodulation(dataList, seqLenList):
            
            dataList = list(zip(*dataList))

            rstList = []

            for _data, _seqLen in zip(dataList, seqLenList):
                if _seqLen == 0:
                    rstList.append(_data[0])
                else:
                    rstList.append(_data[:_seqLen])

            return rstList







        dataList, seqLenList = _dataModulation(dataList)

        augedXList = []

        for x in dataList:
            for augmentation in augmentations:
                #print(f"RUN PHASE 3-4-1-{augmentation}")
                x = _applyAugmentationFunction(x, augmentation)
                # print(augmentation)
                if augmentation == "toTensor()":
                    break
            augedXList.append(x)

        # TRANSPOSE
        #augedXList = list(map(list, zip(*augedXList)))
        augedXList = _dataDemodulation(augedXList, seqLenList)

        return augedXList

    def _tensorize(self, dataList, valueRange, tensorType):
        
        def _tensorizeOneTensor(data, multiplier, offset, tensorType):
            return (data * multiplier + offset).to(tensorType)

        TENSOR_TYPE_DICT = {
            'float': torch.float32,
            'float32': torch.float32,
            
            'double': torch.float64,
            'float64': torch.float64,

            'half': torch.half,
            'float16': torch.float16,

            'int': torch.int,

            'uint8': torch.uint8,

            'int8': torch.int8,

            'short': torch.int16,
            'int16': torch.int16,

            'int': torch.int32,
            'int32': torch.int32,

            'long': torch.int64,
            'int64': torch.int64,
        }

        assert tensorType in TENSOR_TYPE_DICT.keys(), f"dataset.py :: Type must be one of these list: {[x for x in TENSOR_TYPE_DICT.keys()]}"

        tensorType = TENSOR_TYPE_DICT[tensorType.replace(' ','').lower()]
        rangeMin, rangeMax = [float(x) if tensorType.is_floating_point is True else int(x) for x in valueRange.replace(' ','').split('~')]
        multiplier = rangeMax - rangeMin
        offset = rangeMin

        rstList = []
        for _data in dataList:
            # N C D(seq) H W
            if isinstance(_data, list):
                t_rst = torch.stack([_tensorizeOneTensor(x, multiplier, offset, tensorType) for x in _data]).permute(1,0,2,3)
            # N C H W
            else:
                t_rst = _tensorizeOneTensor(_data, multiplier, offset, tensorType)

            rstList.append(t_rst)

        return rstList
            
    
    def _setTensorValueRange(self, tnsr, valueRangeType: str):

        if valueRangeType == '-1~1':
            tnsr = tnsr * 2 - 1
        elif valueRangeType == '0~255':
            tnsr = np.round(tnsr * 255).type(torch.FloatTensor)

        return tnsr






    def _methodLoadLabel(self, componentIndex, filePath, preProc):
        labelValueList = self.LabelDataDictList[componentIndex][filePath]  # label values(list(list))

        for preProcFunc in preProc:
            LabelList = preProcFunc(labelValueList)  # preProc(labelValueList)

        return LabelList

    def _readImage(self, FilePath, preProc):

        def _loadPILImagesFromHDD(filePath):
            return Image.open(filePath)

        def _saveTensorsToHDD(tnsr, filePath):
            print(f"Write Tensor to {filePath}.npy...")
            utils.saveTensorToNPY(tnsr, filePath)

        def _loadNPArrayFromHDD(filePath):
            return utils.loadNPY(filePath)

        def _PIL2Tensor(pilImage):
            return self.PILImageToTensorFunction(pilImage)

        def _NPYMaker(filePath, preProc):
            PILImage = _loadPILImagesFromHDD(filePath)  # PIL
            PPedPILImage = self._torchvisionPreprocessing(PILImage, preProc)
            rstTensor = _PIL2Tensor(PPedPILImage)

            _saveTensorsToHDD(rstTensor, filePath)

            return rstTensor

        def _methodNPYExists(filePath):
            npa = _loadNPArrayFromHDD(filePath)

            return npa

        def _methodNPYNotExists(filePath, preProc):
            PILImage = _loadPILImagesFromHDD(filePath)
            PPedPILImage = self._torchvisionPreprocessing(PILImage, preProc)

            return PPedPILImage
        '''
        if os.path.isfile(FilePath + ".npy") is True:
            try:
                rst = self._methodNPYExists(
                    FilePath + ".npy"
                )  # if .npy Exists, load preprocessed .npy File as Pytorch Tensor -> load to GPU directly -> Augmentation on GPU -> return
            except ValueError:
                if self.makePreprocessedFile is True:
                    rst = self._NPYMaker(
                        FilePath, preProc
                    )  # if .npy doesn't Exists and self.makePreprocessedFile is True, make .npy file and augmentating tensor and return
                else:
                    rst = self._methodNPYNotExists(
                        FilePath, preProc
                    )  # if .npy doesn't Exists, load Image File as PIL Image -> Preprocess PIL Image on CPU -> convert to Tensor -> load to GPU -> Augmentation on GPU -> return
        else:
            if self.makePreprocessedFile is True:
                rst = self._NPYMaker(
                    FilePath, preProc
                )  # if .npy doesn't Exists and self.makePreprocessedFile is True, make .npy file and augmentating tensor and return
            else:
                rst = self._methodNPYNotExists(
                    FilePath, preProc
                )  # if .npy doesn't Exists, load Image File as PIL Image -> Preprocess PIL Image on CPU -> convert to Tensor -> load to GPU -> Augmentation on GPU -> return
        '''
        rst = _methodNPYNotExists(FilePath, preProc)
        return rst


    def _readItem(self, Type, FilePath, index, preProc):
        # data information defined in config
            
        ###################################################################################
        # CASE TEXT
        ###################################################################################
        if Type == "Text":
            componentIndex = index % len(self.datasetComponentObjectList)
            popped = self._popItemInGlobalFileListByIndex(index)
            dataFilePath = popped["dataFilePath"]
            rst = self._methodLoadLabel(componentIndex, dataFilePath, preProc)

        ###################################################################################
        # CASE IMAGE
        ###################################################################################
        elif Type == "Image":
            rst = self._readImage(FilePath, preProc) if self.isCaching is False else self.cache[FilePath]
            
        ###################################################################################
        # CASE IMAGESEQUENCE
        ###################################################################################
        elif Type == "ImageSequence":

            seqFileList = FilePath
            rstList = []

            for seqFilePath in seqFileList:
                rstList.append(self._readImage(seqFilePath, preProc)  if self.isCaching is False else self.cache[seqFilePath])

            rst = rstList

        return rst

    def __getitem__(self, index):
        #print(f"RUN PHASE 3-1")
        # popping File Path at GFL(self.globalFileList) by index
        popped = self._popItemInGlobalFileListByIndex(index)

        componentIndex = index % len(self.datasetComponentObjectList)
        preProc = self.datasetComponentObjectList[componentIndex].preprocessingList

        t_dataList = []
        rstDict = {}

        ################################################################################### 
        # DATA FLOW :::>>>
        # Read Data as Non-tensor Type(PIL...etc) -> Augmentation(PIL...) -> to Tensor (dimension, type, range)
        ###################################################################################


        ###################################################################################
        # Read DATA & LABEL
        ###################################################################################
        for _key in self.outOrder:
            t_dataList.append(self._readItem(self.datasetComponentObjectList[0].outDict[_key]['type'], self.mainPath + popped[_key], index, preProc))


        ###################################################################################
        # Data Augmentation
        ###################################################################################
        t_dataList = self._dataAugmentation(t_dataList, self.augmentation)


        ###################################################################################
        # Tensorize
        ###################################################################################
        t_dataList = self._tensorize(t_dataList, self.range, self.type)


        for _i, _key in enumerate(self.outOrder):
            rstDict[_key] = t_dataList[_i]

        '''
        dataFilePath = popped["dataFilePath"]
        labelFilePath = popped["labelFilePath"]
        #print(f"RUN PHASE 3-2")
        componentIndex = index % len(self.datasetComponentObjectList)

        preProc = self.datasetComponentObjectList[componentIndex].preprocessingList
        dataType = self.datasetComponentObjectList[componentIndex].datasetConfig.dataType
        labelType = self.datasetComponentObjectList[componentIndex].datasetConfig.labelType

        rstDict = {}

        filePathList = [dataFilePath] if labelFilePath is None else [dataFilePath, labelFilePath]
        typeList = [dataType] if labelType == {} else [dataType, labelType]
        #print(f"RUN PHASE 3-3")
        ###################################################################################
        # ADD DATA & LABEL
        ###################################################################################

        for Type, FilePath in zip(typeList, filePathList):
            rstDict[Type["dataName"]] = self._readItem(Type, FilePath, index, preProc)
        #print(f"RUN PHASE 3-4")
        ###################################################################################
        # Data Augmentation
        ###################################################################################--------------------------------------------------------------

        if isinstance(rstDict[dataType["dataName"]], list):  # TODO:
            if labelType != {}:  # if label Exists
                rstDict[dataType["dataName"]], rstDict[labelType["dataName"]] = list(
                    zip(
                        *list(
                            self._dataAugmentation(x, self.augmentation)
                            for x in list(zip(rstDict[dataType["dataName"]], rstDict[labelType["dataName"]]))
                        )
                    )
                )
                rstDict[labelType["dataName"]] = list(
                    self._setTensorValueRange(x, self.valueRangeType) for x in rstDict[labelType["dataName"]]
                )
            else:  # if label not Exists
                rstDict[dataType["dataName"]] = list(
                    self._dataAugmentation(x, self.augmentation)[0] for x in list(zip(rstDict[dataType["dataName"]]))
                )
            rstDict[dataType["dataName"]] = list(
                self._setTensorValueRange(x, self.valueRangeType) for x in rstDict[dataType["dataName"]]
            )
        else:
            if labelType != {}:  # if label Exists
                #print(f"RUN PHASE 3-4-1")
                rstDict[dataType["dataName"]], rstDict[labelType["dataName"]] = self._dataAugmentation(
                    [rstDict[dataType["dataName"]], rstDict[labelType["dataName"]]], self.augmentation
                )
                #print(f"RUN PHASE 3-4-2")
                rstDict[labelType["dataName"]] = self._setTensorValueRange(
                    rstDict[labelType["dataName"]], self.valueRangeType
                )
            else:  # if label not Exists
                rstDict[dataType["dataName"]] = self._dataAugmentation([rstDict[dataType["dataName"]]], self.augmentation)[0]
            #print(f"RUN PHASE 3-4-3")
            rstDict[dataType["dataName"]] = self._setTensorValueRange(rstDict[dataType["dataName"]], self.valueRangeType)

        # print(rstDict)
        #print(f"RUN PHASE 3-5")
        ###################################################################################
        # Data Demension Align
        ###################################################################################

        types = [dataType, labelType]

        for typ in types:
        
            #if isinstance(rstDict[typ["dataName"]], list):

            if typ != {}:
                if typ["dataType"] == "Text":
                    pass
                elif typ["dataType"] == "Image":
                    assert len(rstDict[typ["dataName"]].size()) == 4
                    rstDict[typ["dataName"]] = rstDict[typ["dataName"]].squeeze(0)
                elif typ["dataType"] == "ImageSequence":
                    assert len(rstDict[typ["dataName"]][0].size()) == 4
        '''
        #print(f"RUN PHASE 3-6")
        return rstDict

    def __len__(self):

        if self.isEval is True:
            cnt = 0
            for dcL in list(map(len, self.datasetComponentObjectList)):
                cnt += dcL

        else:
            cnt = max(list(map(len, self.datasetComponentObjectList))) * len(self.datasetComponentObjectList)

        return cnt




def _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr: str):

    def _typing(_x):
        #None
        if _x.lower() == 'none':
            _value = None
        #String
        elif (_x[1:] if _x[0] == "-" else _x).replace(".", "", 1).isdigit() is False:
            _value = str(_x)
        else:
            #int
            if _x.find(".") == -1:
                _value = int(_x)
            #float
            else: 
                _value = float(_x)
        return _value

    assert (
        augmentationFuncStr.split("(")[0] in AUGMENTATION_DICT.keys()
    ), "dataLoader/dataset.py :: invalid Augmentation Function!! check param.yaml."

    # is it randomized augmentation?
    if (augmentationFuncStr.rfind(':') >= 0  
        and augmentationFuncStr.rfind(':') > augmentationFuncStr.rfind(')') 
        and augmentationFuncStr.rfind(':') < len(augmentationFuncStr)):
        probability = float(augmentationFuncStr.split(':')[-1])
        augmentationFuncStr = augmentationFuncStr[:augmentationFuncStr.rfind(')')]
    else:
        probability = 1.0

    # over the wall of probability!
    if probability >= random.random():

        # make augfunc list
        augmentationFuncStrList = [x.strip() for x in augmentationFuncStr.split('->')]

        for eachAugmentationFuncStr in augmentationFuncStrList:
            augFunc = AUGMENTATION_DICT[eachAugmentationFuncStr.split("(")[0]]
            argStr = eachAugmentationFuncStr.split("(")[1].replace(")","")
            
            args = []
            kwargs = {}

            #Extract lists
            listsInArgStr = []

            listsInArgStr = [x[1:-1] for x in re.findall('(\[(?:\[??[^\[]*?\]))', argStr)]
            listsInArgStr = [] if listsInArgStr is None else list(listsInArgStr)

            for _i, _listStr in enumerate(listsInArgStr):
                argStr = argStr.replace(f"[{_listStr}]", f"!@#$%LIST@{_i}%$#@!", 1)

            for x in list(filter(lambda y: y != "", argStr.replace(" ", "").split(","))):

                if x.find("=") == -1:
                    isKwargs = False
                    _x = x
                else:
                    isKwargs = True
                    _key = str(x.replace(' ','').split('=')[0])
                    _x = str(x.replace(' ','').split('=')[1])

                #list
                if _x.startswith('!@#$%') and _x.endswith('%$#@!'):
                    _x = listsInArgStr[int(_x[5:-5].split("@")[1])]
                    _value = [_typing(_y) for _y in list(filter(lambda y: y != "", _x.replace(" ", "").split(",")))]
                else:
                    _value = _typing(_x)

                if isKwargs is False:
                    args.append(_value)
                else:
                    kwargs[_key] = _value


            tnsr = augFunc(tnsr, *args, **kwargs)

    return tnsr