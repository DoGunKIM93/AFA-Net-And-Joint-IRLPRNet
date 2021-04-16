"""
data_loader.py
"""


# FROM Python LIBRARY
import os
import random
import numpy as np
import inspect
import itertools
import glob
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
import backbone.preprocessing
import backbone.augmentation
import backbone.utils as utils
from backbone.config import Config


# Prevent memory Error in PIL
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Read Preprocessings from backbone.preprocessing and Augmentations from backbone.augmentation automatically
PREPROCESSING_DICT = dict(
    x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith("__")) and (inspect.isclass(x[1]))
)
AUGMENTATION_DICT = dict(
    x for x in inspect.getmembers(backbone.augmentation) if (not x[0].startswith("_")) and (inspect.isfunction(x[1]))
)


class DatasetConfig:
    def __init__(
        self,
        name: str,
        origin: Optional[str] = None,
        dataType: Optional[Dict[str, str]] = None,
        labelType: Optional[Dict[str, str]] = None,
        availableMode: Optional[List[str]] = None,
        classes: Optional[List[str]] = None,
        preprocessings: Optional[List[str]] = None,
        useDatasetConfig: bool = True,
    ):

        self.name = name

        self.origin = origin
        self.dataType = dataType
        self.labelType = labelType
        self.availableMode = availableMode
        self.classes = classes
        self.preprocessings = preprocessings

        self.useDatasetConfig = useDatasetConfig

        if self.useDatasetConfig is True:
            self._getDatasetConfig()

    def _splitDataType(self, dataType: Dict[str, str]):
        # return form of {'dataName': 'LR', dataType': 'text', 'tensorType': 'int'}

        dataTypeList = ["Text", "Image", "ImageSequence"]
        tensorTypeList = ["Float", "Int", "Long", "Double"]

        if dataType is not None:
            for key in dataType:
                assert (
                    dataType[key].split("-")[0] in dataTypeList
                ), f"data_loader.py :: dataset config {self.name} has invalid dataType."
                assert (
                    dataType[key].split("-")[1] in tensorTypeList
                ), f"data_loader.py :: dataset config {self.name} has invalid tensorType."
            return dict(zip(["dataName", "dataType", "tensorType"], [key] + dataType[key].split("-")))
        else:
            return {}

    def _getDatasetConfig(self):

        yamlData = Config.datasetConfigDict[f"{self.name}"]

        self.origin = str(yamlData["origin"])

        self.dataType = self._splitDataType(yamlData["dataType"])

        self.labelType = self._splitDataType(yamlData["labelType"])

        self.availableMode = list(map(str, yamlData["availableMode"]))
        self.classes = list(map(str, yamlData["classes"]))
        self.preprocessings = list(map(str, yamlData["preprocessings"])) if yamlData["preprocessings"] is not None else []


# Representation of single dataset property
class DatasetComponent:
    def __init__(self, name, mode, classParameter, labelClassName, sequenceLength):

        self.name = name

        self.datasetConfig = None
        self._getDatasetConfigByComponentName()

        self.mode = mode
        self.classParameter = classParameter
        self.labelClassName = labelClassName
        self.sequenceLength = sequenceLength

        self.dataFileList = None
        self.labelFileList = None
        self._getDataFileList()

        self.preprocessingList = None
        self._makePreprocessingList()

    def _getDatasetConfigByComponentName(self):
        self.datasetConfig = DatasetConfig(Config.paramDict["data"]["datasetComponent"][self.name]["dataConfig"])

    def _getDataFileList(self):

        # get origin path
        mainPath = Config.param.data.path.datasetPath
        path = f"{self.datasetConfig.origin}/{self.mode}/"

        datasetConfigClasses = self.datasetConfig.classes
        # dynamic construction of class path based on defined classes
        for i in range(len(datasetConfigClasses)):
            classPathList = (
                list(
                    itertools.chain.from_iterable(
                        list(
                            map(
                                lambda y: list(
                                    map(
                                        lambda x: str(x) if type(x) is int else x + "/" + str(y) if type(y) is int else y,
                                        classPathList,
                                    )
                                ),
                                self.classParameter[datasetConfigClasses[i]],
                            )
                        )
                    )
                )
                if i is not 0
                else self.classParameter[datasetConfigClasses[i]]
            )
            # Sorry for who read this

        # add origin path in front of all elements of class path list
        pathList = list(map(lambda x: path + x, classPathList))

        if self.datasetConfig.dataType["dataType"] == "Text":
            # construct all of readable file lists in class path lists
            dataFileLists = list(
                map(
                    lambda x: list(
                        filter(
                            lambda x: (x.endswith(".txt")),
                            list(map(lambda y: mainPath + x + "/" + y, sorted(os.listdir(mainPath + x)))),
                        )
                    ),
                    pathList,
                )
            )

        elif self.datasetConfig.dataType["dataType"] == "Image":
            # construct all of readable file lists in class path lists
            dataFileLists = list(
                map(
                    lambda x: list(
                        filter(
                            lambda x: (
                                x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp")
                            ),
                            list(map(lambda y: mainPath + x + "/" + y, sorted(os.listdir(mainPath + x)))),
                        )
                    ),
                    pathList,
                )
            )

        elif self.datasetConfig.dataType["dataType"] == "ImageSequence":
            # construct all of sequence folders in class path lists
            dataFileLists = list(map(lambda x: list(map(lambda y: x + "/" + y, sorted(os.listdir(mainPath + x)))), pathList))
            dataFileLists = [
                list(
                    map(
                        lambda x: [
                            mainPath + x + "/" + z
                            for z in list(
                                filter(
                                    lambda y: (
                                        y.endswith(".png") or y.endswith(".jpg") or y.endswith(".jpeg") or y.endswith(".bmp")
                                    ),
                                    sorted(os.listdir(mainPath + x)),
                                )
                            )
                        ],
                        xlist,
                    )
                )
                for xlist in dataFileLists
            ]

        # if label Exists
        if len(self.datasetConfig.labelType) > 0:

            labelPath = f"{path}{self.labelClassName}/"

            if self.datasetConfig.labelType["dataType"] == "Text":
                # construct all of readable file lists in class path lists
                labelFiles = sorted(list(filter(lambda x: (x.endswith(".txt")), os.listdir(mainPath + labelPath))))
                # add origin path in front of all elements of label file list
                labelFiles = list(map(lambda x: mainPath + labelPath + x, labelFiles))

            elif self.datasetConfig.labelType["dataType"] == "Image":
                # construct all of readable file lists in class path lists
                labelFiles = sorted(
                    list(
                        filter(
                            lambda x: (
                                x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp")
                            ),
                            os.listdir(mainPath + labelPath),
                        )
                    )
                )
                # add origin path in front of all elements of label file list
                labelFiles = list(map(lambda x: mainPath + labelPath + x, labelFiles))

            elif self.datasetConfig.labelType["dataType"] == "ImageSequence":
                # construct all of sequence folders in class path lists
                labelFiles = sorted(os.listdir(mainPath + labelPath))
                labelFiles = list(
                    map(
                        lambda x: [
                            mainPath + labelPath + x + "/" + z
                            for z in list(
                                filter(
                                    lambda y: (
                                        y.endswith(".png") or y.endswith(".jpg") or y.endswith(".jpeg") or y.endswith(".bmp")
                                    ),
                                    sorted(os.listdir(mainPath + labelPath + x)),
                                )
                            )
                        ],
                        labelFiles,
                    )
                )

                # print(labelFiles)

            if len(labelFiles) == 1:
                # Label case ex (.txt file)
                labelFiles = labelFiles * len(dataFileLists[0])
            else:
                # Label case for same count of image files (GT)
                assert [len(dataFileLists[0])] * len(dataFileLists) == list(
                    map(len, dataFileLists)
                ), f"data_loader.py :: ERROR! dataset {self.name} has NOT same count of data files for each classes."
                assert [len(labelFiles)] * len(dataFileLists) == list(
                    map(len, dataFileLists)
                ), f"data_loader.py :: ERROR! label and data files should be had same count. (dataset {self.name})"

        else:
            labelFiles = [None] * len(dataFileLists[0])

        labelFiles = [labelFiles] * len(dataFileLists)

        dataFileDictList = list(
            map(
                lambda x: dict(zip(["dataFilePath", "labelFilePath"], x)),
                list(zip(itertools.chain.from_iterable(dataFileLists), itertools.chain.from_iterable(labelFiles))),
            )
        )

        # set dataFileList without main path
        self.dataFileList = dataFileDictList

    def _makePreprocessingList(self):
        self.preprocessingList = list(
            map(
                (
                    lambda x: PREPROCESSING_DICT[x.split("(")[0]](
                        *list(filter(lambda y: y != "", x.split("(")[1][:-1].replace(" ", "").split(",")))
                    )
                ),
                self.datasetConfig.preprocessings,
            )
        )

    def _getSeqDataLen(self):
        cnt = 0
        for seqFileDict in self.dataFileList:
            # for key in seqFileDict:
            cnt += len(seqFileDict[list(seqFileDict.keys())[0]]) - (self.sequenceLength - 1)
        # print(cnt)
        return cnt

    def __len__(self):
        return (
            len(self.dataFileList) if self.datasetConfig.dataType["dataType"] != "ImageSequence" else self._getSeqDataLen()
        )


# TODO: Distribution
class Dataset(torchDataset):
    def __init__(
        self,
        datasetComponentParamList: List,
        batchSize: int,
        valueRangeType: str,
        isEval: bool,
        augmentation: List[str],
        numWorkers: int,
        makePreprocessedFile: bool,
        isCaching: bool,
        forceImageDataType: bool,
    ):
        '''
        Dataset Settings
        '''
        self.datasetComponentParamList = datasetComponentParamList

        self.datasetComponentObjectList = None
        self._constructDatasetComponents()
        self._datasetComponentObjectListIntegrityTest()

        self.batchSize = batchSize
        self.valueRangeType = valueRangeType
        self.isEval = isEval
        self.augmentation = augmentation
        self.numWorkers = numWorkers

        self.makePreprocessedFile = makePreprocessedFile

        self.PILImageToTensorFunction = transforms.ToTensor()

        self.isCaching = isCaching
        self.forceImageDataType = forceImageDataType

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


    def _cachingFunc(self, zippedLists, isMultiProc = True):
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

    def _caching(self):
        
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

        self.mapper = mapper
        # print(self.mapper)
        # print("str(len(self.mapper))" + str(len(self.mapper)))

    def _popItemInGlobalFileListByIndex(self, index):

        datasetComponentType = self.datasetComponentObjectList[0].datasetConfig.dataType["dataType"]

        if datasetComponentType != "ImageSequence":

            componentIndex, componentFileListIndex = self.mapper[index]
            # print(componentIndex, componentFileListIndex)

            # componentFileListIndex = ( index // len(datasetComponentLengthList) ) % datasetComponentLengthList[componentIndex]

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

                """
                datasetComponentSequenceLengthList = []
                for dComponent in range(len(self.globalFileList)):
                    tdC = []
                    for dSeq in range(len(self.globalFileList[dComponent])):
                        tdC.append(len(self.globalFileList[dComponent][dSeq]))
                    datasetComponentSequenceLengthList.append(tdC)

                sequenceListIndex = ( index // len(datasetComponentLengthList) ) % datasetComponentLengthList[componentIndex]
                sequenceFileIndex =
                """

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

    def _constructDatasetComponents(self):
        self.datasetComponentObjectList = [DatasetComponent(*x) for x in self.datasetComponentParamList]

    def _datasetComponentObjectListIntegrityTest(self):
        # Test All dataComponents have same
        # dataType
        # TensorType
        # name
        datasetComponentdataTypeList = [x.datasetConfig.dataType for x in self.datasetComponentObjectList]
        assert [datasetComponentdataTypeList[0]] * len(
            datasetComponentdataTypeList
        ) == datasetComponentdataTypeList, "data_loader.py :: All datasets in dataloader must have same dataType."

        datasetComponentlabelTypeList = [x.datasetConfig.labelType for x in self.datasetComponentObjectList]
        assert [datasetComponentlabelTypeList[0]] * len(
            datasetComponentlabelTypeList
        ) == datasetComponentlabelTypeList, "data_loader.py :: All datasets in dataloader must have same tensorType."

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

    def _applyAugmentationFunction(self, tnsr, augmentationFuncStr: str):

        return _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr)

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
                #print(f"RUN PHASE 3-4-1-{augmentation}")
                x = self._applyAugmentationFunction(x, augmentation)
                # print(augmentation)
                if augmentation == "toTensor()":
                    break
            augedXList.append(x)

        # TRANSPOSE
        augedXList = list(map(list, zip(*augedXList)))

        augedXTensorList = [torch.stack(x) for x in augedXList]

        return augedXTensorList

    def _setTensorValueRange(self, tnsr, valueRangeType: str):

        if valueRangeType == '-1~1':
            tnsr = tnsr * 2 - 1
        elif valueRangeType == '0~255':
            tnsr = np.round(tnsr * 255).type(torch.FloatTensor)

        return tnsr

    def _loadPILImagesFromHDD(self, filePath):
        return Image.open(filePath)

    def _saveTensorsToHDD(self, tnsr, filePath):
        print(f"Write Tensor to {filePath}.npy...")
        utils.saveTensorToNPY(tnsr, filePath)

    def _loadNPArrayFromHDD(self, filePath):
        return utils.loadNPY(filePath)

    def _PIL2Tensor(self, pilImage):
        return self.PILImageToTensorFunction(pilImage)

    def _NPYMaker(self, filePath, preProc):
        PILImage = self._loadPILImagesFromHDD(filePath)  # PIL
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

    def _methodLoadLabel(self, componentIndex, filePath, preProc):
        labelValueList = self.LabelDataDictList[componentIndex][filePath]  # label values(list(list))

        for preProcFunc in preProc:
            LabelList = preProcFunc(labelValueList)  # preProc(labelValueList)

        return LabelList

    def _readImage(self, FilePath, preProc):
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
        rst = self._methodNPYNotExists(FilePath, preProc)
        return rst


    def _readItem(self, Type, FilePath, index, preProc):
        # data information defined in config
        if FilePath is not None:
            
            ###################################################################################
            # CASE TEXT
            ###################################################################################
            if Type["dataType"] == "Text":
                componentIndex = index % len(self.datasetComponentObjectList)
                popped = self._popItemInGlobalFileListByIndex(index)
                dataFilePath = popped["dataFilePath"]
                rst = self._methodLoadLabel(componentIndex, dataFilePath, preProc)

            ###################################################################################
            # CASE IMAGE
            ###################################################################################
            elif Type["dataType"] == "Image":
                rst = self._readImage(FilePath, preProc) if self.isCaching is False else self.cache[FilePath]
                
            ###################################################################################
            # CASE IMAGESEQUENCE
            ###################################################################################
            elif Type["dataType"] == "ImageSequence":

                seqFileList = FilePath
                rstList = []

                for seqFilePath in seqFileList:
                    rstList.append(self._readImage(seqFilePath, preProc)  if self.isCaching is False else self.cache[seqFilePath])

                rst = rstList
                

        # data information not defined in config
        else:
            rst = None

        return rst

    def __getitem__(self, index):
        #print(f"RUN PHASE 3-1")
        # popping File Path at GFL(self.globalFileList) by index
        popped = self._popItemInGlobalFileListByIndex(index)
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
                    if self.forceImageDataType is True:
                        rstDict[typ["dataName"]] = [x.squeeze(0) for x in rstDict[typ["dataName"]]]
                '''
                if labelType != {}:
                    if labelType["dataType"] == "Text":
                        pass
                    elif labelType["dataType"] == "Image":
                        assert len(rstDict[labelType["dataName"]][0].size()) == 4
                        rstDict[labelType["dataName"]] = rstDict[labelType["dataName"]].squeeze(0)
                    elif labelType["dataType"] == "ImageSequence":
                        assert len(rstDict[labelType["dataName"]][0].size()) == 4
                        if self.forceImageDataType is True:
                            rstDict[dataType["dataName"]] = rstDict[dataType["dataName"]].squeeze(0)
                '''
            '''
            else:
                if dataType["dataType"] == "Text":
                    pass
                elif dataType["dataType"] == "Image":
                    assert len(rstDict[dataType["dataName"]].size()) == 4
                    rstDict[dataType["dataName"]] = rstDict[dataType["dataName"]].squeeze(0)
                elif dataType["dataType"] == "ImageSequence":
                    assert len(rstDict[dataType["dataName"]].size()) == 4
                    if self.forceImageDataType is True:
                        rstDict[dataType["dataName"]] = rstDict[dataType["dataName"]].squeeze(0)

                if labelType != {}:
                    if labelType["dataType"] == "Text":
                        pass
                    elif labelType["dataType"] == "Image":
                        assert len(rstDict[labelType["dataName"]].size()) == 4
                        rstDict[labelType["dataName"]] = rstDict[labelType["dataName"]].squeeze(0)
                    elif labelType["dataType"] == "ImageSequence":
                        assert len(rstDict[labelType["dataName"]].size()) == 4
                        if self.forceImageDataType is True:
                            rstDict[dataType["dataName"]] = rstDict[dataType["dataName"]].squeeze(0)
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



class DataLoader(torchDataLoader):
    def __init__(
        self,
        dataLoaderName: str,
        fromParam: bool = True,
        datasetComponentParamList: Optional[List[str]] = None,
        batchSize: Optional[int] = None,
        valueRangeType: Optional[str] = None,
        isEval: Optional[bool] = None,
        augmentation: Optional[List[str]] = None,
        numWorkers: Optional[int] = None,
        makePreprocessedFile: Optional[bool] = None,
        isCaching: Optional[bool] = None,
        sequenceLength: Optional[bool] = None,
    ):

        # INIT PARAMs #
        self.name = dataLoaderName
        print(f"Preparing Dataloader {self.name}... ")

        self.datasetComponentParamList = datasetComponentParamList
        self.batchSize = batchSize
        self.valueRangeType = valueRangeType
        self.isEval = isEval
        self.augmentation = augmentation
        
        self.makePreprocessedFile = makePreprocessedFile
        
        self.isCaching = isCaching
        self.sequenceLength = sequenceLength
        self.forceImageDataType = None

        self.fromParam = fromParam

        if self.fromParam is True:
            self._getDataloaderParams()


        if self.isCaching is True and self.numWorkers > 0:
            print("data_loader.py :: WARNING :: Cached dataloader with numWorker > 0 slow down inferencing speed. Therefore, numWorkers was set by 0")
            self.numWorkers = 0
        else:
            self.numWorkers = self.numWorkers


        # CONSTRUCT DATASET #
        self.dataset = None
        self._constructDataset()

        super(DataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=self.batchSize,
            shuffle=False if self.isEval else True,
            num_workers=self.numWorkers,
            collate_fn=self.Collater,
            pin_memory=False
        )

        print(f"    - Data prepared : {len(self.dataset)} data")

    def Collater(self, samples):
        rstDict = {}

        for key in samples[0]:
            tnsrList = [sample[key] for sample in samples]
            rstDict[key] = tnsrList

        return rstDict

    def _getDataloaderParams(self):

        yamlData = Config.paramDict["data"]["dataLoader"][f"{self.name}"]

        datasetNameList = list(map(str, yamlData["datasetComponent"]))
        datasetModeList = list(Config.paramDict["data"]["datasetComponent"][name]["mode"] for name in datasetNameList)
        datasetClassParameterList = list(
            Config.paramDict["data"]["datasetComponent"][name]["classParameter"] for name in datasetNameList
        )
        datasetLabelClassNameList = list(
            (
                Config.paramDict["data"]["datasetComponent"][name]["labelClassName"]
                if "labelClassName" in Config.paramDict["data"]["datasetComponent"][name].keys()
                else "GT"
            )
            for name in datasetNameList
        )
        self.sequenceLength = yamlData["sequenceLength"] if "sequenceLength" in yamlData.keys() else None

        sequenceLengthForDatasetComponentParamList = 1 if self.sequenceLength == 0 else self.sequenceLength

        sequenceLengthForDatasetComponentParamList = [sequenceLengthForDatasetComponentParamList] * len(datasetNameList)

        self.datasetComponentParamList = zip(
            datasetNameList, datasetModeList, datasetClassParameterList, datasetLabelClassNameList, sequenceLengthForDatasetComponentParamList
        )

        self.batchSize = int(yamlData["batchSize"])
        self.valueRangeType = str(yamlData["valueRangeType"])
        self.isEval = yamlData["isEval"]
        self.augmentation = yamlData["augmentation"]
        self.numWorkers = Config.param.train.dataLoaderNumWorkers
        self.makePreprocessedFile = yamlData["makePreprocessedFile"]
        self.isCaching = yamlData["caching"]
        self.forceImageDataType = True if self.sequenceLength == 0 else False
        

    def _constructDataset(self):

        self.dataset = Dataset(
            datasetComponentParamList=self.datasetComponentParamList,
            batchSize=self.batchSize,
            valueRangeType=self.valueRangeType,
            isEval=self.isEval,
            augmentation=self.augmentation,
            numWorkers=self.numWorkers,
            makePreprocessedFile=self.makePreprocessedFile,
            isCaching=self.isCaching,
            forceImageDataType=self.forceImageDataType
        )

        for dc in self.dataset.datasetComponentObjectList:
            print(f"    - {dc.name}: {len(dc)} data")

    def __iter__(self) -> "_BaseDataLoaderIter":
        assert (
            self.num_workers >= 0
        ), "data_loader.py :: Current Version of Data Loader Only Support more than one num_workers."
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterWithDataAugmentation(self, self.augmentation)
        else:
            return _MultiProcessingDataLoaderIterWithDataAugmentation(self, self.augmentation)


class _MultiProcessingDataLoaderIterWithDataAugmentation(_MultiProcessingDataLoaderIter):
    def __init__(self, loader, augmentation):
        self.augmentation = augmentation
        super(_MultiProcessingDataLoaderIterWithDataAugmentation, self).__init__(loader)

    def _applyAugmentationFunction(self, tnsr, augmentationFuncStr: str):

        return _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr)

    def _GPUDataAugmentation(self, tnsrList, augmentations: List[str]):

        x = tnsrList

        augmentationsAfterToTensor = augmentations[augmentations.index("toTensor()") + 1 :]

        for augmentation in augmentationsAfterToTensor:
            x = self._applyAugmentationFunction(x, augmentation)

        return x

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()

        shapeList = []
        tempLabelList = []
        labelMaxLShape = 0
        for key in data:

            if key == "Text":
                shapeList = list(map(lambda x: data[key][x].shape[1], range(len(data[key]))))
                labelMaxLShape = max(shapeList)
                tempLabelList = list(map(lambda x: torch.zeros(1, labelMaxLShape, 15), range(len(data[key]))))
                for i in range(len(data[key])):
                    tempLabelList[i][:, 0 : data[key][i].shape[1], :] = data[key][i]
                data[key] = tempLabelList

        # a = time.perf_counter()
        AugedTensor = {}
        if isinstance(data[list(data.keys())[0]][0], list):
            AugedTList = self._GPUDataAugmentation(
                [torch.cat([torch.cat(x, 0).unsqueeze(0) for x in data[key]], 0).cuda() for key in data], self.augmentation
            )
        else:
            AugedTList = self._GPUDataAugmentation([torch.stack(data[key]).cuda() for key in data], self.augmentation)

        for i, key in enumerate(data):
            AugedTensor[key] = AugedTList[i]

        # print(time.perf_counter() - a)
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
                return self._process_data(data)  # CUDA MULTIPROC ERROR HERE!!


class _SingleProcessDataLoaderIterWithDataAugmentation(_SingleProcessDataLoaderIter):
    def __init__(self, loader, augmentation):
        self.augmentation = augmentation
        super(_SingleProcessDataLoaderIterWithDataAugmentation, self).__init__(loader)

    def _applyAugmentationFunction(self, tnsr, augmentationFuncStr: str):

        return _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr)

    def _GPUDataAugmentation(self, tnsrList, augmentations: List[str]):

        x = tnsrList

        augmentationsAfterToTensor = augmentations[augmentations.index("toTensor()") + 1 :]

        for augmentation in augmentationsAfterToTensor:
            x = self._applyAugmentationFunction(x, augmentation)

        return x

    def _process_data(self, data):
        shapeList = []
        tempLabelList = []
        labelMaxLShape = 0
        for key in data:

            if key == "Text":
                shapeList = list(map(lambda x: data[key][x].shape[1], range(len(data[key]))))
                labelMaxLShape = max(shapeList)
                tempLabelList = list(map(lambda x: torch.zeros(1, labelMaxLShape, 15), range(len(data[key]))))
                for i in range(len(data[key])):
                    tempLabelList[i][:, 0 : data[key][i].shape[1], :] = data[key][i]
                data[key] = tempLabelList

        # a = time.perf_counter()
        AugedTensor = {}
        if isinstance(data[list(data.keys())[0]][0], list):
            AugedTList = self._GPUDataAugmentation(
                [torch.cat([torch.cat(x, 0).unsqueeze(0) for x in data[key]], 0).cuda() for key in data], self.augmentation
            )
        else:
            AugedTList = self._GPUDataAugmentation([torch.stack(data[key]).cuda() for key in data], self.augmentation)

        for i, key in enumerate(data):
            AugedTensor[key] = AugedTList[i]

        # print(time.perf_counter() - a)
        return AugedTensor

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = pin_memory.pin_memory(data)
        return self._process_data(data)


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
    ), "data_loader.py :: invalid Augmentation Function!! check param.yaml."

    

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
            argStr = STRING.split("(")[1][:-1]
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
