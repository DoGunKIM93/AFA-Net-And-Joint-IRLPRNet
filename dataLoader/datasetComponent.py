# FROM Python LIBRARY
import os
import yaml
import inspect
import itertools
import pickle #5 as pickle

from typing import List, Dict, Tuple, Union, Optional
from PIL import Image

# FROM This Project
import backbone.preprocessing
import backbone.augmentation
from backbone.config import Config

from .datasetConfig import DatasetConfig


# Read Preprocessings from backbone.preprocessing and Augmentations from backbone.augmentation automatically
PREPROCESSING_DICT = dict(
    x for x in inspect.getmembers(backbone.preprocessing) if (not x[0].startswith("__")) and (inspect.isclass(x[1]))
)
AUGMENTATION_DICT = dict(
    x for x in inspect.getmembers(backbone.augmentation) if (not x[0].startswith("_")) and (inspect.isfunction(x[1]))
)

METADATA_DICT_KEYS = ['height', 'width', 'channel']

EXT_DICT = {"Text" :             ['txt'],
            "Image" :            ['png','jpg','jpeg','gif','bmp','tif','tiff'],
            "ImageSequence" :    ['png','jpg','jpeg','gif','bmp','tif','tiff'],
            "Video" :            ['avi','mp4','mkv','wmv','mpg','mpeg'], 
            }

IS_ITERABLE_DICT = {"Text" :             False,
                    "Image" :            False,
                    "ImageSequence" :    True,
                    "Video" :            False, 
                    }


# Representation of single dataset property
class DatasetComponent:
    def __init__(self, name, outDict, outOrder, filterDict, sequenceLength):

        self.name = name

        self.datasetConfig = None
        self._getDatasetConfigByComponentName()

        self.outDict = outDict
        self.outOrder = outOrder

        self.metaDataDict = None
        if self._loadDatasetMetadata() is False:
            self._makeDatasetMetadata()
            self._loadDatasetMetadata()

        self.filterDict = filterDict
        self._parsingFilterDict()

        self.sequenceLength = sequenceLength

        self.dataFileList = None
        self.labelFileList = None
        self._getDataFileList()
    
        self.preprocessingList = None
        self._makePreprocessingList()

    def _getDatasetConfigByComponentName(self):
        self.datasetConfig = DatasetConfig(Config.paramDict["data"]["datasetComponent"][self.name]["dataConfig"])



    def _loadDatasetMetadata(self):

        #load metadata.air
        mainPath = Config.param.data.path.datasetPath
        path = f"{self.datasetConfig.origin}/" 

        try:
            metadataFile = open(mainPath + path + 'metadata.air', 'rb')
        except:
            return False

        metadataDict = pickle.load(metadataFile)
        metadataFile.close()

        '''
        # comapre keys
        storedKeys = list(metadataDict[list(metadataDict.keys())[0]].keys())
        stroedtItemCount = len(metadataDict.keys()) 
        if sorted(storedKeys) != sorted(METADATA_DICT_KEYS):
            return False


        # 데이터파일리스트가 메타데이터 키들의 서브셋인지 (메타데이터가 기록된 파일들의 서브셋인지)
        if set(dataFileList).issubset(set(metadataDict.keys())) is False:
            return False
        '''

        self.metaDataDict = metadataDict 

    def _makeDatasetMetadata(self):

        mainPath = Config.param.data.path.datasetPath
        path = f"{self.datasetConfig.origin}/" 

        print("There is no valid metadata. Create one...")
        metadataDict = {}
        print("Get file list...")

        EXT_LIST = []
        #for key in EXT_DICT.keys(): #TODO:
        #    EXT_LIST += EXT_DICT[key]
        EXT_LIST += EXT_DICT['Image']

        fileList = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(mainPath+path):
            for file in f:
                if file.lower().endswith(tuple(EXT_LIST)):
                    fileList.append(os.path.relpath(os.path.join(r, file), mainPath))
        print("File list get.")

        for i, fileName in enumerate(fileList):
            print(f"Analyze... {i}/{len(fileList)}          ", end='\r')
            img = Image.open(mainPath+fileName)
            elemDict = {}

            for keys in METADATA_DICT_KEYS:
                if keys == 'channel':   
                    elemData = len(img.getbands())
                elif keys == 'height':   
                    elemData = img.size[1]
                elif keys == 'width':   
                    elemData = img.size[0]

                elemDict[keys] = elemData
            
            metadataDict[fileName] = elemDict
        print(f"Saving {len(fileList)}*{len(METADATA_DICT_KEYS)} recodes to metadata.air...                            ")
        with open(mainPath + path + 'metadata.air', 'wb') as f:
            pickle.dump(metadataDict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Finished.")
    


    def _parsingFilterDict(self):
        newFilterDict = {}

        for _key in self.filterDict.keys():
            #parsing filterDict
            filterStr = str(self.filterDict[_key]).replace(' ','')
            filterMax = None
            filterMin = None

            if '~' not in filterStr:
                filterMax = int(filterStr)
                filterMin = int(filterStr)
            elif filterStr[0] == '~':
                filterMax = int(filterStr[1:])
                filterMin = 0
            elif filterStr[-1] == '~':
                filterMax = 2**32-1
                filterMin = int(filterStr[:-1])
            else:
                filterMin, filterMax = tuple([int(x) for x in filterStr.split('~')])

            assert filterMax is not None and filterMin is not None, "datasetComponent.py :: filter parsing error."
            newFilterDict[_key] = [filterMin, filterMax]
        
        self.filterDict = newFilterDict



    def _getDataFileList(self):

        def _getDatasetPath():
            # get origin path
            mainPath = Config.param.data.path.datasetPath if Config.param.data.path.datasetPath[-1] == '/' else Config.param.data.path.datasetPath + '/'
            datasetPath = f"{self.datasetConfig.origin}/"#{self.mode}/"
            return mainPath, datasetPath

        def _getClassPathList(classParameterDictList):
            datasetConfigClassList = self.datasetConfig.classes
            classPathList = []

            for classParameterDict in classParameterDictList:
                DatasetComponentClassList = list(classParameterDict.keys())
                t_classPathList = ['']
                # dynamic construction of class path based on defined classes
                for datasetConfigClass in datasetConfigClassList:
                    
                    # if DatasetComponentClassList have datasetConfigClass
                    if datasetConfigClass in DatasetComponentClassList:
                        newClassPathList = []
                        for folderName in classParameterDict[datasetConfigClass]:
                            newClassPathList += ([x + ('' if x == '' else '/') + folderName for x in t_classPathList])
                        t_classPathList = newClassPathList
                    else:
                        break
                classPathList += t_classPathList

            '''
            for i in range(len(datasetConfigClasses)):
                classPathList = (
                    list(
                        itertools.chain.from_iterable(
                            list(
                                map(
                                    lambda y: list(
                                        map(
                                            lambda x: (str(x) if type(x) is int else x) + "/" + (str(y) if type(y) is int else y),
                                            classPathList,                                                                                
                                        )
                                    ),
                                    self.classParameter[datasetConfigClasses[i]],
                                )
                            )
                        )
                    )
                    if i != 0
                    else self.classParameter[datasetConfigClasses[i]]
                )
                # Sorry for who read this
            '''
            return classPathList

        def _makeDataFileList(dataType:str, mainPath, pathList):

            assert dataType in EXT_DICT.keys()

            if IS_ITERABLE_DICT[dataType] is False:
                dataFileLists = list(
                    map(
                        lambda x: list(
                            filter(
                                lambda x: (x.lower().endswith(tuple(EXT_DICT[dataType]))),
                                list(map(lambda y: x + "/" + y, sorted(os.listdir(mainPath + x)))),
                            )
                        ),
                        pathList,
                    )
                )
            
            else:
                dataFileLists = list(map(lambda x: list(map(lambda y: x + "/" + y, sorted(filter(lambda z : os.path.isdir(mainPath + x + "/" + z), os.listdir(mainPath + x))))), pathList))
                dataFileLists = [
                    list(
                        map(
                            lambda x: [
                                x + "/" + z
                                for z in list(
                                    filter(lambda y: (y.lower().endswith(tuple(EXT_DICT[dataType]))), sorted(os.listdir(mainPath + x)))
                                )
                            ],
                            xlist,
                        )
                    )
                    for xlist in dataFileLists
                ]

            return dataFileLists

        def _dataFilePathDictValidation(dataFilePathDict):

            #Count validation
            oldLength = -1
            for key in dataFilePathDict.keys():
                length = len(dataFilePathDict[key])

                assert length == 1 or oldLength == -1 or oldLength == length, "datasetComponent.py :: all out data has same data count."
                
                if length != 1 and oldLength == -1:
                    oldLength = length

            for key in dataFilePathDict.keys():
                length = len(dataFilePathDict[key])                
                if length == 1:
                    dataFilePathDict[key] = dataFilePathDict[key] * oldLength

            return dataFilePathDict

        def _filtering(dataFilePathChunk):

            def _filterDatum(dataFilePath):
                if len(self.filterDict) == 0:
                    return True

                #if metadata doesn't have datafile's metadata
                if dataFilePath not in self.metaDataDict.keys():
                    return None

                _metadata = self.metaDataDict[dataFilePath]

                #if metadata's keys aren't a superset of datafile's keys
                if set(self.filterDict.keys()).issubset(set(_metadata.keys())) is False:
                    return None

                for _key in _metadata.keys():
                    filterMin, filterMax = self.filterDict[_key]
                    if not (filterMin <= _metadata[_key] <= filterMax):
                        return False
                
                return True
            
            for dataFilePath in dataFilePathChunk:
                #iterables
                if isinstance(dataFilePath, list):
                    for dataFilePathElem in dataFilePath:
                        filterRst = _filterDatum(dataFilePathElem)
                        if filterRst is None:
                            return None
                        elif filterRst is False:
                            return False
                #not iterables
                else:
                    filterRst = _filterDatum(dataFilePath)
                    if filterRst is None:
                        return None
                    elif filterRst is False:
                        return False

            return True
                    

        #get dataset origin folder path, dataset path in dataset origin folder
        mainPath, datasetPath = _getDatasetPath()

        dataDictKeyList = list(self.outDict.keys())
        dataFilePathDict = {}

        #type validation
        assert [self.outDict[dataDictKeyList[0]]['type']] * len(dataDictKeyList) \
             == [self.outDict[dataDictKeyList[x]]['type'] for x in range(len(dataDictKeyList))] \
                 , "datasetComponent.py :: all type of out data must be same." 

        for dataDictKey in dataDictKeyList:
            _type = self.outDict[dataDictKey]['type']
            _classParameterDict = self.outDict[dataDictKey]['classParameter']

            #get relative class folder paths.
            _classPathList = _getClassPathList(_classParameterDict)

            # add dataset path in front of all elements of class path list
            _pathList = [datasetPath + x for x in _classPathList]

            #make data file lists
            _dataFilePathList = _makeDataFileList(_type, mainPath, _pathList)

            dataFilePathDict[dataDictKey] = _dataFilePathList

        for dataDictKey in dataDictKeyList:
            dataFilePathDict[dataDictKey] = sum(dataFilePathDict[dataDictKey], [])


        #validation all dataDict value have same length, if 1 -> make copies
        dataFilePathDict = _dataFilePathDictValidation(dataFilePathDict)

        #zip in order of dataLoader's outOrder
        dataFilePathDictList = list(zip(*[dataFilePathDict[key] for key in self.outOrder]))#list(zip([dataFilePathDict[key] for key in self.outOrder]))

        '''
        dataFileDictList = list(
            map(
                lambda x: dict(zip(["dataFilePath", "labelFilePath"], x)),
                list(zip(itertools.chain.from_iterable(dataFileLists), itertools.chain.from_iterable(labelFiles))),
            )
        )
        '''

        # load meta data and filtering dataFileList.
        retryCount = 0
        while(True):
            retryCount += 1
            assert retryCount <= 5, "datasetComponent.py :: Cannot make metadata file..."

            newDataFilePathDictList = []
            for dataFilePathChunk in dataFilePathDictList:
                filterRst = _filtering(dataFilePathChunk)
                if filterRst is None:
                    break
                elif filterRst is True:
                    newDataFilePathDictList.append(dataFilePathChunk)
            
            if filterRst is not None:
                break
            else:
                self._makeDatasetMetadata()
                self._loadDatasetMetadata()
        
        dataFilePathDictList = newDataFilePathDictList

        # set dataFileList without main path
        self.dataFileList = dataFilePathDictList

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
            len(self.dataFileList)# if self.datasetConfig.dataType["dataType"] != "ImageSequence" else self._getSeqDataLen()
        )
