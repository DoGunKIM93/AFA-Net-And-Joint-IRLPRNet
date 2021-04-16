# FROM Python LIBRARY
import numpy as np
import random

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


from .dataset import Dataset, _applyAugmentationFunctionFunc
from .datasetComponent import DatasetComponent, PREPROCESSING_DICT, AUGMENTATION_DICT
from .datasetConfig import DatasetConfig



# Prevent memory Error in PIL
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
ImageFile.LOAD_TRUNCATED_IMAGES = True



class DataLoader(torchDataLoader):
    def __init__(
        self,
        dataLoaderName: str,
        fromParam: bool = True,
        datasetComponentParamList: Optional[List[str]] = None,
        mainPath: Optional[str] = None,

        batchSize: Optional[int] = None,
        type: Optional[str] = None,
        range: Optional[str] = None,

        isEval: Optional[bool] = None,
        isCaching: Optional[bool] = None,

        sequenceLength: Optional[bool] = None,

        outOrder: Optional[List[str]] = None,
        filter: Dict = None,
        augmentation: Optional[List[str]] = None,

        numWorkers: Optional[int] = None,
    ):

        # INIT PARAMs #
        self.name = dataLoaderName

        # PRINT
        print("")
        print("")
        print(f"Preparing Dataloader {self.name}... ")
        for _key in Config.paramDict['data']['dataLoader'][self.name].keys():
            elem = Config.paramDict['data']['dataLoader'][self.name][_key]
            if isinstance(elem, list):
                print(f"    - {_key}")
                for _e in elem:
                    print(f"        {_e}")
            elif isinstance(elem, dict):
                print(f"    - {_key}")
                for _e in elem.keys():
                    print(f"        {_e}: {elem[_e]}") 
            else:
                print(f"    - {_key}: {elem}")

        self.datasetComponentParamList = datasetComponentParamList
        self.mainPath = mainPath

        self.batchSize = batchSize
        self.range = range
        self.type = type

        self.isEval = isEval
        self.isCaching = isCaching

        self.sequenceLength = sequenceLength

        self.outOrder = outOrder
        self.filter = filter
        self.augmentation = augmentation
        self.numWorkers = numWorkers
                
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

        print(f"{self.name} DataLoader prepared : {len(self.dataset)} data")

    def Collater(self, samples):
        rstDict = {}

        for key in samples[0]:
            tnsrList = [sample[key] for sample in samples]
            rstDict[key] = tnsrList

        return rstDict

    def _getDataloaderParams(self):

        yamlData = Config.paramDict["data"]["dataLoader"][f"{self.name}"]

        self.mainPath = str(Config.paramDict["data"]["path"]["datasetPath"])

        self.range = str(yamlData["range"])
        self.type = str(yamlData["type"])

        self.isEval = yamlData["isEval"]
        self.isCaching = yamlData["caching"] 

        self.outOrder = list(map(str, yamlData["outOrder"]))
        self.filter = yamlData["filter"] if 'filter' in yamlData.keys() else {}
        self.augmentation = yamlData["augmentation"]
                
        self.batchSize = int(yamlData["batchSize"])
        
        self.numWorkers = Config.param.train.dataLoaderNumWorkers

        datasetNameList = list(map(str, yamlData["datasetComponent"]))
        datasetOutDictList = list(Config.paramDict["data"]["datasetComponent"][name]["out"] for name in datasetNameList)
        
        for _i in range(len(datasetOutDictList)):
            for _key in datasetOutDictList[_i].keys():
                if isinstance(datasetOutDictList[_i][_key]['classParameter'], list) is False:
                    datasetOutDictList[_i][_key]['classParameter'] = [datasetOutDictList[_i][_key]['classParameter']]
                for _j in range(len(datasetOutDictList[_i][_key]['classParameter'])):
                    for _class in datasetOutDictList[_i][_key]['classParameter'][_j].keys():
                        datasetOutDictList[_i][_key]['classParameter'][_j][_class] = [str(x) for x in datasetOutDictList[_i][_key]['classParameter'][_j][_class]]
            
        datasetOutOrderList = [self.outOrder] * len(datasetNameList)
        datasetFilterDictList = [self.filter] * len(datasetNameList)

        self.sequenceLength = yamlData["sequenceLength"] if "sequenceLength" in yamlData.keys() else None

        sequenceLengthForDatasetComponentParamList = 1 if self.sequenceLength == 0 else self.sequenceLength
        sequenceLengthForDatasetComponentParamList = [sequenceLengthForDatasetComponentParamList] * len(datasetNameList)

        self.datasetComponentParamList = zip(
            datasetNameList, datasetOutDictList, datasetOutOrderList, datasetFilterDictList, sequenceLengthForDatasetComponentParamList
        )


               

    def _constructDataset(self):

        self.dataset = Dataset(
            datasetComponentParamList=self.datasetComponentParamList,
            mainPath=self.mainPath,

            batchSize=self.batchSize,
            type=self.type,
            range=self.range,

            isEval=self.isEval,
            isCaching=self.isCaching,

            outOrder=self.outOrder,
            filter=self.filter,
            augmentation=self.augmentation
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



