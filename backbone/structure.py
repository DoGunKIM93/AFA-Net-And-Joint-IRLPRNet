'''
structure.py
'''
version = "1.0.200508"

import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse

import apex.amp as amp
from apex.parallel import DistributedDataParallel as DDP

import param as p




class DataLoaderVaultBase():
    def __init__(self):
        super(DataLoaderVaultBase, self).__init__()





class ModelListBase():
    def __init__(self):
        super(ModelListBase, self).__init__()

    def initDataparallel(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]

        for mdlStr in mdlStrLst:
            setattr(self, mdlStr, nn.DataParallel(getattr(self, mdlStr)))

    def initApexAMP(self):
        if p.mixedPrecision is True:
            opt_level = 'O0' if p.mixedPrecision is False else 'O1'
            mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
            for mdlStr in mdlStrLst:
                mdlObj = getattr(self, mdlStr)
                mdlOpt = getattr(self, mdlStr + "_optimizer") if len([attr for attr in vars(self) if attr == (mdlStr+"_optimizer")]) > 0 else None

                if mdlOpt is None:
                    mdlObj = amp.initialize(mdlObj.to('cuda'), opt_level = opt_level)
                    setattr(self, mdlStr, mdlObj)
                else:
                    mdlObj, mdlOpt = amp.initialize(mdlObj.to('cuda'), mdlOpt, opt_level = opt_level)
                    setattr(self, mdlStr, mdlObj)
                    setattr(self, mdlStr + "_optimizer", mdlOpt)

    def getList(self):
        return [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]

    def getModels(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlObjLst = []
        for mdlStr in mdlStrLst:
            mdlObjLst.append(getattr(self, mdlStr))
        return mdlObjLst
    
    def getOptimizers(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and attr.endswith("_optimizer") and not attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlOptLst = []
        for mdlStr in mdlStrLst:
            mdlOptLst.append(getattr(self, mdlStr))
        return mdlOptLst

    def getSchedulers(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and attr.endswith("_scheduler") and not attr.endswith("_pretrained")]
        mdlSchLst = []
        for mdlStr in mdlStrLst:
            mdlSchLst.append(getattr(self, mdlStr))
        return mdlSchLst

    def getPretrainedPaths(self):
        mdlStrLst = [attr for attr in vars(self) if not attr.startswith("__") and not attr.endswith("_optimizer") and not attr.endswith("_scheduler") and attr.endswith("_pretrained")]
        mdlPpaLst = []
        for mdlStr in mdlStrLst:
            mdlPpaLst.append(getattr(self, mdlStr))
        return mdlPpaLst

    def getPretrainedPath(self, mdlStr):
        pP = p.pretrainedPath + getattr(self, mdlStr + "_pretrained")
        return pP

