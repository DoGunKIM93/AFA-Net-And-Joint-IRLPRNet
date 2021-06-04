'''
param.py
'''

# Config file parser... hagishilta.. ha;..... mandulgi jonna shilta..

import yaml
from munch import Munch, munchify

class Config():

    param = None
    paramDict = None
    
    datasetConfig = None
    datasetConfigDict = None

    inference = None
    inferenceDict = None

    @classmethod
    def readParam(cls, filename): 
        with open(filename) as yamlFile:
            cls.paramDict = yaml.full_load(yamlFile)
            cls.param = munchify(cls.paramDict)
    
    @classmethod
    def readDatasetConfig(cls, filename): 
        with open(filename) as yamlFile:
            cls.datasetConfigDict = yaml.full_load(yamlFile)
            cls.datasetConfig = munchify(cls.datasetConfigDict)

    @classmethod
    def readInference(cls, filename): 
        with open(filename) as yamlFile:
            cls.inferenceDict = yaml.full_load(yamlFile)
            cls.inference = munchify(cls.inferenceDict)


def readConfigs():
    Config.readParam("param.yaml")
    Config.readDatasetConfig("datasetConfig.yaml")

readConfigs()
