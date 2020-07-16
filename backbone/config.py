'''
param.py
'''
version = '1.01.200716'

# Config file parser... hagishilta.. ha;..... mandulgi jonna shilta..

import yaml
from munch import Munch, munchify

class Config():

    param = None
    paramDict = None
    
    datasetConfig = None
    datasetConfigDict = None

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

