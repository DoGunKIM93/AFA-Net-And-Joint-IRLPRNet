'''
param.py
'''
version = '1.0.200709'

# Config file parser... hagishilta.. ha;..... mandulgi jonna shilta..

import yaml
from munch import Munch, munchify

class Config():

    param = None
    datasetConfig = None

    @classmethod
    def readParam(cls, filename): 
        with open(filename) as yamlFile:
            cls.param = munchify(yaml.full_load(yamlFile))
    
    @classmethod
    def readDatasetConfig(cls, filename): 
        with open(filename) as yamlFile:
            cls.datasetConfig = munchify(yaml.full_load(yamlFile))

