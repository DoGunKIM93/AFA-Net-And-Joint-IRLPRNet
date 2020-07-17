
from backbone.config import Config

from functools import reduce
import itertools
import os

Config.readDatasetConfig("datasetConfig.yaml")
Config.readParam("param.yaml")

datasetConfig = Config.datasetConfigDict['DIV2K_SR']
mode = Config.paramDict['data']['datasetComponent']['DIV2K_SR_train']['mode']
classParameter = Config.paramDict['data']['datasetComponent']['DIV2K_SR_train']['classParameter']

mainPath = Config.param.data.path.datasetPath






#mainPath = Config.param.data.path.datasetPath
path = f"{datasetConfig['origin']}/{mode}/"

TVTList = datasetConfig['availableMode']
#TVTDict = dict(zip(TVTList, list(map( lambda x: f"{path}/{x}/", TVTList))))

for i in range(len(datasetConfig['classes'])):
    classPathList = list(itertools.chain.from_iterable(list(map( lambda y : list(map(lambda x : str(x) if type(x) is int else x + '/' + str(y) if type(y) is int else y , classPathList)), classParameter[datasetConfig['classes'][i]])))) if i is not 0 else classParameter[datasetConfig['classes'][i]]
    # Sorry for who read this

pathList = list(map( lambda x : path + x , classPathList))

fileList = list(itertools.chain.from_iterable(list(map( lambda x :  list(map( lambda y : x + "/" + y, os.listdir(mainPath + x))) , pathList))))
fileList = [x for x in fileList if (x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg") or x.endswith(".bmp")) ]


print(len(fileList))

