<div align="center"><img src='airlogo.png' width=600></div>





Index
=====

 [Getting Started](/confluence/display/AILAB/1.+Getting+Started)
----------------------------------------------------------------

*   [PREREQUISITES](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-PREREQUISITES)
    *   [OS](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-OS)
    *   [Python](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-Python)
    *   [Pytorch](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-Pytorch)
    *   [Recommended Method](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-RecommendedMethod)
*   [START WITH DOCKER CONTAINER LOCALLY](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-STARTWITHDOCKERCONTAINERLOCALLY)
    *   [Pull container image](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-Pullcontainerimage)
    *   [Run Docker Container](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-RunDockerContainer)
    *   [Install AIR Research Framework](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-InstallAIRResearchFramework)
*   [START WITH HAIQV](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-STARTWITHHAIQV)
    *   [Make new Notebook server](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-MakenewNotebookserver)
*   [RUN YOUR FIRST MODEL](https://rnd.hsnc.co.kr/confluence/display/AILAB/1.+Getting+Started#id-1.GettingStarted-RUNYOURFIRSTMODEL)

 [Develop Your First Model](/confluence/display/AILAB/2.+Develop+Your+First+Model)
----------------------------------------------------------------------------------

*   [OPTIONAL: DEVELOP WITH VS CODE](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-OPTIONAL:DEVELOPWITHVSCODE)
    *   [Use VS Code with native Docker container environment](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-UseVSCodewithnativeDockercontainerenvironment)
    *   [Use VS Code with HAIQV Platform](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-UseVSCodewithHAIQVPlatform)
*   [SRCNN: THE FIRST CNN-BASED SUPER RESOLUTION MODEL](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-SRCNN:THEFIRSTCNN-BASEDSUPERRESOLUTIONMODEL)
    *   [Define networks](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-Definenetworks)
    *   [Define Data](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-DefineData)
    *   [Training code](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-Trainingcode)
    *   [Start training](https://rnd.hsnc.co.kr/confluence/display/AILAB/2.+Develop+Your+First+Model#id-2.DevelopYourFirstModel-Starttraining)

 [Data Loader](/confluence/display/AILAB/Data+Loader)
-----------------------------------------------------

*   [DATASET CONFIG](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-DATASETCONFIG)
*   [DATASET COMPONENT](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-DATASETCOMPONENT)
*   [DATA LOADER](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-DATALOADER)
    *    [Multiple datasets](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-Multipledatasets)
    *    [Fast loading with preprocessed cache](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-Fastloadingwithpreprocessedcache)
    *    [Hight speed data augmentation with CUDA](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-HightspeeddataaugmentationwithCUDA)
*   [EXAMPLES](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-EXAMPLES)
    *    [DIV2K Dataset](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-DIV2KDataset)
        *    [datasetConfig](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-datasetConfig)
        *    [datasetComponent](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-datasetComponent)
        *    [dataLoader](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-dataLoader)
*   [ARCHITECTURE](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Loader#DataLoader-ARCHITECTURE)

 [Data Augmentation](/confluence/display/AILAB/Data+Augmentation)
-----------------------------------------------------------------

*   [DATA AUGMENTATION FUNTIONS IN DATALOADER](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-DATAAUGMENTATIONFUNTIONSINDATALOADER)
*   [DATATYPE TRANSFORM](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-DATATYPETRANSFORM)
    *    [toTensor](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-toTensor)
*   [RESIZE](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-RESIZE)
    *    [sizeMatch](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-sizeMatch)
    *    [resize](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-resize)
    *    [resizeToMultipleOf](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-resizeToMultipleOf)
    *    [shrink](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-shrink)
    *    [shrinkWithRandomMethod](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-shrinkWithRandomMethod)
    *    [resizeWithTextLabel](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-resizeWithTextLabel)
    *    [virtualScaling](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-virtualScaling)
*   [CROP](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-CROP)
    *    [centerCrop](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-centerCrop)
    *    [centerCropToMultipleOf](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-centerCropToMultipleOf)
    *    [randomCrop](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-randomCrop)
    *    [randomCropWithRandomSize](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-randomCropWithRandomSize)
*   [FLIP & ROTATE](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-FLIP&ROTATE)
    *    [randomFlip](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-randomFlip)
    *    [randomRotate](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-randomRotate)
*   [FILTER](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-FILTER)
    *    [randomGaussianBlur](https://rnd.hsnc.co.kr/confluence/display/AILAB/Data+Augmentation#DataAugmentation-randomGaussianBlur)

 [Models: _**model.py**_](/confluence/display/AILAB/Models%3A+model.py)
-----------------------------------------------------------------------

*   [DEFINE MODEL](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-DEFINEMODEL)
*   [BACKBONE.MODULE.MODULE](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-BACKBONE.MODULE.MODULE)
*   [PRE-DEFINED MODELS](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-PRE-DEFINEDMODELS)
    *    [backbone.predefined.ESPCN](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.ESPCN)
    *    [backbone.predefined.VDSR](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.VDSR)
    *    [backbone.predefined.SPSR](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.SPSR)
    *    [backbone.predefined.EDVR](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.EDVR)
    *    [backbone.predefined.VESPCN](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.VESPCN)
    *    [backbone.predefined.EfficientNet](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.EfficientNet)
    *    [backbone.predefined.RetinaFace](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.RetinaFace)
    *    [backbone.predefined.ResNeSt](https://rnd.hsnc.co.kr/confluence/display/AILAB/Models%3A+model.py#Models:model.py-backbone.predefined.ResNeSt)

 [Parameters & Settings: _**param.py**_](/confluence/display/AILAB/Parameters%3A+param.yaml)
--------------------------------------------------------------------------------------------

*   [SETTING PARAMETERS WITH param.yaml](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-SETTINGPARAMETERSWITHparam.yaml)
*   [general](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-general)
    *    [general.GPUNum](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-general.GPUNum)
*   [data](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-data)
    *    [data.path](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-data.path)
    *    [data.path.datasetPath](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-data.path.datasetPath)
    *    [data.path.pretrainedPath](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-data.path.pretrainedPath)
    *    [data.datasetComponent](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-data.datasetComponent)
    *    [data.dataLoader](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-data.dataLoader)
*   [train](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train)
    *    [train.step](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.step)
    *    [train.step.maxEpoch](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.step.maxEpoch)
    *    [train.step.validationStep](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.step.validationStep)
    *    [train.step.archiveStep](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.step.archiveStep)
    *    [train.step.earlyStopStep](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.step.earlyStopStep)
    *    [train.method](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.method)
    *    [train.method.mixedPrecision](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.method.mixedPrecision)
    *    [train.dataLoaderNumWorkers](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-train.dataLoaderNumWorkers)
*   [save](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-save)
    *    [save.font](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-save.font)
    *    [save.font.path](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-save.font.path)
    *    [save.maxSaveImageNumberTrain](https://rnd.hsnc.co.kr/confluence/display/AILAB/Parameters%3A+param.yaml#Parameters:param.yaml-save.maxSaveImageNumberTrain)

 [Training: edit.py](/confluence/display/AILAB/Training+and+Inferencing%3A+edit.py)
-----------------------------------------------------------------------------------



# 버그추가 및 기능제보 문의 : 솔루션 유닛 김진