"""
inference.py

Usage ::: inference.py -n [inference model name] -i [input image(s)/video(s) Folder Path] -o [output image/video File Path / images/videos Folder Path] -f [fps]
Ex) 

1. single video
python inference.py -n General-SR-PSNR-Advanced-DeFiAN -i /home/jovyan/dataset_military/dataset/100.5M_배_IR.mp4 -o /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/singleVideoTest

2. single image
python inference.py -n General-SR-PSNR-Advanced-DeFiAN -i /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/testIN/test2.png -o /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/testOUT/test2_result.png

3. multi videos
python inference.py -n General-SR-PSNR-Advanced-DeFiAN -i /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/videos/ -o /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/multiVideoTest

4. multi images
python inference.py -n General-SR-PSNR-Advanced-DeFiAN -i /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/testIN/ -o /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/testOUT

"""


# READ YAML
# CONSTRUCT MODEL
# PASS DATA

import os
import argparse
import inspect
import numpy as np
import torch
import torchvision.transforms
import backbone.augmentation as augmentation
import backbone.predefined
import backbone.utils as utils
import time

from typing import List, Dict, Tuple, Union, Optional
from collections.abc import Iterable
from PIL import Image
from torchvision.utils import save_image
from backbone.config import Config
from torch.nn import DataParallel

from dataLoader.datasetComponent import EXT_DICT
from tools.videoToImageSequence import makeVideoFileList, videoToImages
from tools.imageSequenceToVideo import makeImageSequenceFileList, imagesToVideo


# Read Augmentations from backbone.augmentation automatically
AUGMENTATION_DICT = dict(
    x for x in inspect.getmembers(augmentation) if (not x[0].startswith("_")) and (inspect.isfunction(x[1]))
)

# GPU Setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.inference.general.GPUNum)


def _generateFileList(path: str, type: str) -> List[str]:
    formatList = []

    if type == 'All' : 
        formatList = EXT_DICT['ImageSequence'] + EXT_DICT['Video']
    elif type == 'ImageSequence':
        formatList = EXT_DICT['ImageSequence']
    elif type == 'Video':
        formatList = EXT_DICT['Video']

    fileList = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if (
            os.path.isfile(os.path.join(path, f))
            and f.split(".")[-1].lower() in formatList
        )
    ]

    return fileList


def _calRuntime(oldTime: float, numOfFiles=None):

    nowtime = time.perf_counter()
    fps = 1/(nowtime - oldTime)

    if numOfFiles == None:
        print(f"Time per inference: {(nowtime - oldTime):.2f} sec                    ")
        print(f"FPS per inference: {fps:.2f} fps                    ")
    else:
        print(f"Time per Batch: {(nowtime - oldTime):.2f} sec, Average time per Batch: {(nowtime - oldTime)/numOfFiles:.2f} sec,")    
        print(f"FPS per Batch: {fps:.2f} fps                    ")


def _loadModel(inferencePresetName):
    # model informations
    model_name = Config.inferenceDict["model"][inferencePresetName]["name"]
    model_param = Config.inferenceDict["model"][inferencePresetName]["param"]
    model_weight = Config.inferenceDict["model"][inferencePresetName]["weight"]

    # import module by string
    module = getattr(backbone.predefined, model_name)

    # load model network with params
    paramList = str(model_param).replace(" ", "").split(",")
    model = module(
        *list(
            (x if x.replace(".", "", 1).isdigit() is False else (int(x) if x.find(".") is -1 else float(x)))
            for x in paramList
        )
    )
    model.cuda()
    model = DataParallel(model)

    # load checkpoint weight
    checkpoint = torch.load(Config.inference.data.path.pretrainedPath + model_weight)
    # model.load_state_dict(checkpoint['model'],strict=True)
    while(True):
        try:
            mthd = "NORMAL"
            model.load_state_dict(checkpoint["model"], strict=True)
            break
        except:
            pass
        try:
            mthd = "GLOBAL STRUCTURE"
            model.load_state_dict(checkpoint, strict=True)
            break
        except:
            pass
        try:
            mthd = "INNER MODEL"
            model.module.load_state_dict(checkpoint["model"], strict=True)
            break
        except:
            pass
        try:
            mthd = "NORMAL state_dict"
            model.load_state_dict(checkpoint["state_dict"])
            break
        except:
            pass
        try:
            mthd = "INNER MODEL GLOBAL STRUCTURE"
            model.module.load_state_dict(checkpoint, strict=True)
            break
        except:
            pass
        try:
            mthd = "INNER NORMAL state_dict"
            model.module.load_state_dict(checkpoint["state_dict"])
            break
        except:
            pass
        try:
            mthd = "UNSTRICT (WARNING : load weights imperfectly)"
            model.load_state_dict(checkpoint["model"], strict=False)
            break
        except:
            pass
        try:
            mthd = "GLOBAL STRUCTURE UNSTRICT (WARNING : load weights imperfectly)"
            model.load_state_dict(checkpoint, strict=False)
            break
        except:
            mthd = "FAILED"
            print("utils.py :: model load failed..... I'm sorry~")
            break

    model.eval()

    print(
        f"{model_name} Loaded with {mthd} mode."
        if mthd != "FAILED"
        else f"{model_name} Load Failed."
    )

    paramSize = 0
    for parameter in model.parameters():
        paramSize = paramSize + np.prod(np.array(parameter.size()))
    print(f"parameter size of {model_name} : {str(paramSize)}")

    return model


def _modelInference(inp, model):
    # inference TENSOR with no grad
    with torch.no_grad():
        out = model(inp)
    return out


def _convertTensorTo(inp, outputType):
    if outputType == "TENSOR":
        pass
    elif outputType == "PIL":
        out = torchvision.transforms.ToPILImage()(out)
    return out


def _applyAugmentationFunction(tnsr, augmentationFuncStr: str):

    return _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr)


def _applyAugmentationFunctionFunc(tnsr, augmentationFuncStr: str):

    assert (
        augmentationFuncStr.split("(")[0] in AUGMENTATION_DICT.keys()
    ), "inference.py :: invalid Augmentation Function!! chcek inference.yaml."

    augFunc = AUGMENTATION_DICT[augmentationFuncStr.split("(")[0]]
    args = []
    for x in list(filter(lambda y: y != "", augmentationFuncStr.split("(")[1][:-1].replace(" ", "").split(","))):
        if (x[1:] if x[0] == "-" else x).replace(".", "", 1).isdigit() is False:
            args.append(str(x))
        else:
            if x.find(".") == -1:
                args.append(int(x))
            else: 
                args.append(float(x))
    
    tnsr = augFunc(tnsr, *args)

    return tnsr


def inferenceSingle(inp, inferencePresetName, model=None, outputType=None, outputPath=None, originalSave=True):
    print("Processing Input...")

    # process input
    if isinstance(inp, str):
        inp = Image.open(inp)
        outputType = "FILE" if outputType is None else outputType
    else:
        outputType = augmentation._getType([inp]) if outputType is None else outputType

    assert outputType in [
        None,
        "FILE",
        "TENSOR",
        "PIL",
    ], f"inference.py :: outputType '{outputType}' is not supported. Supported types: 'FILE', 'TENSOR', 'PIL'"
    
    assert outputType != "FILE" or (
        outputType == "FILE" and outputPath is not None
    ), "inference.py :: outputPath must be exist if outputType is 'FILE'"

    model_augmentation = Config.inferenceDict["model"][inferencePresetName]["augmentation"]
    model_valueRangeType = Config.inferenceDict["model"][inferencePresetName]["valueRangeType"]
    
    # augmentation for input image
    for augmentationFuc in model_augmentation:
        inp = _applyAugmentationFunction([inp], augmentationFuc)
        inp = inp[0]
    
    inp = inp.unsqueeze(0).cuda()
    inp = inp * 2 - 1 if model_valueRangeType == "-1~1" else torch.round(inp * 255) if model_valueRangeType == "0~255" else inp

    # load Model
    if model is None:    
        print("Load Model...")
        model = _loadModel(inferencePresetName)

    # inference
    print("Inferencing...")
    timePerInference = time.perf_counter()  # 1 inference 당 시간    
    out = _modelInference(inp, model)

    # convert to output
    if outputType == "FILE":
        # result save
        [
            utils.saveImageTensorToFile(
                {"Result": x}, 
                f'{outputPath.split(".")[0]}-{i}.{outputPath.split(".")[1]}', 
                caption=False, 
                valueRange=model_valueRangeType
            )
            for i, x in enumerate(out)
        ] if isinstance(out, tuple) or isinstance(out, list) else utils.saveImageTensorToFile(
            {"Result": out}, 
            outputPath, 
            caption=False, 
            valueRange=model_valueRangeType
        )

        if originalSave == True:
            # original save
            utils.saveImageTensorToFile(
                {"Result": inp},
                f'{outputPath.split(".")[0]}-original.{outputPath.split(".")[1]}',
                caption=False,
                valueRange=model_valueRangeType
            )
        print("Saved.")

        _calRuntime(timePerInference)
        
        return
    else:
        rst = _convertTensorTo(out, outputType)
        print("Finished.")
        return rst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--inferencePresetName", "-n", help="인퍼런스 프리셋 이름")
    parser.add_argument("--inputPath", "-i", help="입력할 이미지/비디오 파일 경로(단일) or 이미지/비디오 폴더 경로(다중)")
    parser.add_argument("--outputPath", "-o", help="출력할 이미지/비디오 파일 경로(단일) or 이미지/비디오 폴더 경로(다중)")
    parser.add_argument("--fps", "-f", default="24", help="fps (video only)")

    args = parser.parse_args()

    print("Load Model...")
    model = _loadModel(args.inferencePresetName)

    # input :  folder
    if os.path.isdir(args.inputPath):
        if os.path.exists(args.outputPath) is False:
            os.makedirs(args.outputPath)
        assert os.path.isdir(args.outputPath), "if inputPath is a dir, outputPath must be a dir"

        fileList = _generateFileList(args.inputPath, "All")
        fileList.sort()
        timePerBatch = time.perf_counter()  # 1 inference 당 시간   

        # folders : images
        if fileList[0].split(".")[-1].lower() in EXT_DICT['Image']:

            for i, imageFile in enumerate(fileList):
                print(f"[{i}/{len(fileList)}] processing {imageFile} to {args.outputPath}/{imageFile.split('/')[-1]}")
                inferenceSingle(
                    imageFile,
                    args.inferencePresetName,
                    model=model,
                    outputType="FILE",
                    outputPath=f"{args.outputPath}/{imageFile.split('/')[-1]}",
                )

            _calRuntime(timePerBatch, len(fileList))

        # folder : videos
        elif fileList[0].split(".")[-1].lower() in EXT_DICT['Video']:
            
            videofiles = makeVideoFileList(args.inputPath)
            ImageSqeuencePathList = videoToImages(videofiles, args.outputPath)

            for i, ImageSqeuencePath in enumerate(ImageSqeuencePathList):
                imageSquenceResultFolder = args.outputPath + "/" + ImageSqeuencePath.split('/')[-2] + "_results/"

                if os.path.exists(imageSquenceResultFolder) is False:
                    os.makedirs(imageSquenceResultFolder)

                imageFileLst = _generateFileList(ImageSqeuencePath, "ImageSequence")
                imageFileLst.sort()
                timePerBatch = time.perf_counter()  # 1 inference 당 시간

                for i, imageFile in enumerate(imageFileLst):
                    print(f"[{i}/{len(imageFileLst)}] processing {imageFile} to {args.outputPath}/{imageFile.split('/')[-1]}")
                    inferenceSingle(
                        imageFile,
                        args.inferencePresetName,
                        model=model,
                        outputType="FILE",
                        outputPath=f"{imageSquenceResultFolder}{imageFile.split('/')[-1]}",
                        originalSave = False,
                    )

                _calRuntime(timePerBatch, len(fileList))

                ### video file generation ###
                files = makeImageSequenceFileList(imageSquenceResultFolder)
                videoPathList = imagesToVideo(files, imageSquenceResultFolder, "mp4", "mp4v", 30)
                print(f'Video directory path list: {videoPathList}')

        print("FINISHED!")

    # input : file
    else:
        # file : image
        if args.inputPath.split(".")[-1].lower() in EXT_DICT['Image']:
            inferenceSingle(args.inputPath, args.inferencePresetName, model=model, outputType="FILE", outputPath=args.outputPath)
        # file : video
        elif args.inputPath.split(".")[-1].lower() in EXT_DICT['Video']:
            imageSquenceResultFolder = args.outputPath + "/" + os.path.splitext(args.inputPath)[0].split('/')[-1] + "_results/"
            if os.path.exists(imageSquenceResultFolder) is False:
                os.makedirs(imageSquenceResultFolder)

            videofiles = makeVideoFileList(args.inputPath)
            ImageSqeuencePathList = videoToImages(videofiles, args.outputPath)
            imageFileLst = _generateFileList(ImageSqeuencePathList[0], "ImageSequence")
            imageFileLst.sort()

            timePerBatch = time.perf_counter()  # 1 inference 당 시간   
            
            for i, imageFile in enumerate(imageFileLst):
                print(f"[{i}/{len(imageFileLst)}] processing {imageFile} to {args.outputPath}/{imageFile.split('/')[-1]}")
                inferenceSingle(
                    imageFile,
                    args.inferencePresetName,
                    model=model,
                    outputType="FILE",
                    outputPath=f"{imageSquenceResultFolder}{imageFile.split('/')[-1]}",
                    originalSave = False,
                )
            
            _calRuntime(timePerBatch, len(imageFileLst))

            ### video file generation ###
            files = makeImageSequenceFileList(imageSquenceResultFolder)
            videoPathList = imagesToVideo(files, imageSquenceResultFolder, "mp4", "mp4v", 30)
            print(f'Video directory path list: {videoPathList}')
            print("FINISHED!")