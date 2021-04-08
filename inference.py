"""
inference.py
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

from collections.abc import Iterable
from PIL import Image
from torchvision.utils import save_image
from backbone.config import Config
from torch.nn import DataParallel

# Read Augmentations from backbone.augmentation automatically
AUGMENTATION_DICT = dict(
    x for x in inspect.getmembers(augmentation) if (not x[0].startswith("_")) and (inspect.isfunction(x[1]))
)

# GPU Setting
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.inference.general.GPUNum)


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
    elif outputType == "NPARRAY":
        out = out.cpu().numpy()
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


def inferenceSingle(inp, inferencePresetName, model=None, outputType=None, outputPath=None):
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
        "NPARRAY",
        "PIL",
    ], f"inference.py :: outputType '{outputType}' is not supported. Supported types: 'FILE', 'TENSOR', 'NPARRAY', 'PIL'"
    assert outputType != "FILE" or (
        outputType == "FILE" and outputPath is not None
    ), "inference.py :: outputPath must be exist if outputType is 'FILE'"

    ##backend-server에서 어느 부분까지 커버할지, inference는 어느 부분까지 공용으로 사용할지 고려해서 data input 부분 정리하기 (bsae64, PIL, tensor 등)
    model_augmentation = Config.inferenceDict["model"][inferencePresetName]["augmentation"]
    model_valueRangeType = Config.inferenceDict["model"][inferencePresetName]["valueRangeType"]
    
    # augmentation for input image
    for augmentationFuc in model_augmentation:
        inp = _applyAugmentationFunction([inp], augmentationFuc)
        inp = inp[0]
    
    inp = inp.unsqueeze(0).cuda()
    inp = inp * 2 - 1 if model_valueRangeType == "-1~1" else torch.round(inp * 255) if model_valueRangeType == "0~255" else inp

    if model is None:
        # load Model
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
                valueRangeType=model_valueRangeType
            )
            for i, x in enumerate(out)
        ] if isinstance(out, tuple) or isinstance(out, list) else utils.saveImageTensorToFile(
            {"Result": out}, 
            outputPath, 
            caption=False, 
            valueRangeType=model_valueRangeType
        )
        # original save
        utils.saveImageTensorToFile(
            {"Result": inp},
            f'{outputPath.split(".")[0]}-original.{outputPath.split(".")[1]}',
            caption=False,
            valueRangeType=model_valueRangeType
        )
        print("Saved.")

        oldtimePerInference = timePerInference
        timePerInference = time.perf_counter()
        print(f"Time per inference: {(timePerInference - oldtimePerInference):.2f} sec                    ")

        return
    else:
        rst = _convertTensorTo(out, outputType)
        print("Finished.")
        return rst


parser = argparse.ArgumentParser()

parser.add_argument("--inferencePresetName", "-n", help="인퍼런스 프리셋 이름")
parser.add_argument("--inputPath", "-i", help="입력할 이미지 파일 경로(단일영상) or 폴더 경로(다중영상)")
parser.add_argument("--outputPath", "-o", help="출력할 이미지 파일 경로(단일영상) or 비디오 파일 경로(다중영상)")
parser.add_argument("--fps", "-f", default="24", help="fps (video only)")

args = parser.parse_args()

print("Load Model...")
model = _loadModel(args.inferencePresetName)


if os.path.isdir(args.inputPath):

    if os.path.exists(args.outputPath) is False:
        os.mkdir(args.outputPath)
    assert os.path.isdir(args.outputPath), "if inputPath is a dir, outputPath must be a dir"

    imageFileLst = [
        os.path.join(args.inputPath, f)
        for f in os.listdir(args.inputPath)
        if (
            os.path.isfile(os.path.join(args.inputPath, f))
            and f.split(".")[-1].lower() in ["png", "jpg", "jpeg", "gif", "bmp"]
        )
    ]

    imageFileLst.sort()

    for i, imageFile in enumerate(imageFileLst):
        print(f"[{i}/{len(imageFileLst)}] processing {imageFile} to {args.outputPath}/{imageFile.split('/')[-1]}")
        inferenceSingle(
            imageFile,
            args.inferencePresetName,
            model=model,
            outputType="FILE",
            outputPath=f"{args.outputPath}/{imageFile.split('/')[-1]}",
        )

else:
    inferenceSingle(args.inputPath, args.inferencePresetName, model=model, outputType="FILE", outputPath=args.outputPath)
