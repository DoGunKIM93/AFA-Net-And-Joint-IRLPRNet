"""
inference.py
"""
version = "1.02.211008"


# READ YAML
# CONSTRUCT MODEL
# PASS DATA

import os
import argparse
import torch
import torchvision.transforms
import backbone.augmentation as augmentation
import backbone.predefined
import backbone.utils as utils

from collections.abc import Iterable
from PIL import Image
from torchvision.utils import save_image
from backbone.config import Config
from torch.nn import DataParallel

# GPU 지정
os.environ["CUDA_VISIBLE_DEVICES"] = str(Config.param.general.GPUNum)


def _loadModel(inferencePresetName):

    # import module by string
    module = getattr(backbone.predefined, Config.inferenceDict[inferencePresetName]["model"])

    # load model network with params
    paramList = str(Config.inferenceDict[inferencePresetName]["param"]).replace(" ", "").split(",")
    model = module(
        *list(
            (x if x.replace(".", "", 1).isdigit() is False else (int(x) if x.find(".") is -1 else float(x)))
            for x in paramList
        )
    )
    model.cuda()
    model = DataParallel(model)

    # load checkpoint weight
    checkpoint = torch.load(Config.param.data.path.pretrainedPath + Config.inferenceDict[inferencePresetName]["weight"])
    # model.load_state_dict(checkpoint['model'],strict=True)
    try:
        mthd = "NORMAL"
        model.load_state_dict(checkpoint["model"], strict=True)
    except:
        try:
            mthd = "GLOBAL STRUCTURE"
            model.load_state_dict(checkpoint, strict=True)
        except:
            try:
                mthd = "INNER MODEL GLOBAL STRUCTURE"
                model.module.load_state_dict(checkpoint, strict=True)
            except:
                try:
                    mthd = "UNSTRICT (WARNING : load weights imperfectly)"
                    model.load_state_dict(checkpoint["model"], strict=False)
                except:
                    try:
                        mthd = "GLOBAL STRUCTURE UNSTRICT (WARNING : load weights imperfectly)"
                        model.load_state_dict(checkpoint, strict=False)
                    except:
                        mthd = "FAILED"
                        print("inference.py :: model load failed..... I'm sorry~")

    model.eval()
    print(
        f"{Config.inferenceDict[inferencePresetName]['model']} Loaded with {mthd} mode."
        if mthd != "FAILED"
        else f"{Config.inferenceDict[inferencePresetName]['model']} Load Failed."
    )
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
    inp = augmentation.toTensor([inp])[0].unsqueeze(0).cuda()
    inp = inp * 2 - 1 if Config.inferenceDict[inferencePresetName]["valueRangeType"] == "-1~1" else inp

    if model is None:
        # load Model
        print("Load Model...")
        model = _loadModel(inferencePresetName)

    # inference
    print("Inferencing...")
    out = _modelInference(inp, model)

    # convert to output
    if outputType == "FILE":
        [
            utils.saveImageTensorToFile(
                {"Result": x}, f'{outputPath.split(".")[0]}-{i}.{outputPath.split(".")[1]}', caption=False
            )
            for i, x in enumerate(out)
        ] if isinstance(out, tuple) or isinstance(out, list) else utils.saveImageTensorToFile(
            {"Result": out}, outputPath, caption=False
        )
        utils.saveImageTensorToFile(
            {"Result": inp},
            f'{outputPath.split(".")[0]}-original.{outputPath.split(".")[1]}',
            caption=False,
            valueRangeType=Config.inferenceDict[inferencePresetName]["valueRangeType"],
        )
        print("Saved.")
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
