'''
data_loader.py
'''
version = "1.58.200707"

#FROM Python LIBRARY
import os
import random
import math
import numpy as np
import time

from PIL import Image
from PIL import PngImagePlugin


#FROM PyTorch
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder


import param as p






PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


class JointImageDataset(Dataset):

    def __init__(self,
                DatasetList,
                mode, 
                shuffle):

        self.DatasetList = DatasetList
        self.shuffle = shuffle


        self.totalDatasetLength = 0
        for ds in self.DatasetList:
            print(self.DatasetList)
            if len(ds) >= self.totalDatasetLength:
                self.totalDatasetLength = len(ds)

        

        print(f'Joint Dataset Loaded: {len(self.DatasetList)}')
           


    def __getitem__(self, index):
        rstList = []
        for ds in self.DatasetList:
            elems = ds.__getitem__(index % len(ds))
            for elem in elems:
                rstList.append(elem)
        
        return rstList
        




    def __len__(self):
        return self.totalDatasetLength



class NonPairedSingleImageDataset(Dataset):
    def __init__(self,
                LRDatapath,
                cropTransform,
                commonTransform,
                LRTransform):

        self.imageType = 'Single'

        self.LRDatapath = LRDatapath
        self.cropTransform = cropTransform
        self.commonTransform = commonTransform
        self.LRTransform = LRTransform
        self.numDataLR = 0
        self.LRImageFileNames = []

        self.LRImageFileNames = os.listdir(self.LRDatapath)
        self.LRImageFileNames = [file for file in self.LRImageFileNames if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"))]
        self.numDataLR = len(self.LRImageFileNames)
        self.LRImageFileNames.sort()
        print('LR Image prepared : %d images'%self.numDataLR)

    def __getitem__(self, index):
        dot = "" 
        for i in range(index%5):
            dot += "."
        print(f"preprocessing{dot}     ", end="\r")
        LRImageOri = Image.open(os.path.join(self.LRDatapath, self.LRImageFileNames[index]))

        if self.cropTransform == None:
            Images = self.commonTransform(LRImageOri)
            LRImage = self.LRTransform(Images)

            if p.colorMode == 'grayscale' and LRImage.size(0) == 3:
                return [(LRImage[0:1,:,:] + LRImage[1:2,:,:] + LRImage[2:3,:,:]) / 3]
            elif p.colorMode == 'color' and LRImage.size(0) == 1:
                LRImage = torch.cat((LRImage,LRImage,LRImage),0)
                return [LRImage]
            else:
                return [LRImage]
        else:
            rst = []
            for LRImage in self.cropTransform(LRImageOri):
                Images = self.commonTransform(LRImage)
                LRImage = self.LRTransform(Images)

                if p.colorMode == 'grayscale' and LRImage.size(0) == 3:
                    rst.append([(LRImage[0:1,:,:] + LRImage[1:2,:,:] + LRImage[2:3,:,:]) / 3])
                elif p.colorMode == 'color' and LRImage.size(0) == 1:
                    LRImage = torch.cat((LRImage,LRImage,LRImage),0)
                    rst.append([LRImage])
                else:
                    rst.append([LRImage])
            return rst  

    def __len__(self):
        return self.numDataLR


class SingleImageDataset(Dataset):
    def __init__(self,
                LRDatapath,
                HRDatapath,
                cropTransform,
                commonTransform,
                LRTransform,
                HRTransform):

        self.imageType = 'Single'

        self.LRDatapath = LRDatapath
        self.HRDatapath = HRDatapath
        self.cropTransform = cropTransform
        self.commonTransform = commonTransform
        self.LRTransform = LRTransform
        self.HRTransform = HRTransform

        self.numDataLR = 0
        self.numDataHR = 0
        self.LRImageFileNames = []
        self.HRImageFileNames = []


        self.LRImageFileNames = os.listdir(self.LRDatapath)
        self.LRImageFileNames = [file for file in self.LRImageFileNames if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"))]
        self.numDataLR = len(self.LRImageFileNames)
        self.LRImageFileNames.sort()
        print('LR Image prepared : %d images'%self.numDataLR)

        self.HRImageFileNames = os.listdir(self.HRDatapath)
        self.HRImageFileNames = [file for file in self.HRImageFileNames if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"))]
        self.numDataHR = len(self.HRImageFileNames)
        self.HRImageFileNames.sort()
        print('HR Image prepared : %d images'%self.numDataHR)
           

    def __getitem__(self, index):

        #a = time.perf_counter()

        dot = "" 
        for i in range(index%5):
            dot += "."
        print(f"preprocessing{dot}     ", end="\r")
        LRImageOri = Image.open(os.path.join(self.LRDatapath, self.LRImageFileNames[index]))
        HRImageOri = Image.open(os.path.join(self.HRDatapath, self.HRImageFileNames[index]))
        #b = time.perf_counter()
        if self.cropTransform == None:
            Images = self.commonTransform([LRImageOri, HRImageOri])
            LRImage = self.LRTransform(Images[0])
            HRImage = self.HRTransform(Images[1])

            if p.colorMode == 'grayscale' and HRImage.size(0) == 3:
                return [[(LRImage[0:1,:,:] + LRImage[1:2,:,:] + LRImage[2:3,:,:]) / 3, (HRImage[0:1,:,:] + HRImage[1:2,:,:] + HRImage[2:3,:,:]) / 3]]
            elif p.colorMode == 'color' and HRImage.size(0) == 1:
                HRImage = torch.cat((HRImage,HRImage,HRImage),0)
                return [[LRImage, HRImage]]
            else:
                return [[LRImage, HRImage]]
        else:
            rst = []
            c = time.perf_counter() 
            for LRImage, HRImage in self.cropTransform([LRImageOri, HRImageOri]):
                Images = self.commonTransform([LRImage, HRImage])
                LRImage = self.LRTransform(Images[0])
                HRImage = self.HRTransform(Images[1])
                if p.colorMode == 'grayscale' and HRImage.size(0) == 3:
                    rst.append([(LRImage[0:1,:,:] + LRImage[1:2,:,:] + LRImage[2:3,:,:]) / 3, (HRImage[0:1,:,:] + HRImage[1:2,:,:] + HRImage[2:3,:,:]) / 3])
                elif p.colorMode == 'color' and HRImage.size(0) == 1:
                    HRImage = torch.cat((HRImage,HRImage,HRImage),0)
                    rst.append([LRImage, HRImage])
                else:
                    rst.append([LRImage, HRImage])
            print(c - time.perf_counter())
            #print(b-a, time.perf_counter() - b)
            return rst  

    def __len__(self):
        return self.numDataLR


class MultiImageDataset(Dataset):
    def __init__(self,
                LRDatapath,
                HRDatapath,
                cropTransform,
                commonTransform,
                LRTransform,
                HRTransform,
                sequenceLength,
                isEval):

        self.imageType = 'Multi'

        self.LRDatapath = LRDatapath
        self.HRDatapath = HRDatapath
        self.cropTransform = cropTransform
        self.commonTransform = commonTransform
        self.LRTransform = LRTransform
        self.HRTransform = HRTransform
        self.sequenceLength = sequenceLength
        self.isEval = isEval

        self.numDataLR = 0
        self.numDataHR = 0
        self.LRImageFileNames = []
        self.HRImageFileNames = []
        self.seqlenList = []

        LRSubDirList = os.listdir(self.LRDatapath)
        LRSubDirList.sort()
        HRSubDirList = os.listdir(self.HRDatapath)
        HRSubDirList.sort()

        for sdir in LRSubDirList:
            self.LRImageFileNames.append(os.listdir(self.LRDatapath + sdir + "/"))
        for sdir in HRSubDirList:
            self.HRImageFileNames.append(os.listdir(self.HRDatapath + sdir + "/"))

        if self.isEval == False:
            self.numDataLR = len(self.LRImageFileNames)
            self.numDataHR = len(self.HRImageFileNames)
        else:
            self.numDataLR = 0
            self.numDataHR = 0

        for i, sdir in enumerate(self.LRImageFileNames):
            self.LRImageFileNames[i] = [file for file in sdir if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"))]
            self.LRImageFileNames[i].sort()
            if self.isEval : self.numDataLR += len(sdir)
            self.seqlenList.append(len(sdir))
        if self.isEval : print(f"LR Sequence prepared : {self.numDataLR} Images")
        else : print(f"LR Sequence prepared : {self.numDataLR} Sequences")
        
        for i, sdir in enumerate(self.HRImageFileNames):
            self.HRImageFileNames[i] = [file for file in sdir if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".bmp"))]
            self.HRImageFileNames[i].sort()
            if self.isEval : self.numDataHR += len(sdir)
        if self.isEval : print(f"HR Sequence prepared : {self.numDataHR} Images")
        else : print(f"HR Sequence prepared : {self.numDataHR} Sequences")
           

    def __getitem__(self, index):
        dot = "" 
        for i in range(index%5):
            dot += "."
        print(f"preprocessing{dot}     ", end="\r")

        if self.isEval == False:
            LRImageSeqFileName = self.LRImageFileNames[index]
            HRImageSeqFileName = self.HRImageFileNames[index]
            if self.isEval == True:
                seqStartIdx = 0
            else:
                seqStartIdx = random.randint(0, len(LRImageSeqFileName) - self.sequenceLength)
            seqEndIdx = seqStartIdx + self.sequenceLength

            ImagesSeqOri = []
            for i in range(seqStartIdx, seqEndIdx):
                ImagesSeqOri.append( [ Image.open(os.path.join(self.LRDatapath + str(index).zfill(3), LRImageSeqFileName[i])), Image.open(os.path.join(self.HRDatapath + str(index).zfill(3), HRImageSeqFileName[i])) ] )
        
        else:
            wendy = index
            for i, length in enumerate(self.seqlenList):
                if (wendy - length < 0):
                    break
                else:
                    wendy -= length
            folderIndex = i
            imageIndex = wendy

            ImagesSeqOri = []
            LRImageSeqFileName = self.LRImageFileNames[folderIndex]
            HRImageSeqFileName = self.HRImageFileNames[folderIndex]

            for i in range(imageIndex - self.sequenceLength//2, imageIndex + self.sequenceLength//2 + 1):
                
                if i < 0:
                    offset = 0
                elif i >= len(self.LRImageFileNames[folderIndex]):
                    offset = len(self.LRImageFileNames[folderIndex]) - 1
                else:
                    offset = i

                #print(index, folderIndex, offset, len(self.LRImageFileNames[folderIndex]), str(folderIndex).zfill(3) + "/" + LRImageSeqFileName[offset])

                ImagesSeqOri.append( [ Image.open(os.path.join(self.LRDatapath + str(folderIndex).zfill(3), LRImageSeqFileName[offset])), 
                                       Image.open(os.path.join(self.HRDatapath + str(folderIndex).zfill(3), HRImageSeqFileName[offset])) ] )


        if self.cropTransform == None:
            ImagesSeq = self.commonTransform(ImagesSeqOri)
            
            LRImageSeq = []
            HRImageSeq = []

            for LRImage, HRImage in ImagesSeq:

                HRH = HRImage.size(2)
                HRW = HRImage.size(3)
                LRH = LRImage.size(2)
                LRW = LRImage.size(3)

                HRR = HRH / HRW
                LRR = LRH / LRW

                HRrszSize = [math.floor(math.sqrt(p.testMaxPixelCount / HRR)), math.floor(math.sqrt(p.testMaxPixelCount / HRR) * HRR)]
                LRrszSize = [math.floor(math.sqrt(p.testMaxPixelCount / LRR)), math.floor(math.sqrt(p.testMaxPixelCount / LRR) * LRR)]

                LRImage = transforms.Resize(size=p.LRrszSize,interpolation=3)(LRImage)
                HRImage = transforms.Resize(size=p.HRrszSize,interpolation=3)(HRImage)
                LRImage = self.LRTransform(LRImage)
                HRImage = self.HRTransform(HRImage)
                 
                if HRImage.size(0) > 3:
                    LRImage = LRImage[0:3,:,:]
                    HRImage = HRImage[0:3,:,:]  
                if p.colorMode == 'grayscale' and HRImage.size(0) == 3:
                    LRImage = (LRImage[0:1,:,:] + LRImage[1:2,:,:] + LRImage[2:3,:,:]) / 3
                    HRImage = (HRImage[0:1,:,:] + HRImage[1:2,:,:] + HRImage[2:3,:,:]) / 3

                LRImageSeq.append(torch.unsqueeze(LRImage, 0))
                HRImageSeq.append(torch.unsqueeze(HRImage, 0))

            LRImageSeq = torch.cat(LRImageSeq, 0)
            HRImageSeq = torch.cat(HRImageSeq, 0)

            return [[LRImageSeq, HRImageSeq]]
                
        else:
            rst = []
            for ImagesSeq in self.cropTransform(ImagesSeqOri):
                Images = self.commonTransform(ImagesSeq)

                LRImageSeq = []
                HRImageSeq = []

                for LRImage, HRImage in ImagesSeq:
                    LRImage = transforms.Resize(size=p.trainSize,interpolation=3)(LRImage)
                    HRImage = transforms.Resize(size=p.trainSize,interpolation=3)(HRImage)
                    LRImage = self.LRTransform(LRImage)
                    HRImage = self.HRTransform(HRImage)

                    if HRImage.size(0) > 3:
                        LRImage = LRImage[0:3,:,:]
                        HRImage = HRImage[0:3,:,:]
                    if p.colorMode == 'grayscale' and HRImage.size(0) == 3:
                        LRImage = (LRImage[0:1,:,:] + LRImage[1:2,:,:] + LRImage[2:3,:,:]) / 3
                        HRImage = (HRImage[0:1,:,:] + HRImage[1:2,:,:] + HRImage[2:3,:,:]) / 3

                    LRImageSeq.append(torch.unsqueeze(LRImage, 0))
                    HRImageSeq.append(torch.unsqueeze(HRImage, 0))

                LRImageSeq = torch.cat(LRImageSeq, 0)
                HRImageSeq = torch.cat(HRImageSeq, 0)

                
                    
                rst.append( [LRImageSeq, HRImageSeq] )

            return rst  

    def __len__(self):
        return self.numDataLR






def SRDataset(dataset,
                datasetType,
                dataPath,
                scaleFactor,
                scaleMethod, # 'bicubic' || 'unknown' || 'mild' || 'wild' || 'difficult' || 'virtual' -> software
                batchSize,
                mode, # 'train' || 'valid' || 'test' || 'inference'
                cropSize=None, #[H,W] || None
                MaxPixelCount=None, #[H,W] || None
                colorMode='color', # 'color' || 'grayscale'
                sameOutputSize=False,
                samplingCount=p.samplingCount,
                num_workers=16,
                valueRangeType=p.valueRangeType
                ):
    """Build and return data set."""
    


    ############ datapath ##############

    LRDatapath = dataPath
    HRDatapath = dataPath


    if (dataset == 'DIV2K'):

        # mode
        if (datasetType == 'train') or (datasetType == 'valid'):

            LRDatapath += f'DIV2K/{datasetType}/'
            HRDatapath += f'DIV2K/{datasetType}/GT/'
            
        else:
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return
    
        # scale method
        if (scaleMethod == 'virtual' or scaleFactor == 1):
            LRDatapath += 'GT/'

        elif scaleMethod == 'bicubic':
            if not(scaleFactor == 2 or scaleFactor == 3 or scaleFactor == 4 or scaleFactor == 8) :
                print(f"data_loader.py :: ERROR : {dataset} {scaleMethod} support only X2, X3, X4, X8 Scale")
                return
            else:
                LRDatapath += f'{scaleMethod}/{str(scaleFactor)}/'

        elif scaleMethod == 'unknown':
            if not(scaleFactor == 2 or scaleFactor == 3 or scaleFactor == 4 ) :
                print(f"data_loader.py :: ERROR : {dataset} {scaleMethod} support only X2, X3, X4 Scale")
                return
            else:
                LRDatapath += f'{scaleMethod}/{str(scaleFactor)}/'

        elif (scaleMethod == 'mild' or
             scaleMethod == 'wild' or
             scaleMethod == 'difficult'):
            
            if scaleFactor != 4:
                print(f"data_loader.py :: ERROR : {dataset} {scaleMethod} support only X4 Scale")
                return
            else:
                LRDatapath += f'{scaleMethod}/{str(scaleFactor)}/'
        
        else:
            print(f"data_loader.py :: ERROR : {dataset} scaling method not found")
            return     
    
    elif (dataset == '291'):
        # mode
        LRDatapath += '291/'
        HRDatapath += '291/GT/'
    
        if (datasetType != 'train'):
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

        # scale method
        if (scaleMethod == 'virtual'):
            LRDatapath += 'GT/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return    

    elif (dataset == 'REDS'):

        LRDatapath += 'REDS/'
        HRDatapath += 'REDS/'

        if (datasetType == 'train'):
            LRDatapath += 'train/'
            HRDatapath += 'train/'
        elif (datasetType == 'valid'):
            LRDatapath += 'validation/'
            HRDatapath += 'validation/'
        elif (datasetType == 'test' or datasetType == 'inference'):
            LRDatapath += 'test/'
            HRDatapath += 'test/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

        if (scaleMethod == 'blur'):
            LRDatapath += 'blur/1/'
            HRDatapath += 'GT/'
        elif (scaleMethod == 'blur_comp'):
            LRDatapath += 'blur_comp/1/'
            HRDatapath += 'GT/'
        elif (scaleMethod == 'virtual'):
            if (datasetType == 'test' or datasetType == 'inference'):
                print(f"data_loader.py :: ERROR : {dataset}:{datasetType} doesn't provide \"virtual\" scaling method ")
                return   
            LRDatapath += 'GT/'
            HRDatapath += 'GT/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return   
    
    elif (dataset == 'Vid4'):
        LRDatapath += 'Vid4/test/'
        HRDatapath += 'Vid4/test/'

        if (datasetType != 'test' and datasetType != 'inference'):
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return
        
        if (scaleMethod == 'bicubic'):
            if (scaleFactor != 4):
                print(f"data_loader.py :: ERROR : {dataset} {scaleMethod} support only X4 Scale")
                return
            LRDatapath += 'bicubic/4/'
            HRDatapath += 'GT/'
        elif (scaleMethod == 'virtual'):
            LRDatapath += 'GT/'
            HRDatapath += 'GT/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} scaling method not found")
            return     

    elif (dataset == 'DiaDora'):
        LRDatapath += 'DiaDora/'
        HRDatapath += 'DiaDora/'

        if (datasetType != 'train' and datasetType != 'test' and datasetType != 'inference'):
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

        if (scaleMethod != 'virtual'):
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return    

        if (datasetType == 'train'):
            LRDatapath += 'train/GT/'
            HRDatapath += 'train/GT/'
        elif (datasetType == 'test' or datasetType == 'inference'):
            LRDatapath += 'test/GT/'
            HRDatapath += 'test/GT/'

    elif (dataset == 'CelebA'):

        if scaleMethod != 'virtual':
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return   
        else:
            LRDatapath += "CelebA/CelebA/Img/img_align_celeba_png.7z/"
            HRDatapath = LRDatapath

        if (datasetType == 'test' or datasetType == 'inference'):
            LRDatapath += 'test/GT/'
            HRDatapath += 'test/GT/'
        elif datasetType == 'train':
            LRDatapath += 'train/GT/'
            HRDatapath += 'train/GT/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

    elif (dataset == 'FFHQ-Face'):

        if scaleMethod != 'virtual':
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return   
        else:
            LRDatapath += "FFHQ/Face/"
            HRDatapath = LRDatapath

        if (datasetType == 'test' or datasetType == 'inference'):
            LRDatapath += 'test/GT/'
            HRDatapath += 'test/GT/'
        elif datasetType == 'train':
            LRDatapath += 'train/GT/'
            HRDatapath += 'train/GT/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

    elif (dataset == 'FFHQ-General'):

        if scaleMethod != 'virtual':
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return   
        else:
            LRDatapath += "FFHQ/General/"
            HRDatapath = LRDatapath

        if (datasetType == 'test' or datasetType == 'inference'):
            LRDatapath += 'test/GT/'
            HRDatapath += 'test/GT/'
        elif datasetType == 'train':
            LRDatapath += 'train/GT/'
            HRDatapath += 'train/GT/'
        else:
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

    elif (dataset == 'Set5' or
         dataset == 'Set14' or
         dataset == 'Urban100' or
         dataset == 'Manga109' or
         dataset == 'historical' or
         dataset == 'BSDS100'):
        LRDatapath += 'benchmark/' + dataset + '/test/'
        HRDatapath += 'benchmark/' + dataset + '/test/GT/'

        if (datasetType != 'test' and datasetType != 'inference'):
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

        if (scaleMethod == 'virtual' or scaleFactor == 1):
            LRDatapath += 'GT/'

        elif scaleMethod == 'bicubic':
            if not(scaleFactor == 2 or scaleFactor == 3 or scaleFactor == 4 or scaleFactor == 8) :
                print(f"data_loader.py :: ERROR : {dataset} {scaleMethod} support only X2, X3, X4, X8 Scale")
                return
            else:
                LRDatapath += scaleMethod + '/' + str(scaleFactor) + '/'
        
        else:
            print(f"data_loader.py :: ERROR : {dataset} scaling method not found")
            return
    
    elif (dataset == 'CUSTOM'):
        if (datasetType != 'test' and datasetType != 'inference'):
            print(f"data_loader.py :: ERROR : {dataset} doesn't provide {datasetType} dataset")
            return

        if scaleMethod != 'virtual':
            print(f"data_loader.py :: ERROR : {dataset} only provide \"virtual\" scaling method ")
            return 
        else:
            LRDatapath += p.customPath
            if not os.path.exists(LRDatapath):
                os.makedirs(LRDatapath)
    else:
        print("data_loader.py :: ERROR : Dataset not found")
        return

    ##################################


    ################TRANSFORMS##################

    transformList = []
    commonTransformList = []
    resizingSize = 0
    outputSize = 0



    cropTransform = None
    if mode == 'train':
        if cropSize != None:
            cropTransform = PairedRandomCrop(cropSize, samplingCount=samplingCount, ResizeMinMax=p.randomResizeMinMax)



    if mode == 'train':
        commonTransformList.append(PairedRandomFlip())
        commonTransformList.append(PairedRandomRotate())
    elif mode == 'valid':
        commonTransformList.append(PairedCenterCrop(cropSize))
    if scaleMethod != 'virtual' and sameOutputSize == True:
        commonTransformList.append(PairedSizeMatch())
    commonTransform = transforms.Compose(commonTransformList)



    LRTransformList = transformList.copy()

    
    if scaleMethod == 'virtual':
        if datasetType != 'inference':
            if sameOutputSize == True:
                LRTransformList.append(ResizeByScaleFactor(1/p.scaleFactor, same_outputSize=True))
            else:
                LRTransformList.append(ResizeByScaleFactor(1/p.scaleFactor, same_outputSize=False))
    
    LRTransformList.append(transforms.ToTensor())
    if valueRangeType == '0~1':
        pass
    elif valueRangeType == '-1~1':
        LRTransformList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    LRTransform = transforms.Compose(LRTransformList)


    
    HRTransformList = transformList.copy()
    
    HRTransformList.append(transforms.ToTensor())
    if valueRangeType == '0~1':
        pass
    elif valueRangeType == '-1~1':
        HRTransformList.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    HRTransform = transforms.Compose(HRTransformList)




    ##################################
    
    if (dataset == 'REDS' or dataset == 'Vid4' or dataset == 'DiaDora'):
        if mode == 'train':
            dataset = MultiImageDataset(LRDatapath, HRDatapath, cropTransform, commonTransform, LRTransform, HRTransform, p.sequenceLength, False)
        else:
            dataset = MultiImageDataset(LRDatapath, HRDatapath, cropTransform, commonTransform, LRTransform, HRTransform, p.sequenceLength, True)
    else:
        if (dataset == 'CUSTOM'):
            dataset = NonPairedSingleImageDataset(LRDatapath, cropTransform, commonTransform, LRTransform)
        else:
            dataset = SingleImageDataset(LRDatapath, HRDatapath, cropTransform, commonTransform, LRTransform, HRTransform)


    return dataset



def SRDataLoader(dataset,
                datasetType,
                dataPath,
                scaleFactor,
                scaleMethod, # 'bicubic' || 'unknown' || 'mild' || 'wild' || 'difficult' || 'virtual' -> software
                batchSize,
                mode, # 'train' || 'valid' || 'test'
                cropSize=None, #[H,W] || None
                MaxPixelCount=None, #[H,W] || None
                colorMode='color', # 'color' || 'grayscale'
                sameOutputSize=False,
                samplingCount=p.samplingCount,
                num_workers=16,
                valueRangeType=p.valueRangeType
                ):

    dataset = SRDataset(dataset,
                datasetType,
                dataPath,
                scaleFactor,
                scaleMethod,
                batchSize,
                mode,
                cropSize,
                MaxPixelCount,
                colorMode, 
                sameOutputSize,
                samplingCount,
                num_workers,
                valueRangeType)

    """Build and return data loader."""

    if mode == 'train':
        shuffle = True
    elif (mode == 'test' or mode == 'valid' or mode == 'inference'):
        shuffle = False

    #dataset = JointImageDataset([dataset, dataset], True, True)
    #print(len(dataset))
        
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batchSize,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader


class PairedRandomCrop(object):

    def __init__(self, outputSize, samplingCount, ResizeMinMax = None, ResizeBlurFactor = None):

        assert isinstance(outputSize, (int, tuple, list))
        self.samplingCount = samplingCount
        self.ResizeMinMax = ResizeMinMax
        self.ResizeBlurFactor = ResizeBlurFactor

        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
            self.outputSize = outputSize


    def __call__(self, sample):

        resizeFactor = random.uniform(self.ResizeMinMax[0], self.ResizeMinMax[1])
        
        cropSize = [self.outputSize[0], self.outputSize[1]] 
        if self.ResizeMinMax is not None:
            cropSize[0] = math.ceil(cropSize[0] * resizeFactor)
            cropSize[1] = math.ceil(cropSize[1] * resizeFactor)
            

        if isinstance(sample[0], (tuple, list)):
            LRImage = sample[0][0]
            HRImage = sample[0][1]

            ratio = LRImage.size[0] / HRImage.size[0]

            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]
             
            newHRHeight, newHRWidth = cropSize
            newLRHeight = math.ceil(cropSize[0] * ratio)
            newLRWidth = math.ceil(cropSize[1] * ratio)

 
            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'

            rst = []

            for i in range(self.samplingCount):
                
                if HRHeight == newHRHeight: 
                    topHR = 0
                else:
                    topHR = np.random.randint(0, HRHeight - newHRHeight)
                if HRWidth == newHRWidth: 
                    leftHR = 0
                else:
                    leftHR = np.random.randint(0, HRWidth - newHRWidth)

                topLR = math.ceil(topHR * ratio)
                leftLR = math.ceil(leftHR * ratio)

                frames = []
                for imagePair in sample:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]

                    LRImageSampled = LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight))
                    HRImageSampled = HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))

                    if self.ResizeMinMax is not None:
                        LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio), math.ceil(self.outputSize[1]  * ratio)), resample = Image.BICUBIC)
                        HRImageSampled = HRImageSampled.resize(self.outputSize, resample = Image.BICUBIC)

                        if self.ResizeBlurFactor is not None and resizeFactor > 1:
                            blurFactor = 1 / math.sqrt(resizeFactor)

                            LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio * blurFactor), math.ceil(self.outputSize[1] * ratio * blurFactor)), resample = Image.BICUBIC)
                            HRImageSampled = HRImageSampled.resize((math.ceil(self.outputSize[0] * blurFactor), math.ceil(self.outputSize[1] * blurFactor)), resample = Image.BICUBIC)   

                            LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio), math.ceil(self.outputSize[1]  * ratio)), resample = Image.BICUBIC)
                            HRImageSampled = HRImageSampled.resize(self.outputSize, resample = Image.BICUBIC)    

                    frames.append([LRImageSampled, HRImageSampled])
                
                rst.append(frames)

            return rst
        else:
            LRImages = []
            HRImages = []

            LRImage = sample[0]
            HRImage = sample[1]

            ratio = LRImage.size[0] / HRImage.size[0]

            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]
            
            newHRHeight, newHRWidth = cropSize
            newLRHeight = math.ceil(cropSize[0] * ratio)
            newLRWidth = math.ceil(cropSize[1] * ratio)

            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'

            rst = []

            for i in range(self.samplingCount):
                
                if HRHeight == newHRHeight: 
                    topHR = 0
                else:
                    topHR = np.random.randint(0, HRHeight - newHRHeight)
                if HRWidth == newHRWidth: 
                    leftHR = 0
                else:
                    leftHR = np.random.randint(0, HRWidth - newHRWidth)

                topLR = math.ceil(topHR * ratio)
                leftLR = math.ceil(leftHR * ratio)

                LRImageSampled = LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight))
                HRImageSampled = HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))

                if self.ResizeMinMax is not None:
                    LRImageSampled = LRImageSampled.resize((math.ceil(self.outputSize[0] * ratio), math.ceil(self.outputSize[1]  * ratio)), resample = Image.BICUBIC)
                    HRImageSampled = HRImageSampled.resize(self.outputSize, resample = Image.BICUBIC)

                rst.append([LRImageSampled, HRImageSampled])

            return rst


class PairedCenterCrop(object):

    def __init__(self, outputSize):
        assert isinstance(outputSize, (int, tuple, list))
        if isinstance(outputSize, int):
            self.outputSize = (outputSize, outputSize)
        else:
            assert len(outputSize) == 2
            self.outputSize = outputSize

    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            LRImage = sample[0][0]
            HRImage = sample[0][1]

            ratio = LRImage.size[0] / HRImage.size[0]

            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]
            
            newHRHeight, newHRWidth = self.outputSize
            newLRHeight = math.ceil(self.outputSize[0] * ratio)
            newLRWidth = math.ceil(self.outputSize[1] * ratio)

            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'
            
            if HRHeight == newHRHeight: 
                topHR = 0
            else:
                topHR = HRHeight//2 - newHRHeight//2
            if HRWidth == newHRWidth: 
                leftHR = 0
            else:
                leftHR = HRWidth//2 - newHRWidth//2


            topLR = math.ceil(topHR * ratio)
            leftLR = math.ceil(leftHR * ratio)

            rst = []
            for imagePair in sample:
                LRImage = imagePair[0]
                HRImage = imagePair[1]

                rst.append([LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight)), 
                            HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))])
            

            return rst
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            ratio = LRImage.size[0] / HRImage.size[0]


            HRWidth, HRHeight = HRImage.size[:2]
            LRWidth, LRHeight = LRImage.size[:2]

            newHRHeight, newHRWidth = self.outputSize
            newLRHeight = math.ceil(self.outputSize[0] * ratio)
            newLRWidth = math.ceil(self.outputSize[1] * ratio)

            assert HRHeight >= newHRHeight and HRWidth >= newHRWidth, 'ERROR :: data_loader.py : Crop Size must be smaller than image data Size'
            
            if HRHeight == newHRHeight: 
                topHR = 0
            else:
                topHR = HRHeight//2 - newHRHeight//2
            if HRWidth == newHRWidth: 
                leftHR = 0
            else:
                leftHR = HRWidth//2 - newHRWidth//2


            topLR = math.ceil(topHR * ratio)
            leftLR = math.ceil(leftHR * ratio)

            LRImage = LRImage.crop((leftLR, topLR, leftLR + newLRWidth, topLR + newLRHeight))
            
            HRImage = HRImage.crop((leftHR, topHR, leftHR + newHRWidth, topHR + newHRHeight))

            return [LRImage, HRImage]


class PairedRandomFlip(object):

    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            if random.random() > 0.5:
                frames = []
                for imagePair in sample:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]
                    frames.append([LRImage.transpose(Image.FLIP_LEFT_RIGHT), HRImage.transpose(Image.FLIP_LEFT_RIGHT)])
            else:
                frames = sample

            if random.random() > 0.5:
                rst = []
                for imagePair in frames:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]
                    rst.append([LRImage.transpose(Image.FLIP_TOP_BOTTOM), HRImage.transpose(Image.FLIP_TOP_BOTTOM)])
            else:
                rst = frames

            return rst
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            if random.random() > 0.5:
                LRImage = LRImage.transpose(Image.FLIP_LEFT_RIGHT)
                HRImage = HRImage.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() > 0.5:
                LRImage = LRImage.transpose(Image.FLIP_TOP_BOTTOM)
                HRImage = HRImage.transpose(Image.FLIP_TOP_BOTTOM)

            return [LRImage, HRImage]


class PairedRandomRotate(object):

    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            if random.random() > 0.5:
                frames = []
                for imagePair in sample:
                    LRImage = imagePair[0]
                    HRImage = imagePair[1]
                    frames.append([LRImage.rotate(90, expand=True), HRImage.rotate(90, expand=True)])
            else:
                frames = sample

            return frames
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            if random.random() > 0.5:
                LRImage = LRImage.rotate(90, expand=True)
                HRImage = HRImage.rotate(90, expand=True)

            return [LRImage, HRImage]


class ResizeByScaleFactor(object):

    def __init__(self, scale_factor, same_outputSize=False):
        self.scale_factor = scale_factor
        self.same_outputSize = same_outputSize

    def __call__(self, sample):
        if isinstance(sample, (tuple, list)):
            W = sample[0].size[0]
            H = sample[0].size[1]
            rW = math.ceil(sample[0].size[0] * self.scale_factor)
            rH = math.ceil(sample[0].size[1] * self.scale_factor)

            rst = []

            for img in sample:
            
                rszimg = img.resize((rW,rH), resample = Image.BICUBIC)

                if(self.same_outputSize == True):
                    rszimg = rszimg.resize((W,H), resample = Image.BICUBIC)

                rst.append(rszimg)

            return rst
        else:
            W = sample.size[0]
            H = sample.size[1]
            rW = math.ceil(sample.size[0] * self.scale_factor)
            rH = math.ceil(sample.size[1] * self.scale_factor)
            
            sample = sample.resize((rW,rH), resample = Image.BICUBIC)

            if(self.same_outputSize == True):
                sample = sample.resize((W,H), resample = Image.BICUBIC)

            return sample


class testTransform(object):

    def __init__(self, txt):
        self.txt = txt

    def __call__(self, sample):
        print(self.txt, sample)

        return sample

class PairedSizeMatch(object):
    '''
    sample[0] 을 [1] 사이즈로 리사이즈 합니다 
    '''
    def __call__(self, sample):
        if isinstance(sample[0], (tuple, list)):
            rst = []
            for imagePair in sample:
                LRImage = imagePair[0]
                HRImage = imagePair[1]

                rst.append([LRImage.resize(HRImage.size[:2]), HRImage])

            return rst
        else:
            LRImage = sample[0]
            HRImage = sample[1]

            LRImage = LRImage.resize(HRImage.size[:2])

            return [LRImage, HRImage]


