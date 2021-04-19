'''

Video to Image Sequence 

Usage ::: videoToImageSequence.py -i [Video Folder(File) Path] -o [output image sequence path]
Ex) 
1. python videoToImageSequence.py -i /home/jovyan/dataset_military/dataset/100.5M_배_IR.mp4 -o /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/tools/Test
2. python videoToImageSequence.py -i /home/jovyan/dataset_military/dataset/ -o ./Test

::: by DGK :::

'''

import cv2
import numpy as np
import os
import argparse
import re
from typing import List, Dict, Tuple, Union, Optional

from os.path import isfile, join, splitext

def tryint(s):
    try:
        return int(s)
    except:
        return s

# inputPath to file list
def makeVideoFileList(inputPath: str) -> List[str]:

    fileList = []

    if os.path.isfile(inputPath):
        pathInList = inputPath
        fileList.append(pathInList)
    else:
        pathInList = [x if x.endswith('/') else x + '/' for x in inputPath.split(' ')]
        
        for pathIn in pathInList:
            tmp = [pathIn + f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
            tmp.sort(key=lambda s: [ tryint(c) for c in re.split('([0-9]+)', s) ])
            fileList = tmp #for sorting the file names properly
    
    return fileList

# video file read 말고 video binary로 받았을 때(or tensor / PIL) image binaries로 return
# return도 input type에 맞게 수정(video file path -> image files save and path, video tensor -> image tensors list)

# def _getType(x) -> str:

#     TYPEDICT = {
#         PngImageFile: "PIL",
#         JpegImageFile: "PIL",
#         PILImage.Image: "PIL",
#         BmpImageFile: "PIL",
#         MpoImageFile: "PIL",
#         GifImageFile: "PIL",
#         torch.Tensor: "TENSOR",
#         type(None): "NONE",
#     }

#     return TYPEDICT[type(x)]


# def _getSize(x) -> List[int]:  # C H W

#     if _getType(x) == "PIL":  # PIL Implemenataion
#         sz = x.size
#         sz = [len(x.getbands()), sz[1], sz[0]]

#     elif _getType(x) == "TENSOR":  # Tensor Implementation
#         sz = list(x.size())[-3:]

#     return sz

# x = x.numpy() # CHW
# x = np.moveaxis(x, 0, -1) #WHC
# x = np.round(x[:, :, ::-1].copy()*255) #BGR)

# encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
# result, x = cv2.imencode('.jpg', x, encode_param)
# x = cv2.imdecode(x,1)

# x = x[:, :, ::-1].copy() / 255 #RGB
# x = np.moveaxis(x, -1, 0) #WHC
# x = torch.tensor(x)

# # image bytes to tensor
# def transform_image(image_bytes):
#     from PIL import Image
    
#     my_transforms = transforms.Compose([transforms.ToTensor()])
#     image = Image.open(io.BytesIO(image_bytes))
    
#     return my_transforms(image).unsqueeze(0)[:,0:3:,:,:]


# # encoded image bytes to tensor
# def encodedTotensor_request(_encoded_image):
#     requestImage = base64.b64decode(_encoded_image)

#     with torch.no_grad():
#         requestImageTensor = transform_image(image_bytes=requestImage)
    
#     return requestImageTensor
    
# # 아래 것이랑 합치기
# def videoToImagesOtherType():
# # video tensor to image sequence tensor
#     base64_video_path = '/home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/100.5M_배_IR_after_base64.txt' # 100.5M_배_IR_before_base64.txt
#     import io
#     import base64
#     n_frames = 343
#     height = 2048
#     width = 2560
#     with open(base64_video_path, "rb") as video_encoded:
#         # data = video_encoded.read()
#         # print("video_encoded Test : ", data)
#         # video_decoded = base64.b64decode(data) 
#         # print("video_decoded", video_decoded)

#         # video_BytesIO = io.BytesIO(video_decoded)
#         # # print("video_decoded bytes", io.BytesIO(video_decoded))
#         # frame_array = np.frombuffer(video_BytesIO.getvalue(), dtype="int8")
#         # print(frame_array.shape)
        
#         # for i in range(n_frames):
#         #     bgr = np.frombuffer(base64.b64decode(video_encoded.read(height*width*3)), dtype=np.uint8) #.reshape((height, width))
#         #     print(bgr)
#         #     print(bgr.shape)

#         data = video_encoded.read()
#         video_decoded = base64.b64decode(data)
#         video_BytesIO = io.BytesIO(video_decoded)
#         print(video_BytesIO.getvalue())
#         # frame_array = np.frombuffer(video_BytesIO.getvalue(), dtype="int8")
#         # print(frame_array.shape)

#         # for i in range(n_frames):
#         #     bgr = np.frombuffer(base64.b64decode(video_encoded.read(height*width*3)), dtype=np.uint8) #.reshape((height, width))
#         #     print(bgr)
#         #     print(bgr.shape)


#     return print("test")    
        
def videoToImages(fileList: list, pathOut: str):
    frame_array = []
    count = 0
    ImageSqeuencePathList = []

    for filename in fileList:
        print("filename : ", filename)
        subPath = splitext(filename)[0].split('/')[-1]

        try: 
            # creating a folder named data 
            if not os.path.exists(pathOut+'/'+subPath): 
                os.makedirs(pathOut+'/'+subPath)
                print("pathOut : ", pathOut+'/'+subPath)
        # if not created then raise error 
        except OSError: 
            print ('Error: Creating directory of data') 
        
        cam = cv2.VideoCapture(filename)
        currentframe = 0

        while(True): 
            # reading from frame 
            ret, frame = cam.read() 
            # print(frame.shape) HWC
            # print(frame)
            if ret: 
                # if video is still left continue creating images 
                currentframe_zero = str(currentframe).zfill(8)
                name = pathOut+'/'+ subPath + '/' + currentframe_zero + '.png'
                #print ('Creating...' + name) 
                #inserting the frames into an image array
                frame_array.append(frame)
                # writing the extracted images (only english path is ok) 
                cv2.imwrite(name, frame) 
                # increasing counter so that it will 
                # show how many frames are created 
                currentframe += 1
            else: 
                break
        # Release all space and windows once done 
        cam.release()
        cv2.destroyAllWindows()          
        ImageSqeuencePathList.append(os.getcwd() + pathOut[1:]+'/'+subPath+'/') if pathOut[0] == '.' else ImageSqeuencePathList.append(pathOut+'/'+subPath+'/')

        #print(f'processing Images... {(count+1)/len(ImageSqeuencePathList)*100:.1f}%')
        print(f'making Frames... {(count+1)/len(ImageSqeuencePathList)*100:.1f}%')
        count = count + 1

    return ImageSqeuencePathList, frame_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', '-i', type=str, help="입력할 비디오 파일 또는 폴더 (절대/상대경로)")
    parser.add_argument('--outputPath', '-o', type=str, help="출력할 영상 시퀀스 파일 이름 (절대/상대경로)")

    args = parser.parse_args()
    
    files = makeVideoFileList(args.inputPath)
    ImageSqeuencePathList, _ = videoToImages(files, args.outputPath)
    # videoToImagesOtherType()
    
    # print(f'Image Sqeuence directory path list: {ImageSqeuencePathList}')
    print("FINISHED!")

    