'''

Image Sequence (file) to MP4 Video

Usage ::: imageSequenceToVideo.py -i [imageSequence Folder Path (relative)] -o [output video path] -e [video file extension] -c [video codec] -f [fps]
Ex) 
1. python imageSequenceToVideo.py -i ./Test/100.5M_배_IR/ -o /home/jovyan/data-vol-1/dgk/git/2021/sr-research-framework/tools/Test/ -e avi -c MJPG -f 30
2. python imageSequenceToVideo.py -i ./Test/ -o ./Test/ -e mp4 -c mp4v -f 30

::: by JIN & DGK :::

'''


import cv2
import numpy as np
import os
import argparse
import re
from typing import List, Dict, Tuple, Union, Optional
from os.path import isfile, join

EXT_DICT = {
            "Text" :             ['txt'],
            "Image" :            ['png','jpg','jpeg','gif','bmp'],
            "ImageSequence" :    ['png','jpg','jpeg','gif','bmp'],
            "Video" :            ['avi','mp4','mkv','wmv','mpg','mpeg'], 
            }

def tryint(s):
    try:
        return int(s)
    except:
        return s

def makeImageSequenceFileList(inputPath: str) -> List[str]:

    files = []
    # for support single folder & multiple folder
    pathInList = [inputPath if isfile(os.path.join(inputPath, filename)) else os.path.join(inputPath, filename)+'/' for filename in os.listdir(inputPath)]
    pathInSet = set(pathInList)
    pathInList = list(pathInSet)
    
    for pathIn in pathInList:
        tmp = [pathIn + f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        tmp.sort(key=lambda s: [ tryint(c) for c in re.split('([0-9]+)', s) ])
        files.append(tmp)#for sorting the file names properly

    return files

# imageSequence read 말고 imageSequence binaries로 받았을 때(or tensors / PILs)  video binary로 return
# return도 input type에 맞게 수정(image files path -> video file save and path, image tensors list -> video tensor)
def imagesToVideo(fileList: list, pathOut: str, extension: str, codec: str, fps: int) -> List[str]:
    
    assert extension in EXT_DICT['Video'], f"inference.py :: outputType '{args.extension}' is not supported. Supported types: 'avi', 'mp4', 'mkv', 'wmv', 'mpg', 'mpeg'"
    assert codec in [
        "mp4v",
        "MJPG",
        "DIVX",
        "XVID",
        "X264"
    ], f"inference.py :: outputType '{args.extension}' is not supported. Supported types: 'mp4v', 'MJPG', 'DIVX', 'XVID', 'X264'"

    videoPathList = []
    # tensor list
    for k in range(len(fileList)):
        frame_array = []
        videoFilePath = ''
        size = ()
        
        for i in range(len(fileList[k])):            
            filenameList = []
            
            if fileList[k][i] not in EXT_DICT["ImageSequence"]:
                pass

            filenameList.append(fileList[k][i])
            for j, filename in enumerate(filenameList):
                if j == 0:
                    img = cv2.imread(filename)
                else:
                    img = cv2.hconcat([img, cv2.imread(filename)])

            #reading each files
            height, width, layers = img.shape
            size = (width,height)
            #inserting the frames into an image array
            frame_array.append(img)

            print(f'processing Images... {(i+1)/len(fileList[k])*100:.1f}%')
                     
        videoFolderPath = pathOut + fileList[k][0].split('/')[-2]
        videoFilePath = videoFolderPath + '.' + extension
        
        out = cv2.VideoWriter(videoFilePath, cv2.VideoWriter_fourcc(*codec), fps=fps, frameSize=size)

        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])

            print(f'making Video... {(i+1)/len(frame_array)*100:.1f}%')

        out.release()
        videoPathList.append(os.getcwd() + videoFilePath[1:]) if videoFilePath[0] == '.' else videoPathList.append(videoFilePath)
        
        print(f':::\nTotal Frame: {len(frame_array)}\nFPS: {fps:.1f}\nVideo length: {len(frame_array) * 1/fps:.1f} seconds\nSize(H X W): {height} X {width}')
    
    return videoPathList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputPath', '-i', type=str, help="입력할 영상 시퀀스 파일들이 있는 폴더 (상대경로)")
    parser.add_argument('--outputPath', '-o', type=str, help="출력할 비디오 파일 경로 (상대경로)")
    parser.add_argument('--extension', '-e', type=str, default="mp4", help="동영상 확장자")
    parser.add_argument('--codec', '-c', type=str, default="mp4v", help="동영상 코덱")
    parser.add_argument('--fps', '-f', type=int, default=30, help="fps")

    args = parser.parse_args()


    files = makeImageSequenceFileList(args.inputPath)
    videoPathList = imagesToVideo(files, args.outputPath, args.extension, args.codec, int(args.fps))
    print(f'Video directory path list: {videoPathList}')
    print("FINISHED!")