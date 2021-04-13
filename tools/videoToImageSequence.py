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

def videoToImages(fileList: list, pathOut: str) -> List[str]:

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

            if ret: 
                # if video is still left continue creating images 
                currentframe_zero = str(currentframe).zfill(8)
                name = pathOut+'/'+ subPath + '/' + currentframe_zero + '.png'
                #print ('Creating...' + name) 
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

    return ImageSqeuencePathList


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', '-i', type=str, help="입력할 비디오 파일 또는 폴더 (절대/상대경로)")
    parser.add_argument('--outputPath', '-o', type=str, help="출력할 영상 시퀀스 파일 이름 (절대/상대경로)")

    args = parser.parse_args()
    
    files = makeVideoFileList(args.inputPath)
    ImageSqeuencePathList = videoToImages(files, args.outputPath)
    print(f'Image Sqeuence directory path list: {ImageSqeuencePathList}')
    print("FINISHED!")

    