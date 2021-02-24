'''

MP4 Video to Image Sequence (file)

Usage ::: VideoToimageSequence.py -i [Video Folder Path (relative)] -o [output video path] -f [fps]

::: by DGK :::

'''

import cv2
import numpy as np
import os
import argparse
import re

from os.path import isfile, join, splitext

def tryint(s):
    try:
        return int(s)
    except:
        return s


parser = argparse.ArgumentParser()
parser.add_argument('--inputPath', '-i', help="입력할 비디오 파일 또는 폴더 (상대경로)")
parser.add_argument('--outputPath', '-o', help="출력할 영상 시퀀스 파일 이름 (상대경로)")

args = parser.parse_args()

files = []

if os.path.isfile(args.inputPath) == True:
    pathInList = args.inputPath
    files.append(pathInList)
else:
    pathInList = [x if x.endswith('/') else x + '/' for x in args.inputPath.split(' ')]
    
    for pathIn in pathInList:
        tmp = [pathIn + f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
        tmp.sort(key=lambda s: [ tryint(c) for c in re.split('([0-9]+)', s) ])
        files = tmp#for sorting the file names properly

pathOut = args.outputPath
count = 0

for filename in files:
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
        ret,frame = cam.read() 

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
    
    #print(f'processing Images... {(count+1)/len(files)*100:.1f}%')
    print(f'making Frames... {(count+1)/len(files)*100:.1f}%')
    count = count + 1

print("FINISHED!")