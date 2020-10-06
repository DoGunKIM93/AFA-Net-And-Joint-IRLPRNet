'''

Image Sequence (file) to MP4 Video

Usage ::: imageSequenceToVideo.py -i [imageSequence Folder Path (relative)] -o [output video path] -f [fps]


::: by JIN :::

'''


import cv2
import numpy as np
import os
import argparse
import re

from os.path import isfile, join


def tryint(s):
    try:
        return int(s)
    except:
        return s


parser = argparse.ArgumentParser()

parser.add_argument('--inputPath', '-i', help="입력할 영상 시퀀스 파일들이 있는 폴더 (상대경로)")
parser.add_argument('--outputPath', '-o', help="출력할 비디오 파일 이름 (상대경로)")
parser.add_argument('--fps', '-f', help="fps")

args = parser.parse_args()


pathInList = [x if x.endswith('/') else x + '/' for x in args.inputPath.split(' ')]
pathOut = args.outputPath

#secPerFrame = float(int(args.secPerFrame.split('/')[0]) / int(args.secPerFrame.split('/')[1]))
fps = int(args.fps)

frame_array = []

files = []

for pathIn in pathInList:
    tmp = [pathIn + f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    tmp.sort(key=lambda s: [ tryint(c) for c in re.split('([0-9]+)', s) ])
    files.append(tmp)#for sorting the file names properly


for i in range(len(files[0])):
    
    filenameList = []
    for j in range(len(files)):
        filenameList.append(files[j][i])

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


    print(f'processing Images... {(i+1)/len(files[0])*100:.1f}%')

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*"mp4v"), fps=fps, frameSize=size)

print(len(frame_array))

for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])

    print(f'making Video... {(i+1)/len(frame_array)*100:.1f}%')

out.release()

print(f'FINISHED!\n:::\nTotal Frame: {len(frame_array)}\nFPS: {fps:.1f}\nVideo length: {len(frame_array) * 1/fps:.1f} seconds\nSize(H X W): {height} X {width}')