'''

Save code & load code by version-subversion.

::: by JIN :::

'''


import os
import argparse
import sys
import datetime

from shutil import copyfile






def saveCode(mainPath, ver, subversion):
    subDirList = ['model', 'log', 'result', 'checkpoint']
    list(os.makedirs(f'{mainPath}/data/{ver}/{x}/{subversion}') for x in subDirList if not os.path.exists(f'{mainPath}/data/{ver}/{x}/{subversion}'))

    subDirUnderModelList = ['backbone']
    list(os.makedirs(f'{mainPath}/data/{ver}/model/{subversion}/{x}') for x in subDirUnderModelList if not os.path.exists(f'{mainPath}/data/{ver}/model/{subversion}/{x}'))

    list(copyfile(f'{mainPath}/{x}', f'{mainPath}/data/{ver}/model/{subversion}/{x}') for x in os.listdir(mainPath) if x.endswith('.py') or x.endswith('.yaml'))
    list(copyfile(f'{mainPath}/backbone/{x}', f'{mainPath}/data/{ver}/model/{subversion}/backbone/{x}') for x in os.listdir(f'{mainPath}/backbone') if x.endswith('.py') or x.endswith('.yaml'))


def backupCode(mainPath, ver, subversion):
    if not os.path.exists(f'{cwd}/backup/'):
        os.makedirs(f'{cwd}/backup/')

    curtime = str(datetime.datetime.now().today()).replace(" ","-")
    bud = f'{cwd}/backup/.{ver}.{subversion}.{curtime}'

    if not os.path.exists(bud):
        os.makedirs(bud)
    if not os.path.exists(f'{bud}/backbone'):
        os.makedirs(f'{bud}/backbone')

    list(copyfile(f'{mainPath}/{x}', f'{bud}/{x}') for x in os.listdir(mainPath) if x.endswith('.py') or x.endswith('.yaml'))
    list(copyfile(f'{mainPath}/backbone/{x}', f'{bud}/backbone/{x}') for x in os.listdir(f'{mainPath}/backbone/') if x.endswith('.py') or x.endswith('.yaml'))


def loadCode(mainPath, ver, subversion):
    list(copyfile(f'{mainPath}/data/{ver}/model/{subversion}/{x}', f'{mainPath}/{x}') for x in os.listdir({mainPath}/data/{ver}/model/{subversion}) if x.endswith('.py') or x.endswith('.yaml'))
    list(copyfile(f'{mainPath}/data/{ver}/model/{subversion}/backbone/{x}', f'{mainPath}/backbone/{x}') for x in os.listdir(f'{mainPath}/data/{ver}/model/{subversion}/backbone') if x.endswith('.py') or x.endswith('.yaml'))



def _ver(mainPath, mode):

    assert mode in ['get', 'set']

    cwde = os.path.join(mainPath, 'edit.py')

    f = open(cwde, 'r')
    lines = f.readlines()
    
    if mode == 'set'

    inVersionArea = False

    version = None
    subversion = None

    for i in range(len(lines)):

        curLine = lines[i][:-1]

        if inVersionArea is False:
            if curLine.find("# VERSION START") != -1:
                inVersionArea = True
        
        elif inVersionArea is True:
            if curLine.find("version") != -1:

                #find quote letter
                if curLine.find('"') != -1:
                    quote = '"'
                elif curLine.find("'") != -1:
                    quote = "'"
                else:
                    assert True

                #get version
                vsrnWord = curLine[curLine.find(quote) + 1:]
                vsrnWord = vsrnWord[:vsrnWord.find(quote)]

                if curLine.find("subversion") == -1:
                    version = vsrnWord
                elif curLine.find("subversion") != -1:
                    subversion = vsrnWord
            
            elif curLine.find("# VERSION END") != -1:
                break

    return version, subversion




parser = argparse.ArgumentParser()

parser.add_argument('--save', '-s', action='store_true', help="현재 코드를 현재 스크립트에 써있는 버전 폴더(data/.....)에 저장 (--save)")
parser.add_argument('--load', '-l', nargs=2, help="코드 로드 (--load [version] [subversion])")
parser.add_argument('--saveas', '-sa', nargs=2, help="코드 다른 버전에 저장 (스크립트 내에서 버전 자동 변경됨) (--saveas [version] [subversion])")

args = parser.parse_args()



cwd = os.getcwd()
if cwd.endswith('/tools'):
    cwd = cwd[:-6]

version, subversion = getver(cwd)



#backup scripts
backupCode(cwd, version, subversion)
print(f'Code Backuped !!!')



if args.save is True:
    saveCode(cwd, version, subversion)
    print(f'Code Saved to data/{version}/*/{subversion} !!!')


elif args.load is not None:
    loadCode(cwd, args.load[0], args.load[1])
    print(f'Code Loaded from data/{args.load[0]}/*/{args.load[1]} !!!')


elif args.saveas is not None:
    saveCode(cwd, args.saveas[0], args.saveas[1])
    print(f'Code Saved to data/{args.saveas[0]}/*/{args.saveas[1]} !!!')

