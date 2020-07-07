'''
param.py
'''
version = '1.34.200703'
 
                                                        # S I N G L E    I M A G E S #
# NAME          Provide Dataset Type                Scale Method (Scale Factor)                                                             Desc
#---------------------------------------------------------------------------------------------------------------------------------------------------
# DIV2K         train   valid                       bicubic(x2,3,4,8), unknown(x2,3,4), mild(x4), wild(x4), difficult(x4), virtual(all)     < hard for training >
# 291           train                               virtual(all)                                                                            < good for training >
# Set5                          test    inference   bicubic(x2,3,4,8), virftual(all)                                                         
# Set14                         test                bicubic(x2,3,4,8), virtual(all)                                                         
# Urban100                      test    inference   bicubic(x2,3,4,8), virtual(all)                                                         
# Manga109                      test                bicubic(x2,3,4,8), virtual(all)                                                         < Cover of comic books >
# historical                    test                bicubic(x2,3,4,8), virtual(all)                                                                 
# BSDS100                       test                bicubic(x2,3,4,8), virtual(all)                                                         
# CelebA        train           test    inference   virtual(all)                                                                            < Face Image Set >
# FFHQ-Face     train           test    inference   virtual(all)                                                                            < Face Image Set >
# FFHQ-General  train           test    inference   virtual(all)                                                                            < Face+Background Image Set >

                                                        # M U L T I P L E   I M A G E S #
# NAME          Provide Dataset Type                Scale Method (Scale Factor)                                                             Desc
#---------------------------------------------------------------------------------------------------------------------------------------------------
# REDS          train   valid   test                blur(all), blur_comp(all), virtual(all)                  
# Vid4                          test                bicubic(x4), virual(all)
# DiaDora       train           test                virual(all)

                                                        # C U S T O M   I M A G E S #
# NAME          Provide Dataset Type                Scale Method (Scale Factor)                                                             Desc
#---------------------------------------------------------------------------------------------------------------------------------------------------
# CUSTOM                                inference   virtual(all)                                                                            < custom for inference >


################ Hyper Parameters ################


# data Set
dataPath = '/home/projSR/dataset/'
scaleFactor = 2
colorMode = 'color' # 'color' || 'grayscale'
sequenceLength = 7 # Only for 'REDS' Dataset

### train
trainDataset = 'DIV2K' # See above table
trainDatasetType = 'train'  # 'train' || 'valid' || 'test'.  # See above table
trainScaleMethod = 'virtual' # See above table
batchSize = 30
samplingCount = 1 # Random Crop (samplingCount) per one Image.  Actual Batch Size = batchSize * samplingCount
cropSize = [256, 256]  # HR Image cropping size. LR Image size: cropSize / scaleFactor. None -> No crop
randomResizeMinMax = [1, 1]

### test
testDataset = 'Set5' # Set5 # See above table
testDatasetType = 'test'  # 'train' || 'valid' || 'test'.  # See above table
testScaleMethod = 'virtual' # See above table
testMaxPixelCount = 590000 # 576*1024=589824 (테스트 시 이미지 비율은 유지하면서 총 픽셀 수가 이 값을 넘지 않게 리사이징)
#testSize = None #[576, 1024] # None || [H, W]    Resize when test

###### Set14 historical  -> HR폴더에 1채널 데이터 LR 폴더에 3채널 데이터의 paired data가 존재. 
# valueRangeType = '-1~1' 일 경우, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))로 인해 발생
# 1채널로 들어오면 transforms.Normalize(0.5, 0.5)로 분기 처리하던가 미리 3채널로 바꿔주는 작업 필요.
###### Manga109  BSDS100 -> LR 후 SR시 기존 HR과 height width이 다름. 차후 수정 필요 
###### benchmark dataset의 경우 bicubic, sameOutputSize = Flase로 해야 함.
inferenceDataset = 'CUSTOM' # See above table
customPath = 'CUSTOM/blendingTest' # customPath Setting len으로 HR있는 것 없는 것 조건 주기()
inferenceDatasetType = 'inference'  # 'train' || 'valid' || 'test' || 'inference'.  # See above tablWe
inferenceScaleMethod = 'virtual' # See above table

# blending
## None, 'simpleBlending', 'gaussianBlending', 'possionBlending'
blendingMode = None

sameOutputSize = True

valueRangeType = '-1~1' # '0~1' || '-1~1'

# model
NGF = 32
NDF = 64

pretrainedPath = '/home/projSR/dataset/pretrained/'

# train

MaxEpoch = 180000
learningRate = 0.0001
validStep = 5000
trainAccidentCoef = None

schedulerPeriod = 300

mixedPrecision = False # Reduce memory size i.e,.

# save
archiveStep = 1

# GPU
GPUNum = '0' # 0~7

############################################
