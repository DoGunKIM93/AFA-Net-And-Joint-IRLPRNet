'''
param.py
'''
version = '1.33.200519'
 
                                                        # S I N G L E    I M A G E S #
# NAME          Provide Dataset Type    Scale Method (Scale Factor)                                                             Desc
#---------------------------------------------------------------------------------------------------------------------------------------------------
# DIV2K         train   valid           bicubic(x2,3,4,8), unknown(x2,3,4), mild(x4), wild(x4), difficult(x4), virtual(all)     < hard for training >
# 291           train                   virtual(all)                                                                            < good for training >
# Set5                          test    bicubic(x2,3,4,8), virftual(all)                                                         
# Set14                         test    bicubic(x2,3,4,8), virtual(all)                                                         
# Urban100                      test    bicubic(x2,3,4,8), virtual(all)                                                         
# Manga109                      test    bicubic(x2,3,4,8), virtual(all)                                                         < Cover of comic books >
# historical                    test    bicubic(x2,3,4,8), virtual(all)                                                                 
# BSDS100                       test    bicubic(x2,3,4,8), virtual(all)                                                         
# CelebA        train           test    virtual(all)                                                                            < Face Image Set >
# FFHQ-Face     train           test    virtual(all)                                                                            < Face Image Set >
# FFHQ-General  train           test    virtual(all)                                                                            < Face+Background Image Set >

                                                        # M U L T I P L E   I M A G E S #
# NAME          Provide Dataset Type    Scale Method (Scale Factor)                                                             Desc
#---------------------------------------------------------------------------------------------------------------------------------------------------
# REDS          train   valid   test    blur(all), blur_comp(all), virtual(all)                  
# Vid4                          test    bicubic(x4), virual(all)
# DiaDora       train           test    virual(all)


################ Hyper Parameters ################


# data Set
dataPath = '/home/projSR/dataset/'
scaleFactor = 4
colorMode = 'color' # 'color' || 'grayscale'
sequenceLength = 7 # Only for 'REDS' Dataset

### train
trainDataset = 'FFHQ-General' # See above table
trainDatasetType = 'train'  # 'train' || 'valid' || 'test'.  # See above table
trainScaleMethod = 'virtual' # See above table
batchSize = 25
samplingCount = 1 # Random Crop (samplingCount) per one Image.  Actual Batch Size = batchSize * samplingCount
cropSize = [224, 224]  # HR Image cropping size. LR Image size: cropSize / scaleFactor. None -> No crop
randomResizeMinMax = [1, 4]

### test
testDataset = 'FFHQ-General' # See above table
testDatasetType = 'test'  # 'train' || 'valid' || 'test'.  # See above table
testScaleMethod = 'virtual' # See above table
testMaxPixelCount = 590000 # 576*1024=589824 (테스트 시 이미지 비율은 유지하면서 총 픽셀 수가 이 값을 넘지 않게 리사이징)
#testSize = None #[576, 1024] # None || [H, W]    Resize when test



sameOutputSize = False

valueRangeType = '0~1' # '0~1' || '-1~1'

# model
NGF = 64
NDF = 64

pretrainedPath = '/home/projSR/dataset/pretrained/'

# train

MaxEpoch = 180000
learningRate = 0.0001
validStep = 500
trainAccidentCoef = None

schedulerPeriod = 300

mixedPrecision = False # Reduce memory size i.e,.

# save
archiveStep = 1

# GPU
GPUNum = '0' # 0~7

############################################
