# 사용법

1. model.py에 모델을 추가
2. edit.py에서 ModelList class 내에 model.py에서 만든 모델을 추가
3. edit.py에 Optimizer 및 LR Scheduler(필수 아님) 추가
4. param.py에서 하이퍼패러미터 수정


# 실행시 주의사항 및 옵션 설명

1. CUDA_VISIBLE_DEVICE=(사용할 GPU들 숫자 (ex)2,3)) python main.py (옵션) 
2. --load, -l : save된 checkpoint 로드 (인자 없을 시 최신 모델 로드, -l 51 -> 51번째 모델 로드)
3. --test, -t : test 모드 (미구현)
4. --nosave, -n : epoch마다 validation 과정에 생기는 이미지를 가장 최근 이미지만 저장


# 하이퍼패러미터 설명

1. trainDataset : 트레인에 사용할 데이터 셋
    - DIV2K : 'bicubic', 'unknown', 'mild', 'wild', 'difficult', 'virtual' 지원
    - 291 : 'virtual'만 지원 (이게 학습 잘됨)
2. testDataset : 테스트에 사용할 데이터 셋
    - 'bicubic' 및 'virtual' 지원
3. test/trainScaleMethod : 학습 데이터셋의 리사이징 방법 (virtual의 경우 software적으로 bicubic interpolation 수행)
4. batchSize : 배치 사이즈 (실제 배치사이즈 = batchSize * samplingCount)
5. samplingCount : 한 이미지에서 몇 장 랜덤크롭 할 것인가
6. cropSize : 학습시 사용할 이미지의 크롭 사이즈 (H, W)
7. scaleFactor : 몇 배 Super Resolution 할 것인가
8. colorMode : 흑백 및 컬러
9. NGF 및 NDF : 모델 channel width에 사용
10. MaxEpoch : 최대 에폭 횟수
11. learningRate : Optimizer Learning Rate에 사용
12. sameOutputSize : 데이터로더가 아웃풋과 동일한 사이즈의 인풋을 반환 하는지 (ex) scaleFactor 4에서 1/4크기의 이미지를 반환하는지, 1/1 크기의 이미지를 반환하는지)
13. archiveStep : 모델을 따로 저장하는 스텝 수

# 수정내역

1. subversion 별로 model 저장 ./data/$version/model/$subversion/
2. validation images 경로 변경 ./data/$version/eval/$subversion/


# TODO

1. Test 기능 구현
2. Multi Image Dataloader 구현



# 버그 및 기능추가 문의 : New Biz. Ut. 김진