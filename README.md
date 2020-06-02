# 사용법

1. model.py에 모델을 추가
2. edit.py에서 ModelList class 내에 model.py에서 만든 모델을 추가
3. edit.py에 Optimizer 및 LR Scheduler(필수 아님) 추가
4. param.py에서 하이퍼패러미터 수정


# 실행시 주의사항 및 옵션 설명

1. --load, -l : save된 checkpoint 로드 (인자 없을 시 최신 모델 로드, -l 51 -> 51번째 모델 로드)
2. --test, -t : test 모드 (미구현)
3. --nosave, -n : epoch마다 validation 과정에 생기는 이미지를 가장 최근 이미지만 저장


# 하이퍼패러미터 설명

1. dataPath : 사용할 데이터셋 경로
1. trainDataset : 트레인에 사용할 데이터 셋
2. testDataset : 테스트에 사용할 데이터 셋
3. test/trainDatasetType : 데이터셋의 타입 (트레인/밸리드/테스트)
3. test/trainScaleMethod : 학습 데이터셋의 리사이징 방법 (virtual의 경우 software적으로 bicubic interpolation 수행)
4. batchSize : 배치 사이즈 (실제 배치사이즈 = batchSize * samplingCount)
5. samplingCount : 한 이미지에서 몇 장 랜덤크롭 할 것인가
6. cropSize : 학습시 사용할 이미지의 크롭 사이즈 (H, W)
7. scaleFactor : 몇 배 Super Resolution 할 것인가
8. colorMode : 흑백 및 컬러
9. sequenceLength : 비디오 데이터 셋에서 인풋으로 한번에 들어갈 프레임 갯수
9. NGF 및 NDF : 모델 channel width에 사용
10. MaxEpoch : 최대 에폭 횟수
11. learningRate : Optimizer Learning Rate에 사용
12. sameOutputSize : 데이터로더가 아웃풋과 동일한 사이즈의 인풋을 반환 하는지 (ex) scaleFactor 4에서 1/4크기의 이미지를 반환하는지, 1/1 크기의 이미지를 반환하는지)
13. archiveStep : 모델을 따로 저장하는 스텝 수
14. valueRangeType : 데이터로더에서 뱉는 이미지의 텐서 범위 ('0~1', '-1~1')
15. pretrainedPath : 프리트레인 모델 기본 경로
16. mixedPrecision : 믹스드 프리시전 (속도 빨라지고 메모리 덜 먹으나 성능 하락할 수 있음)

# 수정내역

20200602

1. 컬러 모드에서 그레이스케일 이미지가 들어갈 경우 에러 해결 (SET14, etc...)

20200529

1. GAN 관련 업데이트
2. 여러 개의 loss Function 반환하도록 trainStep 조정
3. inferenceStep 작동 안함

20200427-1

1. 저장 시 pth 파일로 저장하도록 변경

20200427

1. Pretrained 모델 로드시 버그 수정
2. 기타 버그 수정

20200423

1. Mixed Precision 지원
2. Pretrained 모델 및 non trainable 모델 지원
3. 기타 업데이트
4. 코드 리팩토링

Regacy

1. subversion 별로 model 저장 ./data/$version/model/$subversion/
2. validation images 경로 변경 ./data/$version/eval/$subversion/


# TODO

1. Test 기능 구현



# 버그추가 및 기능제보 문의 : New Biz. Ut. 김진