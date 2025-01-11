# Python MNIST 손글씨 숫자 인식
MNIST 데이터셋을 활용한 딥러닝 기반 손글씨 숫자 인식 프로젝트입니다.

## 📋 프로젝트 소개
CNN(Convolutional Neural Network) 모델을 사용하여 손으로 쓴 0-9까지의 숫자를 인식하는 시스템을 구현했습니다. MNIST 데이터셋으로 학습된 이 모델은 새로운 손글씨 숫자 이미지에 대해 높은 정확도로 예측이 가능합니다.

## 🌟 주요 기능
- 손글씨 숫자 이미지 인식 및 분류
- 실시간 숫자 예측 및 확률 출력
- 이미지 전처리 및 품질 개선

## 🔧 설치 방법
필요한 패키지를 설치하기 위해 다음 명령어를 실행하세요:
``` bash
pip install tensorflow numpy pillow
```

## 📁 프로젝트 구조
- `MNIST_LOAD.py`: CNN 모델 학습 및 저장
- `Handwriting prediction.py`: 학습된 모델을 사용한 숫자 예측
- `mnist_model.h5`: 학습된 모델 파일 (학습 후 생성)

## 모델 구조
- 2개의 컨볼루션 블록 사용
- 각 블록: Conv2D, BatchNormalization, MaxPooling2D, Dropout 레이어로 구성
- 완전연결층: 512개 뉴런과 최종 10개 클래스 출력

## 학습 특징
- 데이터 증강 기법 적용 (회전, 확대/축소, 이동)
- Early Stopping과 Learning Rate 조절을 통한 최적화
- BatchNormalization과 Dropout을 통한 과적합 방지

## 사용 방법
1. 모델 학습
2. 학습된 모델을 사용하여 손글씨 숫자 예측

## 이미지 전처리 과정
- 노이즈 제거 및 대비 향상
- 이미지 크기 조정 (28x28)
- 정규화 및 중앙 정렬

## 요구사항
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pillow (PIL)

## 성능
- 테스트 데이터셋에 대한 높은 정확도 달성
- 실제 손글씨 이미지에 대한 강건한 예측 성능
├── MNIST_LOAD.py # 모델 학습 및 저장
├── Handwriting prediction.py # 숫자 예측 실행
└── mnist_model.h5 # 학습된 모델 파일

## 🛠 기술 스택
- **언어**: Python 3.x
- **프레임워크**: TensorFlow/Keras
- **주요 라이브러리**:
  - NumPy: 데이터 처리
  - Pillow: 이미지 처리
  - Keras: 딥러닝 모델 구현

## 💻 모델 아키텍처
### CNN 구조
1. **첫 번째 컨볼루션 블록**
   - 64개 필터 (3x3)
   - BatchNormalization
   - MaxPooling
   - Dropout (25%)

2. **두 번째 컨볼루션 블록**
   - 128개 필터 (3x3)
   - BatchNormalization
   - MaxPooling
   - Dropout (25%)

3. **완전연결층**
   - 512 뉴런
   - 최종 10개 클래스 출력

## 🔧 실행 방법
1. **환경 설정**
   ```bash
   pip install tensorflow numpy pillow
   ```

2. **모델 학습**
   ```bash
   python MNIST_LOAD.py
   ```

3. **숫자 예측**
   ```bash
   python Handwriting prediction.py
   ```

## 📝 전처리 프로세스
1. **이미지 품질 개선**
   - 노이즈 제거
   - 대비 향상
   - 이미지 정규화

2. **크기 조정**
   - 28x28 픽셀로 통일
   - 중앙 정렬
   - 패딩 적용

## 🎯 성능 최적화
- **데이터 증강**
  - 회전: ±10도
  - 확대/축소: ±10%
  - 이미지 이동: 수평/수직 ±10%

- **학습 최적화**
  - Early Stopping
  - Learning Rate 동적 조절
  - BatchNormalization
  - Dropout을 통한 과적합 방지

## 📊 모델 성능
- 테스트 데이터셋 정확도: 99% 이상
- 실제 손글씨 이미지 인식률: 95% 이상

## 📌 참고사항
- 이미지는 흑백(grayscale) 형식만 지원
- 최적의 인식을 위해 깨끗한 손글씨 이미지 권장
- GPU 환경에서 더 빠른 학습 가능