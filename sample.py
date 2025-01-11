# Keras 라이브러리 임포트
import keras
# Keras에서 layers와 models 모듈 임포트
from keras import layers, models

# Sequential 모델 정의 (층을 순차적으로 쌓는 방식)
model = models.Sequential([
    # 첫 번째 Dense 층
    # - 64개의 뉴런
    # - ReLU 활성화 함수 사용
    # - 입력 형태는 784 차원 (28x28 이미지를 펼친 형태)
    layers.Dense(64, activation='relu', input_shape=(784,)),
    
    # 출력 층
    # - 10개의 뉴런 (0-9 숫자 분류)
    # - softmax 활성화 함수로 각 클래스의 확률 출력
    layers.Dense(10, activation='softmax')
])

# 모델 컴파일
# - optimizer: 'adam' 옵티마이저 사용
# - loss: 희소 범주형 교차 엔트로피 손실 함수
# - metrics: 정확도를 평가 지표로 사용
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])