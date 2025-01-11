from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# MNIST 데이터셋 불러오기
# 훈련용 60,000장, 테스트용 10,000장의 손글씨 이미지를 로드
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리
# 이미지 데이터를 4차원 텐서로 reshape (samples, height, width, channels)
# CNN 입력을 위해 (28, 28) 형태의 2D 이미지를 (28, 28, 1) 형태의 3D 텐서로 변환
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
# 픽셀값을 0~1 범위로 정규화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 레이블을 원-핫 인코딩으로 변환
# 예: 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 향상된 CNN 모델 구성
# Sequential 모델을 사용하여 레이어를 순차적으로 쌓음
model = Sequential([
    # 첫 번째 컨볼루션 블록
    Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 64개의 3x3 필터로 특징 추출
    BatchNormalization(),  # 학습 안정화를 위한 배치 정규화
    Conv2D(64, (3, 3), activation='relu'),  # 추가적인 특징 추출
    BatchNormalization(),
    MaxPooling2D((2, 2)),  # 특징맵의 크기를 절반으로 줄임
    Dropout(0.25),  # 과적합 방지를 위해 25% 노드를 무작위로 비활성화
    
    # 두 번째 컨볼루션 블록
    Conv2D(128, (3, 3), activation='relu'),  # 128개의 필터로 더 복잡한 특징 추출
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # 완전연결층
    Flatten(),  # 특징맵을 1차원 벡터로 평탄화
    Dense(512, activation='relu'),  # 512개의 뉴런을 가진 완전연결층
    BatchNormalization(),
    Dropout(0.5),  # 과적합 방지를 위해 50% 노드를 무작위로 비활성화
    Dense(10, activation='softmax')  # 10개 클래스에 대한 확률 출력
])

# 모델 컴파일
# optimizer: 학습 최적화 알고리즘 선택
# loss: 손실 함수 설정 (다중 분류를 위한 categorical_crossentropy 사용)
# metrics: 평가 지표 설정
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 데이터 증강 설정
# 학습 데이터의 다양성을 높이기 위해 이미지 변형 적용
datagen = ImageDataGenerator(
    rotation_range=10,  # 이미지를 최대 10도까지 회전
    zoom_range=0.1,    # 이미지를 최대 10%까지 확대/축소
    width_shift_range=0.1,  # 이미지를 수평으로 최대 10%까지 이동
    height_shift_range=0.1  # 이미지를 수직으로 최대 10%까지 이동
)

# 콜백 설정
callbacks = [
    # 검증 손실이 5번 연속으로 개선되지 않으면 학습 조기 종료
    EarlyStopping(patience=5, restore_best_weights=True),
    # 검증 손실이 3번 연속으로 개선되지 않으면 학습률을 0.2배로 감소
    ReduceLROnPlateau(factor=0.2, patience=3)
]

# 모델 학습
# 데이터 증강을 적용하여 배치 크기 64로 30 에포크 동안 학습
model.fit(datagen.flow(x_train, y_train, batch_size=64),
          epochs=30,
          validation_data=(x_test, y_test),
          callbacks=callbacks)

# 모델 평가
# 테스트 데이터셋으로 모델의 최종 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'\nTest accuracy: {test_acc:.3f}')

# 모델 저장
# 학습된 모델을 HDF5 형식으로 저장
model.save('mnist_model.h5')
