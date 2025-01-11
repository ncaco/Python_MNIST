import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from keras.models import load_model

# 저장된 모델 불러오기
model = load_model('mnist_model.h5')

def preprocess_image(image_path):
    # 이미지 로드 및 전처리 개선
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)  # 필요한 경우 색상 반전
    
    # 노이즈 제거
    img = img.filter(ImageFilter.SMOOTH)
    
    # 대비 향상
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    
    # 크기 조정 및 패딩
    target_size = 20  # 숫자 크기
    img = img.resize((target_size, target_size), Image.LANCZOS)
    
    # 28x28 캔버스의 중앙에 위치시키기
    background = Image.new('L', (28, 28), 0)
    offset = ((28 - target_size) // 2, (28 - target_size) // 2)
    background.paste(img, offset)
    
    # 배열로 변환 및 정규화
    img_array = np.array(background)
    img_array = img_array.reshape(1, 28, 28, 1)
    img_array = img_array / 255.0
    
    return img_array

def predict_digit(image_path):
    # 전처리된 이미지 얻기
    img_array = preprocess_image(image_path)
    
    # 예측
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = prediction[0][digit] * 100
    
    return f"이 숫자는 {digit}입니다. (확률: {confidence:.2f}%)"

# 사용 예시
result = predict_digit('rand_number1.png')
print(result)
