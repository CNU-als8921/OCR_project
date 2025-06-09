import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import sys

# 상위 디렉토리를 파이썬 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 모델 로드
model = load_model('models/emnist_byclass_cnn_model.h5')

# 라벨 매핑 로드
def load_mapping(mapping_path):
    label_to_char = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            label, ascii_code = map(int, line.strip().split())
            char = chr(ascii_code).upper()
            label_to_char[label] = char
    return label_to_char

mapping = load_mapping('dataset/emnist/versions/3/emnist-byclass-mapping.txt')
unique_chars = sorted(set(mapping.values()))
char_to_index = {char: i for i, char in enumerate(unique_chars)}
index_to_char = {i: char for char, i in char_to_index.items()}

# 이미지 전처리 함수
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array  # (1, 28, 28, 1)

# output_images 디렉토리의 모든 이미지에 대해 예측 수행
output_dir = 'output_images'
results = []
# 파일 이름순 정렬
for filename in sorted(os.listdir(output_dir)):
    if filename.endswith('.png'):
        img_path = os.path.join(output_dir, filename)
        processed_img = preprocess_image(img_path)
        
        # 예측
        predictions = model.predict(processed_img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        predicted_char = index_to_char[predicted_class]
        results.append((filename, predicted_char, confidence))

# 모든 결과를 마지막에 출력
print("\n=== CNN 모델 예측 결과 ===")
for filename, predicted_char, confidence in results:
    print(f"파일: {filename}, 예측: {predicted_char}, 신뢰도: {confidence:.2%}") 