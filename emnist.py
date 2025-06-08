dataset_path = "/home/minc/OCR_project/dataset/emnist/versions/3/"

import numpy as np
import pandas as pd
# import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1. 데이터 불러오기
print("Loading data...")
data = pd.read_csv(dataset_path + 'emnist-byclass-train.csv')
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
images = np.transpose(images, (0, 2, 1, 3))  # Transpose 필요 (Kaggle 기준)

# 2. 매핑 파일을 읽어서 대소문자 통합된 36개 클래스 만들기
def load_mapping(mapping_path):
    label_to_char = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            label, ascii_code = map(int, line.strip().split())
            char = chr(ascii_code).upper()  # 대소문자 통합
            label_to_char[label] = char
    return label_to_char

mapping = load_mapping(dataset_path + 'emnist-byclass-mapping.txt')

# 유일한 문자만 추출해 대소문자 통합된 클래스 인덱스 생성
unique_chars = sorted(set(mapping.values()))
char_to_index = {char: i for i, char in enumerate(unique_chars)}
print(f"총 클래스 수: {len(char_to_index)} → {char_to_index}")

# 라벨 변환
labels_mapped = np.array([char_to_index[mapping[l]] for l in labels])
labels_categorical = to_categorical(labels_mapped, num_classes=len(char_to_index))

# 3. 훈련/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.1, random_state=42)

# 4. 모델 정의 (Functional API)
input_layer = Input(shape=(28, 28, 1))
x = Conv2D(32, kernel_size=3, activation='relu')(input_layer)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(64, kernel_size=3, activation='relu')(x)
x = MaxPooling2D(pool_size=2)(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output_layer = Dense(len(char_to_index), activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# 5. 학습
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_val, y_val))

# 6. 모델 저장
model.save('emnist_byclass_cnn_model.h5')
print("모델이 emnist_byclass_cnn_model.h5로 저장되었습니다.")
