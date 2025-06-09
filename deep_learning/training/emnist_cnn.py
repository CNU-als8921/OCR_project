import numpy as np
import os
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.regularizers import l2

# 데이터 경로
project_root = "/home/minc/OCR_project/"
dataset_path = os.path.join(project_root, 'deep_learning/data/emnist/')

# 데이터 로드
print("Loading data...")
data = pd.read_csv(os.path.join(dataset_path, 'emnist-byclass-train.csv'))
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values.reshape(-1, 28, 28).astype('float32') / 255.0
images = np.transpose(images, (0, 2, 1))  # transpose

# 라벨 매핑
def load_mapping(mapping_path):
    label_to_char = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            label, ascii_code = map(int, line.strip().split())
            char = chr(ascii_code).upper()
            label_to_char[label] = char
    return label_to_char

mapping = load_mapping(os.path.join(dataset_path, 'emnist-byclass-mapping.txt'))
unique_chars = sorted(set(mapping.values()))
char_to_index = {char: i for i, char in enumerate(unique_chars)}
labels_mapped = np.array([char_to_index[mapping[l]] for l in labels])
labels_categorical = to_categorical(labels_mapped, num_classes=len(char_to_index))

# 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.1, random_state=42)

# 데이터셋 준비
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

# 데이터 증강 설정
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest',
    validation_split=0.1
)

# CNN 모델 정의 - 개선된 버전
model = Sequential([
    # 입력 레이어
    Input(shape=(28, 28, 1)),
    
    # 첫 번째 컨볼루션 블록
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # 두 번째 컨볼루션 블록
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # 세 번째 컨볼루션 블록
    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # 완전연결층
    Flatten(),
    Dense(1024, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(char_to_index), activation='softmax')
])

# 모델 컴파일 - 개선된 옵티마이저 설정
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# 콜백 설정 개선
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# 학습
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    validation_data=(X_val, y_val),
    epochs=100,
    steps_per_epoch=len(X_train) // 128,
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# 최종 모델 저장
model.save("emnist_byclass_cnn_model.h5")
print("모델 저장 완료: emnist_byclass_cnn_model.h5") 