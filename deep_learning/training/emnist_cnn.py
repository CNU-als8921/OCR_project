import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# 데이터 경로
dataset_path = "/home/minc/OCR_project/dataset/emnist/versions/3/"

# 데이터 로드
print("Loading data...")
data = pd.read_csv(dataset_path + 'emnist-byclass-train.csv')
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

mapping = load_mapping(dataset_path + 'emnist-byclass-mapping.txt')
unique_chars = sorted(set(mapping.values()))
char_to_index = {char: i for i, char in enumerate(unique_chars)}
labels_mapped = np.array([char_to_index[mapping[l]] for l in labels])
labels_categorical = to_categorical(labels_mapped, num_classes=len(char_to_index))

# 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.1, random_state=42)

# 데이터셋 준비
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)

# CNN 모델 정의
model = Sequential([
    # 첫 번째 컨볼루션 블록
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 두 번째 컨볼루션 블록
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 세 번째 컨볼루션 블록
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # 완전연결층
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(char_to_index), activation='softmax')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 구조 출력
model.summary()

# 콜백
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# 학습
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=[early_stop, reduce_lr]
)

# 모델 저장
model.save("emnist_byclass_cnn_model.h5")
print("모델 저장 완료: emnist_byclass_cnn_model.h5") 