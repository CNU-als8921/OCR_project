import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# tf.data.Dataset 구성
def preprocess(image, label):
    # image: 28x28 → 224x224x3 변환
    image = tf.expand_dims(image, axis=-1)  # (28,28,1)
    image = tf.image.resize(image, [224,224])  # (224,224,1)
    image = tf.image.grayscale_to_rgb(image)   # (224,224,3)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.map(preprocess).batch(64).prefetch(tf.data.AUTOTUNE)

# VGG16 모델 불러오기
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom classifier
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(len(char_to_index), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 콜백
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 학습
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    callbacks=[early_stop, reduce_lr]
)

# 저장
model.save("emnist_byclass_vgg16_transfer.h5")
print("모델 저장 완료: emnist_byclass_vgg16_transfer.h5")
