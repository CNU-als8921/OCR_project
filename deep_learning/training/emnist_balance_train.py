import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def create_case_insensitive_mapping():
    # 숫자 0-9는 그대로 유지
    mapping = {i: i for i in range(10)}
    
    # 대문자 A-Z는 10-35로 매핑
    for i in range(26):
        # 대문자 A-Z (ASCII 65-90)
        mapping[i + 10] = i + 10
        # 소문자 a-z (ASCII 97-122)를 대문자로 매핑
        mapping[i + 36] = i + 10
    
    return mapping

def load_emnist_data():
    print("Loading EMNIST data...")
    data_path = '/home/minc/OCR_project/deep_learning/data/emnist'
    
    # 학습 데이터 로드
    train_data = pd.read_csv(os.path.join(data_path, 'emnist-balanced-train.csv'), header=None)
    test_data = pd.read_csv(os.path.join(data_path, 'emnist-balanced-test.csv'), header=None)
    
    # 데이터 검증
    print("\n=== Data Validation ===")
    print(f"Total training samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    print(f"Number of features: {train_data.shape[1] - 1}")  # 첫 번째 열은 레이블
    
    # 레이블 분포 확인
    train_labels = train_data.iloc[:, 0].values
    test_labels = test_data.iloc[:, 0].values
    
    print("\nOriginal label distribution in training set:")
    for label in sorted(np.unique(train_labels)):
        count = np.sum(train_labels == label)
        print(f"Label {label}: {count} samples")
    
    # 데이터 전처리
    x_train = train_data.iloc[:, 1:].values.astype('float32') / 255.0
    y_train = train_data.iloc[:, 0].values
    x_test = test_data.iloc[:, 1:].values.astype('float32') / 255.0
    y_test = test_data.iloc[:, 0].values
    
    # 이미지 reshape (28x28)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # 대소문자 매핑 적용
    mapping = create_case_insensitive_mapping()
    y_train = np.array([mapping[y] for y in y_train])
    y_test = np.array([mapping[y] for y in y_test])
    
    # 매핑 후 레이블 분포 확인
    print("\nMapped label distribution in training set:")
    unique_mapped_labels = np.unique(y_train)
    for label in sorted(unique_mapped_labels):
        count = np.sum(y_train == label)
        if label < 10:
            label_name = str(label)
        else:
            label_name = chr(label - 10 + 65)  # 10-35를 A-Z로 변환
        print(f"Mapped label {label_name}: {count} samples")
    
    # 원-핫 인코딩 (36개 클래스: 0-9, A-Z)
    y_train = tf.keras.utils.to_categorical(y_train, 36)
    y_test = tf.keras.utils.to_categorical(y_test, 36)
    
    print(f"\nTraining data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_model(num_classes):
    model = tf.keras.Sequential([
        # 첫 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # 두 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # 세 번째 컨볼루션 블록
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),
        
        # 완전 연결 레이어
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    # 학습 곡선 그리기
    plt.figure(figsize=(12, 4))
    
    # 정확도
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 손실
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def train_model():
    # 데이터 로드
    (x_train, y_train), (x_test, y_test) = load_emnist_data()
    
    # 모델 생성
    model = create_model(num_classes=36)  # 36개 클래스: 0-9, A-Z
    
    # 모델 구조 출력
    model.summary()
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # 모델 학습
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=50,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 학습 곡선 저장
    plot_training_history(history)
    
    # 최종 모델 저장
    model.save('final_model.h5')
    
    # 테스트 세트에서 평가
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # 예측 및 분류 보고서
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 분류 보고서
    class_names = [str(i) for i in range(10)] + [chr(i + 65) for i in range(26)]  # 0-9, A-Z
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # 혼동 행렬
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    train_model() 