import os
import sys
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# ====== 경로 설정 (여기만 수정하면 됨) ======
project_root = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(project_root, 'deep_learning/models/balance/final_model.h5')
TEST_DATA_PATH = os.path.join(project_root, 'deep_learning/data/emnist/emnist-balanced-test.csv')
RESULTS_DIR = os.path.join(project_root, 'deep_learning/results')
# =========================================

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EMNIST balanced: 0-9, 10-35(A-Z), 36-61(a-z)
# 대소문자 구분 없이 0-9, 10-35(A-Z)로만 매핑
CLASS_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

# 소문자 레이블(EMNIST balanced 기준 36~61)을 대문자 레이블(10~35)로 매핑
def map_label(label):
    if 36 <= label <= 61:
        return label - 26  # 'a'~'z' -> 'A'~'Z'
    return label

def create_model():
    # 모델 구조 정의
    inputs = Input(shape=(28, 28, 1))
    
    # 첫 번째 컨볼루션 블록
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 두 번째 컨볼루션 블록
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 세 번째 컨볼루션 블록
    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    # 완전 연결 레이어
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(36, activation='softmax')(x)  # 36개 클래스 (0-9, A-Z)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_data(test_file):
    logger.info("테스트 데이터 로드 중...")
    test_data = pd.read_csv(test_file, header=None)
    
    # 첫 번째 열이 레이블
    y_test_raw = test_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    
    # 레이블 매핑 (소문자→대문자)
    y_test = np.array([map_label(l) for l in y_test_raw])
    
    # 데이터 전처리
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # 원-핫 인코딩
    y_test_onehot = np.zeros((len(y_test), 36))  # 36개 클래스
    for i, label in enumerate(y_test):
        y_test_onehot[i, label] = 1
    
    logger.info(f"테스트 데이터 shape: {X_test.shape}")
    logger.info(f"테스트 레이블 shape: {y_test_onehot.shape}")
    
    return X_test, y_test_onehot, y_test

def evaluate_model():
    # 모델 생성
    model = create_model()
    
    # 가중치 로드
    logger.info(f"모델 가중치 로드 중: {MODEL_WEIGHTS_PATH}")
    model.load_weights(MODEL_WEIGHTS_PATH)
    
    # 테스트 데이터 로드
    X_test, y_test_onehot, y_test_raw = load_data(TEST_DATA_PATH)
    
    # 모델 평가
    logger.info("모델 평가 중...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_onehot, verbose=1)
    logger.info(f"테스트 정확도: {test_accuracy:.4f}")
    logger.info(f"테스트 손실: {test_loss:.4f}")
    
    # 예측
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # 분류 보고서
    logger.info("\n분류 보고서:")
    report = classification_report(y_test_raw, y_pred_classes, 
                                 target_names=[CLASS_MAPPING[i] for i in range(36)],
                                 digits=4)
    logger.info(report)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(20, 20))
    cm = confusion_matrix(y_test_raw, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[CLASS_MAPPING[i] for i in range(36)],
                yticklabels=[CLASS_MAPPING[i] for i in range(36)])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 혼동 행렬 저장
    plt.savefig(os.path.join(RESULTS_DIR, 'balanced_model_confusion_matrix.png'))
    
    # 분류 보고서 저장
    with open(os.path.join(RESULTS_DIR, 'balanced_model_classification_report.txt'), 'w') as f:
        f.write(f"테스트 정확도: {test_accuracy:.4f}\n")
        f.write(f"테스트 손실: {test_loss:.4f}\n\n")
        f.write(report)
    
    logger.info(f"결과가 {RESULTS_DIR} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    evaluate_model() 