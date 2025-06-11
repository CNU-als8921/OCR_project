import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import logging
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import matplotlib.font_manager as fm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 클래스 매핑 직접 선언
CLASS_MAPPING = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I',
    19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R',
    28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z'
}

def create_model():
    """모델 구조 생성"""
    inputs = Input(shape=(28,28,1))
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(36, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_custom_dataset(data_dir):
    """커스텀 데이터셋 로드"""
    images = []
    labels = []
    label_names = []
    
    # 레이블 조회를 위한 CLASS_MAPPING 역방향 매핑
    reverse_mapping = {v: k for k, v in CLASS_MAPPING.items()}
    
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.isdir(label_dir):
            continue
            
        # 대소문자 구분 없이 매칭하기 위해 레이블을 대문자로 변환
        label_name_upper = label_name.upper()
        if label_name_upper not in reverse_mapping:
            logger.warning(f"알 수 없는 레이블 건너뛰기: {label_name}")
            continue
            
        label = reverse_mapping[label_name_upper]
        for img_name in os.listdir(label_dir):
            if not img_name.endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logger.warning(f"이미지 로드 실패: {img_path}")
                continue
                
            # 28x28 크기로 리사이즈
            img = cv2.resize(img, (28, 28))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=-1)
            
            images.append(img)
            labels.append(label)
            label_names.append(label_name_upper)
            
    return np.array(images), np.array(labels), label_names

def evaluate_model():
    """전이학습 모델 평가"""
    # 프로젝트 루트 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 모델 로드
    model_path = os.path.join(project_root, 'deep_learning/models/transfer_learning_model.h5')
    try:
        model = load_model(model_path)
        logger.info("전이학습 모델 로드 완료")
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        return
    
    # 커스텀 데이터셋 로드
    data_dir = os.path.join(project_root, 'deep_learning/data/custom_handwriting')
    X_test, y_test, label_names = load_custom_dataset(data_dir)
    
    if len(X_test) == 0:
        logger.error("커스텀 데이터셋에서 이미지를 찾을 수 없습니다!")
        return
    
    logger.info(f"평가를 위해 {len(X_test)}개의 이미지를 로드했습니다")
    
    # 배치 단위로 예측 수행
    predictions = []
    confidences = []
    
    # 32개씩 배치 처리
    batch_size = 32
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i + batch_size]
        batch_predictions = model.predict(batch, verbose=0)
        pred_classes = np.argmax(batch_predictions, axis=1)
        pred_confidences = np.max(batch_predictions, axis=1)
        
        predictions.extend(pred_classes)
        confidences.extend(pred_confidences)
        
        if i % 100 == 0:
            logger.info(f"처리된 이미지: {i}/{len(X_test)}")
    
    predictions = np.array(predictions)
    confidences = np.array(confidences)
    
    # 정확도 계산
    accuracy = np.mean(predictions == y_test)
    
    # 결과 저장 디렉토리 생성
    output_dir = os.path.join(project_root, 'evaluation_results_transfer')
    os.makedirs(output_dir, exist_ok=True)
    
    # 혼동 행렬 생성
    cm = confusion_matrix(y_test, predictions)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(CLASS_MAPPING.values()),
                yticklabels=list(CLASS_MAPPING.values()))
    plt.title('Confusion Matrix - Transfer Learning Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 예측 신뢰도 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(confidences, bins=20)
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'))
    plt.close()
    
    # 분류 보고서 생성
    report = classification_report(y_test, predictions, 
                                 target_names=list(CLASS_MAPPING.values()),
                                 output_dict=True)
    
    # 클래스별 정확도 계산 및 시각화
    class_accuracies = []
    for i in range(len(CLASS_MAPPING)):
        mask = y_test == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == y_test[mask])
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(CLASS_MAPPING)), class_accuracies)
    plt.xticks(range(len(CLASS_MAPPING)), list(CLASS_MAPPING.values()), rotation=45)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'))
    plt.close()
    
    # 평가 지표를 텍스트 파일로 저장
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, predictions, 
                                    target_names=list(CLASS_MAPPING.values())))
    
    logger.info(f"평가 완료. 결과가 {output_dir}에 저장되었습니다")
    logger.info(f"전체 정확도: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model() 