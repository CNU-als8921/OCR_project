import os
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from keras.optimizers import Adam
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 클래스 매핑 직접 선언
CLASS_MAPPING = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'I',
    19:'J', 20:'K', 21:'L', 22:'M', 23:'N', 24:'O', 25:'P', 26:'Q', 27:'R',
    28:'S', 29:'T', 30:'U', 31:'V', 32:'W', 33:'X', 34:'Y', 35:'Z'
}

def create_base_model():
    """기본 모델 구조 생성"""
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

def load_and_preprocess_data(data_dir):
    """원본 데이터 로드 및 전처리"""
    images = []
    labels = []
    
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
    
    return np.array(images), np.array(labels)

def augment_data(images, labels, target_count=100000):
    """데이터 증강 수행"""
    logger.info("데이터 증강 시작...")
    
    # 데이터 증강을 위한 ImageDataGenerator 설정
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    augmented_images = []
    augmented_labels = []
    
    # 각 클래스별로 필요한 증강 횟수 계산
    unique_labels = np.unique(labels)
    samples_per_class = target_count // len(unique_labels)
    
    for label in unique_labels:
        class_images = images[labels == label]
        current_count = len(class_images)
        
        if current_count >= samples_per_class:
            # 이미 충분한 데이터가 있는 경우 랜덤 샘플링
            indices = np.random.choice(current_count, samples_per_class, replace=False)
            augmented_images.extend(class_images[indices])
            augmented_labels.extend([label] * samples_per_class)
        else:
            # 부족한 경우 증강 수행
            augmented_images.extend(class_images)
            augmented_labels.extend([label] * current_count)
            
            remaining = samples_per_class - current_count
            for img in class_images:
                if len(augmented_images) >= target_count:
                    break
                    
                # 각 이미지당 필요한 증강 횟수
                aug_count = min(remaining // current_count + 1, 10)
                for _ in range(aug_count):
                    aug_img = next(datagen.flow(np.expand_dims(img, 0)))[0]
                    augmented_images.append(aug_img)
                    augmented_labels.append(label)
                    remaining -= 1
    
    logger.info(f"데이터 증강 완료. 총 {len(augmented_images)}개의 이미지 생성됨")
    return np.array(augmented_images), np.array(augmented_labels)

def create_transfer_model(base_model):
    """전이학습을 위한 모델 생성"""
    # 기본 모델의 마지막 레이어를 제외한 모든 레이어 고정
    for layer in base_model.layers[:-1]:
        layer.trainable = False
    
    # 새로운 출력 레이어 추가
    x = base_model.layers[-2].output
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(36, activation='softmax')(x)
    
    # 새로운 모델 생성 (낮은 학습률로 설정)
    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # 학습률 낮춤
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # 프로젝트 루트 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 데이터 로드
    data_dir = os.path.join(project_root, 'data/custom_handwriting')
    images, labels = load_and_preprocess_data(data_dir)
    
    # 데이터 증강
    augmented_images, augmented_labels = augment_data(images, labels)
    
    # 레이블을 원-핫 인코딩으로 변환
    augmented_labels = to_categorical(augmented_labels, num_classes=36)
    
    # 데이터 분할 (80% 학습, 20% 검증)
    indices = np.random.permutation(len(augmented_images))
    split = int(len(indices) * 0.8)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    X_train = augmented_images[train_indices]
    y_train = augmented_labels[train_indices]
    X_val = augmented_images[val_indices]
    y_val = augmented_labels[val_indices]
    
    # 기본 모델 생성 및 가중치 로드
    base_model = create_base_model()
    weights_path = os.path.join(project_root, 'models/emnist_byclass_cnn_model.h5')
    try:
        base_model.load_weights(weights_path)
        logger.info("기존 모델 가중치 로드 완료")
    except Exception as e:
        logger.error(f"가중치 로드 실패: {str(e)}")
        return
    
    # 전이학습 모델 생성
    model = create_transfer_model(base_model)
    
    # 콜백 설정 수정
    checkpoint = ModelCheckpoint(
        os.path.join(project_root, 'deep_learning/models/transfer_learning_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=7,  # patience 증가
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,  # learning rate 감소 비율 조정
        patience=4,  # patience 증가
        min_lr=1e-7,  # 최소 learning rate 조정
        verbose=1
    )
    
    # 모델 학습
    logger.info("모델 학습 시작...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,  # batch size 증가
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    logger.info("모델 학습 완료")
    
    # 최종 모델 저장
    model.save(os.path.join(project_root, 'deep_learning/models/transfer_learning_model_final.h5'))
    logger.info("최종 모델 저장 완료")

if __name__ == "__main__":
    train_model() 