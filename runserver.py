import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import cv2
import logging
from datetime import datetime
from deep_learning.preprocessing.split_handwritten_sentence import split_handwritten_sentence

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 루트 디렉토리 설정
project_root = os.path.dirname(os.path.abspath(__file__))

# test 폴더 생성
test_dir = os.path.join(project_root, 'deep_learning/test')
os.makedirs(test_dir, exist_ok=True)

# EMNIST 클래스 매핑 (0-9, A-Z, a-z)
CLASS_MAPPING = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

class Predictor:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Predictor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            logger.info("모델 초기화 시작...")
            self._model = self._create_model()
            self._load_weights()
            self._initialized = True
            logger.info("모델 초기화 완료")
    
    def _create_model(self):
        # 모델 구조 정의
        inputs = Input(shape=(28, 28, 1))
        
        # 첫 번째 컨볼루션 블록
        x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # 두 번째 컨볼루션 블록
        x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # 세 번째 컨볼루션 블록
        x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        
        # 완전 연결 레이어
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(36, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _load_weights(self):
        # 가중치 파일 경로
        weights_path = os.path.join(project_root, 'deep_learning/models/emnist_byclass_cnn_model.h5')
        
        try:
            # 가중치만 로드
            self._model.load_weights(weights_path)
            logger.info("모델 가중치 로드 완료")
            
            # 모델 구조 출력
            self._model.summary(print_fn=logger.info)
        except Exception as e:
            logger.error(f"가중치 로드 중 오류 발생: {str(e)}")
            raise
    
    def predict(self, image_data):
        try:
            # 이미지 전처리
            logger.info(f"입력 이미지 shape: {image_data.shape}")
            logger.info(f"입력 이미지 값 범위: [{np.min(image_data)}, {np.max(image_data)}]")
            
            if image_data.shape != (28, 28, 1):
                image_data = cv2.resize(image_data, (28, 28))
                if len(image_data.shape) == 2:
                    image_data = np.expand_dims(image_data, axis=-1)
            
            logger.info(f"리사이즈 후 이미지 shape: {image_data.shape}")
            logger.info(f"리사이즈 후 이미지 값 범위: [{np.min(image_data)}, {np.max(image_data)}]")
            
            # 정규화
            image_data = image_data.astype('float32') / 255.0
            
            # 배치 차원 추가
            image_data = np.expand_dims(image_data, axis=0)
            
            logger.info(f"최종 입력 shape: {image_data.shape}")
            logger.info(f"최종 입력 값 범위: [{np.min(image_data)}, {np.max(image_data)}]")
            
            # 예측
            predictions = self._model.predict(image_data, verbose=0)
            predicted_class = np.argmax(predictions[0])
            
            # 예측 결과 로깅
            logger.info(f"예측 확률 분포: {predictions[0]}")
            logger.info(f"예측된 클래스: {predicted_class} ({CLASS_MAPPING[predicted_class]})")
            
            return {
                'character': CLASS_MAPPING[predicted_class],
                'confidence': float(predictions[0][predicted_class])
            }
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise

app = FastAPI()

# CORS 미들웨어 추가
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 중에는 모든 출처 허용, 프로덕션에서는 특정 도메인만 허용하도록 수정
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # 필요한 HTTP 메서드만 명시
    allow_headers=["*"],
    expose_headers=["*"]
)

predictor = Predictor()

@app.get("/")
async def root():
    return {"message": "OCR Prediction Server is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            logger.error("이미지를 읽을 수 없습니다.")
            return JSONResponse(
                status_code=400,
                content={"error": "이미지를 읽을 수 없습니다."}
            )
        
        logger.info(f"원본 이미지 shape: {image.shape}")
        logger.info(f"원본 이미지 값 범위: [{np.min(image)}, {np.max(image)}]")
        
        # 이미지 전처리
        # 이미지 크기 조정 및 패딩 추가
        h, w = image.shape
        size = max(h, w)
        square_image = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        # 28x28로 리사이즈
        resized_image = cv2.resize(square_image, (28, 28))
        
        # 흑백 반전
        inverted_original = 255 - image
        inverted_resized = 255 - resized_image
        
        # 이미지 저장 (원본, 반전)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(file.filename)[0]
        save_path = os.path.join(test_dir, f"{original_filename}_{timestamp}_original.png")
        cv2.imwrite(save_path, inverted_original)
        logger.info(f"원본(반전) 이미지 저장됨: {save_path}")
        
        # 전처리된 이미지 저장 (반전)
        preprocessed_path = os.path.join(test_dir, f"{original_filename}_{timestamp}_preprocessed.png")
        cv2.imwrite(preprocessed_path, inverted_resized)
        logger.info(f"전처리(반전) 이미지 저장됨: {preprocessed_path}")
        
        # 예측을 위한 이미지 준비 (반전)
        image_for_prediction = inverted_resized.reshape(28, 28, 1)
        
        # 예측
        result = predictor.predict(image_for_prediction)
        logger.info(f"예측 결과: {result}")
        
        # 예측 결과를 파일명에 추가
        predicted_char = result['character']
        confidence = result['confidence']
        final_path = os.path.join(test_dir, f"{original_filename}_{timestamp}_pred_{predicted_char}_{confidence:.2f}.png")
        os.rename(preprocessed_path, final_path)
        logger.info(f"이미지 이름 변경됨: {final_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"서버 오류: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/predict-sentence")
async def predict_sentence(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error("이미지를 읽을 수 없습니다.")
            return JSONResponse(status_code=400, content={"error": "이미지를 읽을 수 없습니다."})

        # 문장 이미지에서 문자 분리
        char_images = split_handwritten_sentence(image)
        if not char_images:
            logger.error("문자를 분리할 수 없습니다.")
            return JSONResponse(status_code=400, content={"error": "문자를 분리할 수 없습니다."})

        # 각 문자 인식
        result_chars = []
        for char_img in char_images:
            if len(char_img.shape) == 2:
                char_img = np.expand_dims(char_img, axis=-1)
            pred = predictor.predict(char_img)
            result_chars.append(pred['character'])

        sentence = ''.join(result_chars)
        logger.info(f"예측된 문장: {sentence}")
        return {"sentence": sentence, "characters": result_chars}

    except Exception as e:
        logger.error(f"서버 오류: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0",  # 모든 네트워크 인터페이스에서 접근 가능
        port=8000,
        log_level="info"
    ) 