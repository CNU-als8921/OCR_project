import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import cv2
import logging
from datetime import datetime
from deep_learning.preprocessing.split_handwritten_sentence import split_handwritten_sentence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(project_root, 'deep_learning/test')
os.makedirs(test_dir, exist_ok=True)
custom_handwriting_dir = os.path.join(project_root, 'deep_learning/data/custom_handwriting')
os.makedirs(custom_handwriting_dir, exist_ok=True)

CLASS_MAPPING = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z'}

class Predictor:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Predictor, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            logger.info("포털 처리 시작...")
            self._model = self._create_model()
            self._load_weights()
            self._initialized = True
            logger.info("포털 처리 완료")

    def _create_model(self):
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

    def _load_weights(self):
        weights_path = os.path.join(project_root, 'deep_learning/models/emnist_byclass_cnn_model.h5')
        try:
            self._model.load_weights(weights_path)
            logger.info("게주치 로드 완료")
            self._model.summary(print_fn=logger.info)
        except Exception as e:
            logger.error(f"게주치 로드 오류: {str(e)}")
            raise

    def predict(self, image_data):
        try:
            if image_data.shape != (28,28,1):
                image_data = cv2.resize(image_data, (28,28))
                if len(image_data.shape) == 2:
                    image_data = np.expand_dims(image_data, axis=-1)

            image_data = image_data.astype('float32') / 255.0
            image_data = np.expand_dims(image_data, axis=0)
            predictions = self._model.predict(image_data, verbose=0)
            predicted_class = np.argmax(predictions[0])

            return {'character': CLASS_MAPPING[predicted_class], 'confidence': float(predictions[0][predicted_class])}
        except Exception as e:
            logger.error(f"예측 오류: {str(e)}")
            raise

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET","POST","OPTIONS"], allow_headers=["*"], expose_headers=["*"])

predictor = Predictor()

@app.get("/")
async def root():
    return {"message": "OCR Prediction Server is running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            return JSONResponse(status_code=400, content={"error":"이미지 읽기 오류"})

        h,w = image.shape
        size = max(h,w)
        square_image = np.zeros((size,size), dtype=np.uint8)
        y_offset = (size-h)//2
        x_offset = (size-w)//2
        square_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
        resized_image = cv2.resize(square_image, (28,28))
        inverted_resized = 255 - resized_image
        image_for_prediction = inverted_resized.reshape(28,28,1)

        result = predictor.predict(image_for_prediction)
        return result
    except Exception as e:
        logger.error(f"서버 오류: {str(e)}")
        return JSONResponse(status_code=500, content={"error":str(e)})

@app.post("/predict-sentence")
async def predict_sentence(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return JSONResponse(status_code=400, content={"error":"이미지 읽기 오류"})

        char_images = split_handwritten_sentence(image)
        if not char_images:
            return JSONResponse(status_code=400, content={"error":"문자 분리 오류"})

        result_chars = []
        for char_img in char_images:
            if len(char_img.shape) == 2:
                char_img = np.expand_dims(char_img, axis=-1)
            pred = predictor.predict(char_img)
            result_chars.append(pred['character'])

        sentence = ''.join(result_chars)
        return {"sentence":sentence, "characters":result_chars}
    except Exception as e:
        logger.error(f"서버 오류: {str(e)}")
        return JSONResponse(status_code=500, content={"error":str(e)})

@app.post("/api/save-handwriting")
async def save_handwriting_images(images: List[UploadFile] = File(...)):
    try:
        saved_files = []
        for image in images:
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error(f"이미지 읽기 오류: {image.filename}")
                continue
            save_path = os.path.join(custom_handwriting_dir, image.filename)
            cv2.imwrite(save_path, img)
            saved_files.append(image.filename)

        return {"message":"이미지 저장 완료", "saved_files":saved_files}
    except Exception as e:
        logger.error(f"저장 오류: {str(e)}")
        return JSONResponse(status_code=500, content={"error":str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
