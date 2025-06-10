import tensorflow as tf
import os
import numpy as np

def load_and_summarize_model(model_path):
    print(f"Loading model from: {model_path}")
    
    # 모델 로드
    model = tf.keras.models.load_model(model_path)
    
    # 모델 구조 출력
    print("\n=== Model Summary ===")
    model.summary()
    
    # 레이어 상세 정보 출력
    print("\n=== Layer Details ===")
    for layer in model.layers:
        print(f"\nLayer: {layer.name}")
        print(f"Type: {layer.__class__.__name__}")
        
        # 컨볼루션 레이어 정보
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"Filters: {layer.filters}")
            print(f"Kernel Size: {layer.kernel_size}")
            print(f"Activation: {layer.activation.__name__}")
        
        # Dense 레이어 정보
        elif isinstance(layer, tf.keras.layers.Dense):
            print(f"Units: {layer.units}")
            print(f"Activation: {layer.activation.__name__}")
        
        # BatchNormalization 레이어 정보
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            print("Type: BatchNormalization")
        
        # Dropout 레이어 정보
        elif isinstance(layer, tf.keras.layers.Dropout):
            print(f"Rate: {layer.rate}")
    

if __name__ == "__main__":
    model_path = "/home/minc/OCR_project/deep_learning/models/balance/final_model.h5"
    load_and_summarize_model(model_path) 