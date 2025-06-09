import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# 기본 이미지 저장 경로 설정
DEFAULT_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_images')

def preprocess_image(image_path):
    """
    이미지를 전처리합니다.
    """
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지가 없는 경우 에러 처리
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    
    # 이미지 크기 조정 (너무 큰 경우)
    max_size = 1000
    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # 노이즈 제거 (더 작은 커널 사용)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # 적응형 이진화 적용
    binary = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,  # 블록 크기
        2    # C 상수
    )
    
    # 모폴로지 연산으로 노이즈 제거 (더 작은 커널 사용)
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def find_characters(binary_image):
    """
    연결 요소 분석을 통해 각 문자를 찾고, x좌표 범위에 80% 이상 포함되는 컴포넌트들을 하나로 합칩니다.
    노이즈와 작은 튀어나온 부분은 제거합니다.
    """
    # 연결 요소 찾기
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # 배경 제외하고 문자 영역만 추출
    components = []
    for i in range(1, num_labels):  # 0은 배경
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 너무 작은 영역 제외 (노이즈 제거)
        if area > 20:  # 최소 영역 크기
            components.append({
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'area': area
            })
    
    # x 좌표 기준으로 정렬
    components.sort(key=lambda c: c['x'])
    
    # x좌표 범위에 80% 이상 포함되는 컴포넌트들을 그룹화
    merged_components = []
    current_group = []
    
    for i, comp in enumerate(components):
        if not current_group:
            current_group.append(comp)
            continue
            
        # 현재 컴포넌트와 이전 그룹의 마지막 컴포넌트 비교
        last_comp = current_group[-1]
        
        # x좌표 범위 계산
        current_range = (comp['x'], comp['x'] + comp['w'])
        last_range = (last_comp['x'], last_comp['x'] + last_comp['w'])
        
        # 겹치는 영역 계산
        overlap_start = max(current_range[0], last_range[0])
        overlap_end = min(current_range[1], last_range[1])
        overlap_width = max(0, overlap_end - overlap_start)
        
        # 현재 컴포넌트의 넓이 대비 겹치는 영역의 비율 계산
        overlap_ratio = overlap_width / comp['w']
        
        # 80% 이상 겹치면 같은 그룹에 추가
        if overlap_ratio >= 0.8:
            current_group.append(comp)
        else:
            # 현재 그룹을 병합
            if current_group:
                merged = merge_components(current_group)
                merged_components.append(merged)
            current_group = [comp]
    
    # 마지막 그룹 처리
    if current_group:
        merged = merge_components(current_group)
        merged_components.append(merged)
    
    # 노이즈 제거: 작은 튀어나온 부분 제거
    filtered_components = []
    for comp in merged_components:
        # 컴포넌트의 크기가 너무 작으면 제외
        if comp['w'] < 3 or comp['h'] < 3:  # 최소 크기 기준을 더 작게 조정
            continue
            
        # 컴포넌트의 비율이 비정상적이면 제외 (너무 길쭉하거나 넓적한 경우)
        aspect_ratio = comp['w'] / comp['h']
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # 비율 기준을 더 관대하게 조정
            continue
            
        filtered_components.append(comp)
    
    # 최종 문자 영역 반환
    characters = [(comp['x'], comp['y'], comp['w'], comp['h']) for comp in filtered_components]
    return characters

def merge_components(components):
    """
    여러 컴포넌트를 하나로 병합합니다.
    """
    if not components:
        return None
        
    # 모든 컴포넌트를 포함하는 바운딩 박스 계산
    x = min(comp['x'] for comp in components)
    y = min(comp['y'] for comp in components)
    w = max(comp['x'] + comp['w'] for comp in components) - x
    h = max(comp['y'] + comp['h'] for comp in components) - y
    
    # 병합된 컴포넌트의 크기가 비정상적으로 크면 제외
    if w > 100 or h > 100:  # 최대 크기 제한
        return None
    
    return {
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'area': sum(comp['area'] for comp in components)
    }

def process_image_to_characters(image_path):
    """
    이미지를 한 글자씩 분리하여 저장하고 시각화합니다.
    """
    # 출력 디렉토리 생성
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 이미지 전처리
    binary_image = preprocess_image(image_path)
    
    # 원본 이미지 표시
    plt.figure(figsize=(10, 3))
    plt.imshow(binary_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    # 문자 찾기
    characters = find_characters(binary_image)
    
    # 각 문자 처리
    for i, (x, y, w, h) in enumerate(characters):
        # 문자 이미지 추출
        char_image = extract_character(binary_image, x, y, w, h)
        
        # 원본 비율 유지하면서 28x28 크기로 조정
        h, w = char_image.shape
        aspect_ratio = w / h
        
        if aspect_ratio > 1:  # 가로가 더 긴 경우
            new_w = 28
            new_h = int(28 / aspect_ratio)
        else:  # 세로가 더 긴 경우
            new_h = 28
            new_w = int(28 * aspect_ratio)
        
        # 크기 조정
        resized = cv2.resize(char_image, (new_w, new_h))
        
        # 28x28 크기의 검은 배경 생성
        final_image = np.zeros((28, 28), dtype=np.uint8)
        
        # 중앙에 문자 배치
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 이미지 저장
        output_path = os.path.join(OUTPUT_DIR, f'char_{i}.png')
        save_as_mnist_format(final_image, output_path)
        
        # 문자 표시
        plt.figure(figsize=(2, 2))
        plt.imshow(final_image, cmap='gray')
        plt.title(f'Character {i}')
        plt.axis('off')
        plt.show()

def extract_character(image, x, y, w, h, padding=2):
    """
    주어진 좌표에서 문자를 추출하고 배경을 지웁니다.
    """
    # 패딩 추가
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)
    
    # 문자 영역 추출
    char_image = image[y_start:y_end, x_start:x_end].copy()
    
    # 배경 지우기 (0으로 설정)
    char_image[char_image < 128] = 0
    
    return char_image

def save_as_mnist_format(image, output_path):
    """
    전처리된 이미지를 MNIST 형식으로 저장합니다.
    """
    # 이미지를 0-255 범위로 변환
    img_255 = image.astype(np.uint8)
    
    # PIL Image로 변환하여 저장
    img_pil = Image.fromarray(img_255)
    img_pil.save(output_path)
    
    print(f"이미지가 {output_path}에 저장되었습니다.")

def main():
    # images 폴더의 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(DEFAULT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("images 폴더에 이미지 파일이 없습니다.")
        return
    
    print("처리할 수 있는 이미지 파일들:")
    for i, file in enumerate(image_files):
        print(f"{i+1}. {file}")
    
    # 사용자로부터 처리할 이미지 선택
    while True:
        try:
            choice = int(input("\n처리할 이미지 번호를 선택하세요: ")) - 1
            if 0 <= choice < len(image_files):
                break
            print("유효하지 않은 번호입니다. 다시 선택해주세요.")
        except ValueError:
            print("숫자를 입력해주세요.")
    
    selected_image = os.path.join(DEFAULT_IMAGE_DIR, image_files[choice])
    
    try:
        # 이미지 처리
        process_image_to_characters(selected_image)
        print(f"\n모든 문자가 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main() 