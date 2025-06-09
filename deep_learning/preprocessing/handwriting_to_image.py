import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

# 기본 이미지 저장 경로 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TEST_DIR = os.path.join(PROJECT_ROOT, 'deep_learning/test')
OUTPUT_DIR = os.path.join(TEST_DIR, 'output')

def crop_to_content(image):
    """
    이미지에서 실제 내용이 있는 부분만 추출합니다.
    """
    # 이미지 이진화 (반전 없이)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # 모든 윤곽선을 포함하는 최소 사각형 찾기
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    
    # 여백 추가 (40%로 증가)
    padding_x = int(w * 0.4)
    padding_y = int(h * 0.4)
    
    # 이미지 경계를 벗어나지 않도록 패딩 조정
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(image.shape[1] - x, w + 2 * padding_x)
    h = min(image.shape[0] - y, h + 2 * padding_y)
    
    # 이미지 자르기
    cropped = image[y:y+h, x:x+w]
    
    # 크롭된 이미지 시각화
    plt.figure(figsize=(10, 3))
    plt.imshow(cropped, cmap='gray')
    plt.title('Cropped Image')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, '0_cropped.png'))
    plt.close()
    
    return cropped

def preprocess_image(image_path):
    """
    이미지를 전처리합니다.
    """
    # 파일 존재 여부 확인
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    
    # 이미지 읽기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 이미지가 없는 경우 에러 처리
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 이미지 크기 조정 (너무 큰 경우)
    max_size = 1000
    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # 실제 내용이 있는 부분만 자르기
    img = crop_to_content(img)
    
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
    
    # 모폴로지 연산으로 노이즈 제거 (주석 처리)
    # kernel = np.ones((2,2), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def filter_nested_components(characters):
    filtered = []
    for i, (x1, y1, w1, h1) in enumerate(characters):
        rect1 = (x1, y1, x1 + w1, y1 + h1)
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(characters):
            if i == j:
                continue
            rect2 = (x2, y2, x2 + w2, y2 + h2)
            # rect1이 rect2에 거의 완전히 포함되면
            if (rect1[0] >= rect2[0] and rect1[1] >= rect2[1] and
                rect1[2] <= rect2[2] and rect1[3] <= rect2[3]):
                is_nested = True
                break
        if not is_nested:
            filtered.append((x1, y1, w1, h1))
    return filtered

def find_characters(binary_image):
    """
    연결 요소 분석을 통해 각 문자를 찾고 반환합니다. (병합 없이)
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    characters = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 2 and w > 1 and h > 1:
            characters.append((x, y, w, h))
    # 중첩된 컴포넌트 제거
    characters = filter_nested_components(characters)
    # x좌표 기준으로 왼쪽부터 정렬
    characters = sorted(characters, key=lambda c: c[0])
    # 디버깅용 시각화
    debug_img = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in characters:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'debug_components.png'), debug_img)
    return characters

def process_sentence_image(image_path):
    """
    문장 이미지를 처리하여 각 문자를 분리하고 저장합니다.
    """
    try:
        # 출력 디렉토리 생성
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # 원본 이미지 로드
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        plt.figure(figsize=(10, 3))
        plt.imshow(original_img, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, '1_original.png'))
        plt.close()
        
        # 이미지 전처리
        binary_image = preprocess_image(image_path)
        plt.figure(figsize=(10, 3))
        plt.imshow(binary_image, cmap='gray')
        plt.title('Preprocessed Image')
        plt.axis('off')
        plt.savefig(os.path.join(OUTPUT_DIR, '2_preprocessed.png'))
        plt.close()
        
        # 문자 찾기
        characters = find_characters(binary_image)
        
        if not characters:
            print("문자를 찾을 수 없습니다.")
            return
        
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
            
            # 문자 시각화
            plt.figure(figsize=(4, 4))
            plt.imshow(final_image, cmap='gray')
            plt.title(f'Character {i}')
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, f'3_char_{i}.png'))
            plt.close()
        
        print(f"처리된 이미지들이 {OUTPUT_DIR} 폴더에 저장되었습니다.")
            
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        raise

def extract_character(image, x, y, w, h, padding=2):
    """
    문자 영역을 추출하고, 내부에 포함된 작은 컴포넌트(끝에 튀어나온 다른 문자)를 제거합니다.
    중심(중앙)과 가까운 여러 컨투어를 남깁니다.
    """
    # 문자 영역 추출 (패딩 포함)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)
    char_img = image[y1:y2, x1:x2].copy()

    # 중심 좌표 계산
    center_x = (char_img.shape[1] - 1) / 2
    center_y = (char_img.shape[0] - 1) / 2
    max_dist = ((center_x)**2 + (center_y)**2) ** 0.5
    keep_ratio = 0.4  # 중심에서 40% 이내만 남김

    # 내부 컨투어 분석
    contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(char_img)
    for cnt in contours:
        # 컨투어의 중심 계산
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        if dist < max_dist * keep_ratio:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
    char_img = cv2.bitwise_and(char_img, mask)
    return char_img

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
    try:
        # test 폴더 존재 확인
        if not os.path.exists(TEST_DIR):
            print(f"test 폴더가 존재하지 않습니다: {TEST_DIR}")
            return
            
        # test 폴더에서 sentence로 시작하는 이미지 파일 찾기
        image_files = [f for f in os.listdir(TEST_DIR) if f.lower().startswith('sentence') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("test 폴더에 sentence로 시작하는 이미지 파일이 없습니다.")
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
        
        selected_image = os.path.join(TEST_DIR, image_files[choice])
        print(f"\n선택된 이미지: {selected_image}")
        
        # 이미지 처리
        process_sentence_image(selected_image)
        print(f"\n모든 문자가 {OUTPUT_DIR} 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 