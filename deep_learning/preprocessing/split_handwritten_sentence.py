import numpy as np
import cv2


def split_handwritten_sentence(image: np.ndarray) -> list:
    """
    입력: 흑백(또는 컬러) 이미지 (numpy array)
    출력: 분리된 문자 이미지들의 리스트 (각각 28x28 numpy array)
    """
    # 1. 흑백 변환
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 크롭(여백 제거)
    def crop_to_content(img):
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        padding_x = int(w * 0.4)
        padding_y = int(h * 0.4)
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(img.shape[1] - x, w + 2 * padding_x)
        h = min(img.shape[0] - y, h + 2 * padding_y)
        return img[y:y+h, x:x+w]
    image = crop_to_content(image)

    # 3. 블러 + 이진화
    image = cv2.GaussianBlur(image, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 4. x축 projection 기반 문자 분할
    def split_by_x_projection(binary):
        proj = np.sum(binary, axis=0)
        threshold = 10  # 픽셀 합이 이 값 이상이면 글자가 있다고 판단
        in_char = False
        chars = []
        start = 0
        for i, val in enumerate(proj):
            if not in_char and val > threshold:
                in_char = True
                start = i
            elif in_char and val <= threshold:
                in_char = False
                end = i
                chars.append((start, end))
        if in_char:
            chars.append((start, len(proj)))
        # y축 범위는 전체 nonzero 범위로
        ys, xs = np.where(binary > 0)
        if len(ys) == 0:
            return []
        y_min, y_max = ys.min(), ys.max()
        return [(x0, y_min, x1-x0, y_max-y_min+1) for x0, x1 in chars if x1-x0 > 1]

    chars = split_by_x_projection(binary)

    # 5. 문자별 추출 및 후처리
    def extract_character(img, x, y, w, h, padding=2):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img.shape[1], x + w + padding)
        y2 = min(img.shape[0], y + h + padding)
        char_img = img[y1:y2, x1:x2].copy()
        # 중심 기준 여러 컨투어 남기기 (옵션)
        center_x = (char_img.shape[1] - 1) / 2
        center_y = (char_img.shape[0] - 1) / 2
        max_dist = ((center_x)**2 + (center_y)**2) ** 0.5
        keep_ratio = 0.4
        contours, _ = cv2.findContours(char_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(char_img)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
            if dist < max_dist * keep_ratio:
                cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
        char_img = cv2.bitwise_and(char_img, mask)
        # 28x28 리사이즈 및 중앙 정렬
        h2, w2 = char_img.shape
        aspect_ratio = w2 / h2
        if aspect_ratio > 1:
            new_w = 28
            new_h = int(28 / aspect_ratio)
        else:
            new_h = 28
            new_w = int(28 * aspect_ratio)
        resized = cv2.resize(char_img, (new_w, new_h))
        final_image = np.zeros((28, 28), dtype=np.uint8)
        y_offset = (28 - new_h) // 2
        x_offset = (28 - new_w) // 2
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        return final_image

    char_images = [extract_character(binary, x, y, w, h) for (x, y, w, h) in chars]
    return char_images


if __name__ == "__main__":
    # 예시: 파일에서 읽어서 분리 결과 저장
    import sys
    import os
    import matplotlib.pyplot as plt
    if len(sys.argv) < 2:
        print("Usage: python split_handwritten_sentence.py [image_path]")
        exit(1)
    img_path = sys.argv[1]
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    chars = split_handwritten_sentence(img)
    out_dir = os.path.join(os.path.dirname(img_path), 'split_output')
    os.makedirs(out_dir, exist_ok=True)
    for i, cimg in enumerate(chars):
        cv2.imwrite(os.path.join(out_dir, f'char_{i}.png'), cimg)
    print(f"Saved {len(chars)} chars to {out_dir}")
    # (옵션) 전체 문자 시각화
    fig, axes = plt.subplots(1, len(chars), figsize=(len(chars)*2, 2))
    if len(chars) == 1:
        axes = [axes]
    for i, cimg in enumerate(chars):
        axes[i].imshow(cimg, cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'all_chars.png'))
    plt.close() 