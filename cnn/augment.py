import os
import cv2
import numpy as np
import imgaug.augmenters as ia
import PIL

src = "C:/work/ocr-text-recognition/datasets/handwritten/images/00000000.png"

# cv2로 이미지 읽기
img = cv2.imread(src)

# imgaug 적용할 필터 설정
seq = ia.Sequential([
    ia.Add((-40, 40)),
    ia.Fliplr(0.5),
    ia.GaussianBlur(sigma=(0, 3.0)),
    ia.Sharpen(alpha=(0, 1), lightness=(0.75, 2)),
    ia.ChangeColorTemperature((1100, 10000))
])


for _ in range(10):
    # image augmentation 적용
    img_aug = seq.augment_image(img)

    # 결과 보기
    h, w = img.shape[:2]
    zeros_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    zeros_img[:, :w, :] = img
    zeros_img[:, w:, :] = img_aug
    cv2.imshow("result", zeros_img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        exit()

# opencv로 이미지를 읽게 되면 PIL로 변환을 다시 해줘야 된다.
img = PIL.Image.fromarray(img)