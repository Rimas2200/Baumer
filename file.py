import cv2
import numpy as np

# Загрузка изображений
img1_path = 'calibrated_image.jpg'
img2_path = 'path_to_images/photo_110.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None:
    raise ValueError(f"Не удалось загрузить изображение: {img1_path}")
if img2 is None:
    raise ValueError(f"Не удалось загрузить изображение: {img2_path}")

if img1.shape[:2] != img2.shape[:2]:
    print(f"Размер изображения 1: {img1.shape[:2]}")
    print(f"Размер изображения 2: {img2.shape[:2]}")
    raise ValueError("Изображения должны быть одинакового размера для наложения")

def blend_images(img1, img2, alpha):
    beta = 1.0 - alpha
    blended = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    return blended

alpha = 0.5
blended_image = blend_images(img1, img2, alpha)

cv2.imshow('Blended Image', blended_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('blended_image.jpg', blended_image)
