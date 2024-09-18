import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузите изображение в цвете. Укажите путь фотографии
image_bgr = cv2.imread('C:/Users/andre/Desktop/D/Program/1.jpg')

# Преобразуйте изображение из BGR в RGB
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Создайте объект SIFT
sift = cv2.SIFT_create()

# Обнаружьте ключевые точки и вычислите дескрипторы
keypoints, descriptors = sift.detectAndCompute(image_rgb, None)

# Создайте белый холст того же размера, что и исходное изображение
white_background = np.ones_like(image_rgb) * 255

# Нарисуйте точки на белом фоне
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    cv2.circle(white_background, (x, y), 1, (0, 0, 0), -1)  # Черные точки

# Покажите результат
plt.figure(figsize=(10, 10))
plt.imshow(white_background)
plt.axis('off')
plt.title('Keypoints detected by SIFT')
plt.show()
