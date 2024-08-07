import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Параметры калибровочного паттерна
chessboard_size = (9, 6)  # Размер внутренней сетки шахматной доски (9x6)
square_size = 2.0  # Размер квадрата в реальных единицах

# Критерии для остановки алгоритма нахождения углов на шахматной доске
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Подготовка объектов (координаты точек шахматной доски в реальном мире)
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

# Списки для хранения точек в мире и на изображениях
objpoints = []
imgpoints = []

images = glob.glob('path_to_images/*.jpg')

# Размер для уменьшения изображения при отображении
resize_factor = 0.5

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Найти углы шахматной доски
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Улучшить координаты углов
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints[-1] = corners2

        # Отобразить углы на уменьшенном изображении
        img_small = cv2.resize(img, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
        corners2_small = corners2 * resize_factor
        img_small = cv2.drawChessboardCorners(img_small, chessboard_size, corners2_small, ret)
        cv2.imshow('img', img_small)
        cv2.waitKey(1)

cv2.destroyAllWindows()

# Калибровка камеры
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Калибровочная матрица K:")
print(K)
print("\nКоэффициенты дисторсии:")
print(dist_coeffs)

# Минимизация функции потерь и уточнение параметров
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], K, dist_coeffs, criteria=criteria)

print("Уточненная калибровочная матрица K:")
print(K)
print("\nУточненные коэффициенты дисторсии:")
print(dist_coeffs)

# Загрузить одно из изображений для демонстрации
img = cv2.imread(images[0])
h, w = img.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))

# Коррекция искажения
dst = cv2.undistort(img, K, dist_coeffs, None, new_K)

# Обрезка изображения
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

# Приведение изображений к одному размеру
img_resized = cv2.resize(img, (dst.shape[1], dst.shape[0]))

# Наложение изображений с альфа-каналом
alpha = 0.5
overlay = cv2.addWeighted(img_resized, alpha, dst, 1 - alpha, 0)

# Сохранение наложенного изображения
cv2.imwrite('overlay_image.jpg', overlay)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('')
plt.axis('off')
plt.show()

cv2.destroyAllWindows()
