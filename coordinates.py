import numpy as np
import cv2
from matplotlib import pyplot as plt

# Параметры шахматной доски
pattern_size = (9, 6)  # количество внутренних углов шахматной доски (высота, ширина)

# Создание массивов для хранения объектных и изображенных точек
obj_points = []  # точки на объекте (шахматная доска)
img_points_left = []  # точки на изображении для левой камеры
img_points_right = []  # точки на изображении для правой камеры

# Генерация координат объектных точек (шахматная доска)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[1], 0:pattern_size[0]].T.reshape(-1, 2)

# Загрузка изображений с шахматной доской для калибровки
left_img = cv2.imread('test/position_1_field.jpg')
right_img = cv2.imread('test/position_2_field.jpg')

# Преобразование в градации серого
gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

# Поиск углов шахматной доски
ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)

# Визуализация найденных углов
if ret_left:
    cv2.drawChessboardCorners(left_img, pattern_size, corners_left, ret_left)
else:
    print('Не удалось найти углы на левой шахматной доске.')

if ret_right:
    cv2.drawChessboardCorners(right_img, pattern_size, corners_right, ret_right)
else:
    print('Не удалось найти углы на правой шахматной доске.')

# Отображение изображений с углами шахматной доски
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

if ret_left:
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('position_1')
    axes[0].axis('off')
else:
    axes[0].set_title('position_1')
    axes[0].axis('off')

if ret_right:
    axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title('position_2')
    axes[1].axis('off')
else:
    axes[1].set_title('position_2')
    axes[1].axis('off')

plt.show()

# Если углы найдены, добавляем их в список
if ret_left and ret_right:
    obj_points.append(objp)
    img_points_left.append(corners_left)
    img_points_right.append(corners_right)
print("left", corners_left)
print("right", corners_right)

# Калибровка камер
ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(obj_points, img_points_left, gray_left.shape[::-1], None, None)
ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(obj_points, img_points_right, gray_right.shape[::-1], None, None)

# Стереокалибровка (определение матрицы преобразования)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
stereocalib_criteria = (cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS, 30, 0.001)  # Исправленные критерии

# Подготовка данных для стереокалибровки
flags = (cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS)
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
flags_stereo = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS

ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right, K_left, dist_left, K_right, dist_right, gray_left.shape[::-1], criteria=criteria_stereo, flags=flags_stereo)

# Вычисление матрицы проекции (нормализация)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, D1, K2, D2, gray_left.shape[::-1], R, T)

# Вычисление карты смещений (disparity map)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(gray_left, gray_right)

# Проекция изображений в трехмерное пространство
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# Субдискретизация точек для уменьшения количества
step = 10  # Измените это значение для уменьшения количества точек
subsampled_points_3d = points_3d[::step, ::step]
subsampled_colors = gray_left[::step, ::step].flatten()

# Отображение облака точек
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(subsampled_points_3d[:, :, 0].flatten(), subsampled_points_3d[:, :, 1].flatten(), subsampled_points_3d[:, :, 2].flatten(), c=subsampled_colors, cmap='gray', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


new_img_1 = cv2.imread('test/position_1_object.jpg')
new_img_2 = cv2.imread('test/position_2_object.jpg')

# Преобразование в градации серого
gray_new_1 = cv2.cvtColor(new_img_1, cv2.COLOR_BGR2GRAY)
gray_new_2 = cv2.cvtColor(new_img_2, cv2.COLOR_BGR2GRAY)

# Визуализация найденных углов на новых изображениях
cv2.drawChessboardCorners(new_img_1, pattern_size, corners_left, True)
cv2.drawChessboardCorners(new_img_2, pattern_size, corners_right, True)

# Отображение изображений с углами шахматной доски
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(cv2.cvtColor(new_img_1, cv2.COLOR_BGR2RGB))
axes[0].set_title('New position 1')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(new_img_2, cv2.COLOR_BGR2RGB))
axes[1].set_title('New position 2')
axes[1].axis('off')

plt.show()