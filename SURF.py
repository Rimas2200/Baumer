import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import KDTree

# def finite_difference_derivative(I, axis):
#     if axis == 0:  # Производная по x
#         return (np.roll(I, -1, axis=axis) - np.roll(I, 1, axis=axis)) / 2
#     elif axis == 1:  # Производная по y
#         return (np.roll(I, -1, axis=axis) - np.roll(I, 1, axis=axis)) / 2
#     else:
#         raise ValueError("Axis should be 0 (x-axis) or 1 (y-axis).")
#
# # Функция для вычисления матрицы Гессиана (вторые частные производные изображения)
# def hessian_matrix(I, sigma):
#     # Первая производная по x и y
#     Ix = finite_difference_derivative(I, axis=0)
#     Iy = finite_difference_derivative(I, axis=1)
#     # Вторая производная по x
#     Ixx = finite_difference_derivative(Ix, axis=0)
#     # Вторая производная по y
#     Iyy = finite_difference_derivative(Iy, axis=1)
#     # Смешанная производная (по x, по y)
#     Ixy = finite_difference_derivative(Ix, axis=1)
#
#     return Ixx, Iyy, Ixy

# Функция для вычисления матрицы Гессиана (вторые частные производные изображения)
def hessian_matrix(I, sigma):
    # Ixx: Вторая производная по x
    Ixx = gaussian_filter(I, sigma=sigma, order=(2, 0))
    # Iyy: Вторая производная по y
    Iyy = gaussian_filter(I, sigma=sigma, order=(0, 2))
    # Ixy: Смешанная производная (сначала по x, затем по y)
    Ixy = gaussian_filter(I, sigma=sigma, order=(1, 1))
    return Ixx, Iyy, Ixy

# Функция для вычисления детерминанта матрицы Гессиана
def determinant_hessian(Ixx, Iyy, Ixy):
    # Определяем детерминант матрицы Гессиана: det(H) = Ixx * Iyy - Ixy^2
    return Ixx * Iyy - Ixy ** 2

# Функция для обнаружения ключевых точек на изображении
def detect_keypoints(I, hessian_threshold, sigma=1.0):
    # Вычисляем компоненты матрицы Гессиана
    Ixx, Iyy, Ixy = hessian_matrix(I, sigma)
    # Вычисляем детерминант матрицы Гессиана
    detH = determinant_hessian(Ixx, Iyy, Ixy)

    # Находим позиции ключевых точек, где детерминант превышает пороговое значение
    keypoints = np.argwhere(detH > hessian_threshold)
    return keypoints

# Функция для вычисления дескрипторов для ключевых точек
def compute_descriptors(I, keypoints, patch_size=16):
    descriptors = []
    for y, x in keypoints:
        # Извлекаем патч изображения вокруг ключевой точки
        patch = I[y - patch_size // 2:y + patch_size // 2, x - patch_size // 2:x + patch_size // 2]
        # Проверяем, что патч имеет нужный размер
        if patch.shape == (patch_size, patch_size):
            # Вычисляем производные изображения по x и y с использованием оператора Собеля
            Ix = sobel(patch, axis=1)
            Iy = sobel(patch, axis=0)
            # Вычисляем величину и ориентацию градиента
            magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
            orientation = np.arctan2(Iy, Ix)
            # Формируем гистограмму ориентирований для патча
            descriptor = np.histogram(orientation, bins=8, weights=magnitude, range=(-np.pi, np.pi))[0]
            # Нормализуем дескриптор
            norm = np.linalg.norm(descriptor)
            if norm != 0:  # Проверяем, что норма не равна нулю
                descriptors.append(descriptor / norm)
    return np.array(descriptors)

# Функция для сопоставления дескрипторов с использованием метода k ближайших соседей (kNN)
def knn_match(descriptors1, descriptors2, k=2):
    # Проверяем, что массивы дескрипторов не пусты
    if descriptors1.size == 0 or descriptors2.size == 0:
        raise ValueError("One of the descriptors arrays is empty.")
    # Проверяем, что дескрипторы имеют правильную форму (двумерный массив)
    if len(descriptors1.shape) != 2 or len(descriptors2.shape) != 2:
        raise ValueError("Descriptors must be a 2D array of shape (n, m).")

    # Создаем KD-дерево для быстрого поиска ближайших соседей
    tree = KDTree(descriptors2)
    # Выполняем поиск k ближайших соседей для каждого дескриптора
    distances, indices = tree.query(descriptors1, k=k)
    matches = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        matches.append((i, idx))
    return matches

# Функция для применения теста соотношения по Лоу
def lowe_ratio_test(matches, descriptors1, descriptors2, ratio=0.7):
    good_matches = []
    for (i, [j1, j2]) in matches:
        # Вычисляем расстояния до двух ближайших соседей
        d1 = np.linalg.norm(descriptors1[i] - descriptors2[j1])
        d2 = np.linalg.norm(descriptors1[i] - descriptors2[j2])
        # Применяем тест соотношения по Лоу
        if d1 < ratio * d2:
            good_matches.append((i, j1))
    return good_matches

# Функция для визуализации сопоставленных точек на двух изображениях
def draw_matches(img1, keypoints1, img2, keypoints2, good_matches):
    # Преобразуем изображения в цветные для отображения цветных линий
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    # Объединяем два изображения в одно
    img_out = np.hstack((img1_color, img2_color))

    # Соединение совпавших точек
    for (i, j) in good_matches:
        pt1 = tuple(keypoints1[i][::-1])
        pt2 = tuple(keypoints2[j][::-1] + np.array([img1.shape[1], 0]))
        cv2.line(img_out, pt1, pt2, color=(0, 255, 0), thickness=1)

    return img_out


img1 = cv2.imread('SIFT_ORM_SURF/photo_24.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('SIFT_ORM_SURF/photo_111.jpg', cv2.IMREAD_GRAYSCALE)

# Порог для детекции ключевых точек
hessian_threshold = 252

# Нахождение ключевых точек и дескрипторов
keypoints1 = detect_keypoints(img1, hessian_threshold)
keypoints2 = detect_keypoints(img2, hessian_threshold)

# Количество найденных ключевых точек
print(f"Found {len(keypoints1)} keypoints in image 1.")
print(f"Found {len(keypoints2)} keypoints in image 2.")

# Визуализация ключевых точек
for y, x in keypoints1:
    cv2.circle(img1, (x, y), 3, (255, 0, 0), 1)
for y, x in keypoints2:
    cv2.circle(img2, (x, y), 3, (255, 0, 0), 1)

cv2.imshow("Keypoints in Image 1", img1)
cv2.imwrite("KeypointsinImage1.jpg", img1)
cv2.imshow("Keypoints in Image 2", img2)
cv2.imwrite("KeypointsinImage2.jpg", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

descriptors1 = compute_descriptors(img1, keypoints1)
descriptors2 = compute_descriptors(img2, keypoints2)

if descriptors1.size == 0 or descriptors2.size == 0:
    raise ValueError("No descriptors found.")

# Сопоставление дескрипторов с использованием KNN
matches = knn_match(descriptors1, descriptors2, k=2)

# Применение теста соотношения по Лоу
good_matches = lowe_ratio_test(matches, descriptors1, descriptors2, ratio=0.7)

# Визуализация совпадений
matched_img = draw_matches(img1, keypoints1, img2, keypoints2, good_matches)

cv2.imwrite('surf_matched_keypoints.jpg', matched_img)
cv2.imshow('Matched Keypoints with SURF', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
