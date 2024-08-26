import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Путь к папке с изображениями
image_folder = 'C:/Users/andre/Desktop/D/Program/Photo/'

# Список файлов изображений (тестовый набор из 30 изображений)
image_files = [f'{image_folder}{i+1}.jpg' for i in range(30)]

# Создайте объект SIFT с ограничением на количество ключевых точек
sift = cv2.SIFT_create(nfeatures=500)  # Ограничьте количество ключевых точек

# Создайте списки для хранения всех ключевых точек и их цветов
all_keypoints = []
colors = []

for image_path in image_files:
    # Загрузите изображение в цвете
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f'Не удалось загрузить изображение: {image_path}')
        continue

    # Преобразуйте изображение из BGR в RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Обнаружьте ключевые точки и вычислите дескрипторы
    keypoints, descriptors = sift.detectAndCompute(image_rgb, None)

    # Отбор ключевых точек по размеру (например, исключить очень маленькие ключевые точки)
    min_size = 10  # Минимальный размер ключевых точек для включения
    keypoints = [kp for kp in keypoints if kp.size > min_size]

    # Добавьте ключевые точки и их цвета в списки
    for kp in keypoints:
        all_keypoints.append(kp)
        # Получите цвет пикселя вокруг ключевой точки
        x, y = int(kp.pt[0]), int(kp.pt[1])
        color = image_rgb[y, x, :]  # RGB
        colors.append(color)

# Извлеките координаты x, y и размер ключевых точек
x = np.array([kp.pt[0] for kp in all_keypoints])
y = np.array([kp.pt[1] for kp in all_keypoints])
z = np.array([kp.size for kp in all_keypoints])

# Нормализация x и y координат в диапазон [0, 1]
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1)).flatten()
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Логарифмическое масштабирование оси Z для уменьшения скученности
z_scaled = np.log1p(z)  # log1p используется для учета значения 0

# Создайте 3D-график
fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111, projection='3d')

# Преобразуем цвета в нормализованный формат для отображения в matplotlib
colors_normalized = np.array(colors) / 255.0

# Нарисуйте ключевые точки в 3D, используя нормализованные данные
ax.scatter(x_scaled, y_scaled, z_scaled, c=colors_normalized, marker='o', s=10)

ax.set_xlabel('Normalized X Coordinate')
ax.set_ylabel('Normalized Y Coordinate')
ax.set_zlabel('Log-Scaled Size')

plt.title('3D Visualization of Keypoints Detected by SIFT')

# Отобразите график на весь экран
fig.tight_layout()
plt.show()


