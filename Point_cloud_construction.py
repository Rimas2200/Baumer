import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Путь к папке с изображениями
# image_folder = r'test_file\test2'
image_folder = r'output_images2'

# Список для хранения всех ключевых точек, дескрипторов и цветов
all_keypoints = []
all_descriptors = []
all_colors = []

sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=1000)

# Чтение всех изображений
images = glob.glob(f'{image_folder}/*.jpg')
for img_path in images:
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    all_keypoints.append(keypoints)
    all_descriptors.append(descriptors)

    # Сохранение цвета ключевых точек
    img_colors = [img[int(kp.pt[1]), int(kp.pt[0])] for kp in keypoints]
    all_colors.append(img_colors)

# Выбираем первое изображение как эталонное
reference_keypoints = all_keypoints[0]
reference_descriptors = all_descriptors[0]
reference_colors = all_colors[0]

# Матчер дескрипторов (FLANN-based)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=70)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Список для хранения всех совпадений и цветов
points3D = []
colors3D = []

# Перебор всех изображений, начиная со второго
for i in range(1, len(images)):
    matches = flann.knnMatch(reference_descriptors, all_descriptors[i], k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    for match in good_matches:
        ref_pt = reference_keypoints[match.queryIdx].pt
        img_pt = all_keypoints[i][match.trainIdx].pt
        points3D.append([ref_pt[0], ref_pt[1], img_pt[0], img_pt[1]])

        # Сохранение цвета ключевых точек
        ref_color = reference_colors[match.queryIdx]
        colors3D.append(ref_color)

# Преобразование списка в массивы
points3D = np.array(points3D)
colors3D = np.array(colors3D) / 255.0

# Визуализация облака точек
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c=colors3D, s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('3D_point_cloud.png', dpi=300)
plt.show()
