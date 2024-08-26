import cv2

img1 = cv2.imread('SIFT_ORM_SURF/photo_24.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('SIFT_ORM_SURF/photo_111.jpg', cv2.IMREAD_GRAYSCALE)

# Инициализация детектора и дескриптора SIFT
sift = cv2.SIFT_create()

# Нахождение ключевых точек и дескрипторов
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Инициализация матчера дескрипторов (FLANN-based)
index_params = dict(algorithm=1, trees=5)  # Используем алгоритм KDTree
search_params = dict(checks=50)  # Число проверок для поиска ближайших соседей

flann = cv2.FlannBasedMatcher(index_params, search_params)

# Сопоставление дескрипторов с использованием KNN (k=2)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Применение теста соотношения по Лоу
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# Визуализация совпадений
matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Сохранение результата в файл
cv2.imwrite('sift_matched_keypoints.jpg', matched_img)

# Показ результата на экране
cv2.imshow('Matched Keypoints with SIFT', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
