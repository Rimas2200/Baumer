import cv2

img1 = cv2.imread('test/position_1_field.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('test/position_2_field.jpg', cv2.IMREAD_GRAYSCALE)

# Предварительная обработка изображений
img1 = cv2.equalizeHist(img1)
img2 = cv2.equalizeHist(img2)

# Инициализация детектора и дескриптора ORB с изменением параметров
orb = cv2.ORB_create(nfeatures=1000)  # Увеличение количества ключевых точек

# Нахождение ключевых точек и дескрипторов
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

# Визуализация ключевых точек
img1_kp = cv2.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0))
img2_kp = cv2.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0))
cv2.imshow('Keypoints 1', img1_kp)
cv2.imwrite('1.jpg', img1_kp)
cv2.imshow('Keypoints 2', img2_kp)
cv2.imwrite('2.jpg', img2_kp)
cv2.waitKey(0)

# Инициализация матчера дескрипторов (Hamming)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Сопоставление дескрипторов
matches = bf.match(descriptors1, descriptors2)

# Сортировка совпадений по расстоянию
matches = sorted(matches, key=lambda x: x.distance)

# Визуализация совпадений
matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('matched_keypoints_improved.jpg', matched_img)
cv2.imshow('Matched Keypoints', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
