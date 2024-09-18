import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import open3d as o3d

# image_folder = r'test_file\test2'
image_folder = r'output_images2'

all_keypoints = []
all_descriptors = []
all_colors = []

sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=1000)

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

reference_keypoints = all_keypoints[0]
reference_descriptors = all_descriptors[0]
reference_colors = all_colors[0]

# Матчер дескрипторов (FLANN-based)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=70)
flann = cv2.FlannBasedMatcher(index_params, search_params)

points3D = []
colors3D = []

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

        ref_color = reference_colors[match.queryIdx]
        colors3D.append(ref_color)

points3D = np.array(points3D)
colors3D = np.array(colors3D) / 255.0

if points3D.shape[1] == 4:
    points3D = points3D[:, :3]

# Создание выпуклой оболочки
hull = ConvexHull(points3D)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Отображение точек
ax.scatter(points3D[:, 0], points3D[:, 1], points3D[:, 2], c=colors3D, s=1)

# Отображение выпуклой оболочки
for simplex in hull.simplices:
    poly3d = [[points3D[simplex[i]] for i in range(len(simplex))]]
    ax.add_collection3d(Poly3DCollection(poly3d, alpha=.125, linewidths=1, edgecolors='r'))
# Создание облака точек Open3D
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points3D)
point_cloud.colors = o3d.utility.Vector3dVector(colors3D)

# Сохранение облака точек в формате PLY
o3d.io.write_point_cloud("point_cloud.ply", point_cloud)

# Создание выпуклой оболочки в виде сетки Open3D
vertices = points3D[hull.vertices]
triangles = hull.simplices

# Создание треугольной сетки (TriangleMesh)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(points3D)
mesh.triangles = o3d.utility.Vector3iVector(triangles)

mesh.vertex_colors = o3d.utility.Vector3dVector(colors3D)

# Сохранение выпуклой оболочки в формате PLY
o3d.io.write_triangle_mesh("convex_hull.ply", mesh)

print("Облако точек и выпуклая оболочка успешно сохранены в формате PLY.")

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig('3D_point_cloud_with_convex_hull.png', dpi=300)
# o3d.io.write_triangle_mesh("3D_object_poisson_simplified.ply", mesh_simplified)
plt.show()

