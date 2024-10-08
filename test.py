import cv2
import numpy as np
import glob
import open3d as o3d
import matplotlib.pyplot as plt

image_folder = r'test_file\test2'
image_folder = r'output_images3'

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

        points3D.append([ref_pt[0], ref_pt[1], img_pt[0]])

        ref_color = reference_colors[match.queryIdx]
        colors3D.append(ref_color)

points3D = np.array(points3D)
colors3D = np.array(colors3D) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3D)
pcd.colors = o3d.utility.Vector3dVector(colors3D)

# Оценка нормалей с увеличенным радиусом поиска соседей
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))

# Восстановление поверхности методом Пуассона с уменьшенным depth
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)

vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)

voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 5
mesh_simplified = mesh.simplify_vertex_clustering(voxel_size=voxel_size)

o3d.visualization.draw_geometries([mesh_simplified])

o3d.io.write_triangle_mesh("3D_object_poisson_simplified.ply", mesh_simplified)
