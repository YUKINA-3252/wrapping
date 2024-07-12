import copy
import glob
import math
import numpy as np
import open3d as o3d
import os
import pyransac3d as pyrsc
import re

def add_color_normal (pcd):
    # pcd.paint_uniform_color(np.random.rand(3))
    pcd.paint_uniform_color(np.array([255/255, 182/255, 193/255]))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(kdt_n)


def load_pcds(pcd_files):
    pcds = []
    for i in range(len(pcd_files)):
        pcd = pcd_files[i]
        add_color_normal(pcd)
        pcds.append(pcd)

    return pcds


def load_pcds_path(pcd_files):
    pcds = []
    for f in pcd_files:
        pcd = o3d.io.read_point_cloud(f)
        add_color_normal(pcd)
        pcds.append(pcd)

    return pcds


def filter_points_by_distance(points, p, r):
    distances = np.linalg.norm(points - p, axis=1)
    indices = np.where(distances <= r)[0]
    return indices


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("/home/iwata/wrapping_ws/src/wrapping/wrapping_melodic/save_pcd/1709452160477962.pcd")
    pcd.remove_non_finite_points()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))
    pcd_np = np.asarray(pcd.points)
    normals_np = np.asarray(pcd.normals)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping_melodic/scripts"
    file_name = "array_data.txt"
    file_path = os.path.join(directory_path, file_name)
    with open(file_path, 'r') as file:
        p = [float(line.strip()) for line in file.readlines()][:3]
    r = 0.05
    filtered_points_indices = filter_points_by_distance(pcd_np, p, r)
    filtered_normals_np = normals_np[filtered_points_indices]
    mean_normal_vector = np.mean(filtered_normals_np, axis=0) / np.linalg.norm(np.mean(filtered_normals_np, axis=0))

    z_axis = np.array([0, 0, 1])
    cos_theta = np.dot(mean_normal_vector, z_axis)
    rotate_angle = np.degrees(np.arccos(cos_theta))
    print(rotate_angle)

    theta_z = np.arccos(mean_normal_vector[2])
    R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                    [np.sin(theta_z), np.cos(theta_z), 0],
                    [0, 0, 1]])
    projection = np.array([1, 0, 0])
    direction_after_z_rotation = np.dot(R_z, mean_normal_vector)
    theta_x = np.arccos(np.dot(projection, direction_after_z_rotation) / (np.linalg.norm(projection) * np.linalg.norm(direction_after_z_rotation)))
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])
    final_rotation_matrix = np.dot(R_x, R_z)
    print(mean_normal_vector, final_rotation_matrix)
