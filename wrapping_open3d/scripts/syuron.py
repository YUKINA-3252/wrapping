#!/usr/bin/env python

import copy
import numpy as np
from math import acos, degrees
import open3d as o3d
import os
from scipy.spatial import cKDTree


def add_color_normal (pcd, paper):
    # pcd.paint_uniform_color(np.random.rand(3))
    if paper == "paper":
        pcd.paint_uniform_color(np.array([255/255, 182/255, 193/255]))
    else:
        pcd.paint_uniform_color(np.array([157/255, 204/255, 224/255]))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(kdt_n)

if __name__ == "__main__":
    # source = box pcd
    box_directory_path = "/home/iwata/Downloads/Scaniverse_2025-01-19_181539.ply"
    # box_directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/box/merge.pcd"
    source = o3d.io.read_point_cloud(box_directory_path)
    source.remove_non_finite_points()
    source_np = np.asarray(source.points)
    add_color_normal(source, "box")
    o3d.visualization.draw_geometries([source])

    #target = paper pcd
    paper_directory_path = "/home/iwata/Downloads/Scaniverse_2025-01-19_172232.ply"
    # paper_directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/neat_wrapping/merge.pcd"
    target = o3d.io.read_point_cloud(paper_directory_path)
    target.remove_non_finite_points()
    target_np = np.asarray(target.points)
    add_color_normal(target, "paper")
    o3d.visualization.draw_geometries([target])

    initial_transform = np.eye(4)

    # ICPマッチングの実行
    threshold = 0.05  # 最大対応距離（調整可能）
    icp_result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    source.transform(icp_result.transformation)
    o3d.visualization.draw_geometries([source, target], window_name="ICP Matching Result")

    # detect table for source
    distance_threshold = 0.008  # 平面検出の閾値（調整可能）
    ransac_n = 3  # RANSACでの最小点数
    num_iterations = 1000  # RANSACの最大試行回数

    # 平面の検出とインライアの記録
    largest_inlier_indices = []
    largest_inlier_count = 0

    # 点群を複製して処理用に使用
    remaining_cloud = source

    while True:
        # 平面を検出
        plane_model, inlier_indices = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # 平面の点群（インライア）とそれ以外（アウトライア）に分離
        inlier_cloud = remaining_cloud.select_by_index(inlier_indices)
        outlier_cloud = remaining_cloud.select_by_index(inlier_indices, invert=True)

        # インライアが十分に大きい場合だけ処理
        inlier_count = len(inlier_indices)
        if inlier_count > largest_inlier_count:
            largest_inlier_indices = inlier_indices
            largest_inlier_count = inlier_count
        # アウトライアがなくなるか、インライアが小さすぎたら終了
        if inlier_count < 100 or len(remaining_cloud.points) == len(inlier_indices):
            break

        # 残りの点群に対して繰り返し処理
        remaining_cloud = remaining_cloud.select_by_index(inlier_indices, invert=True)

    largest_inlier_cloud = source.select_by_index(largest_inlier_indices)
    only_box_source= source.select_by_index(largest_inlier_indices, invert=True)
    clean_source_radius, ind_radius = only_box_source.remove_radius_outlier(
        nb_points=30,     # 必要な最小近傍点数
        radius=0.01       # 半径
    )

    # detect table for target
    distance_threshold = 0.005  # 平面検出の閾値（調整可能）
    ransac_n = 3  # RANSACでの最小点数
    num_iterations = 1000  # RANSACの最大試行回数

    # 平面の検出とインライアの記録
    largest_inlier_indices = []
    largest_inlier_count = 0

    # 点群を複製して処理用に使用
    remaining_cloud = target

    while True:
        # 平面を検出
        plane_model, inlier_indices = remaining_cloud.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations
        )

        # 平面の点群（インライア）とそれ以外（アウトライア）に分離
        inlier_cloud = remaining_cloud.select_by_index(inlier_indices)
        outlier_cloud = remaining_cloud.select_by_index(inlier_indices, invert=True)

        # インライアが十分に大きい場合だけ処理
        inlier_count = len(inlier_indices)
        if inlier_count > largest_inlier_count:
            largest_inlier_indices = inlier_indices
            largest_inlier_count = inlier_count
        # アウトライアがなくなるか、インライアが小さすぎたら終了
        if inlier_count < 100 or len(remaining_cloud.points) == len(inlier_indices):
            break

        # 残りの点群に対して繰り返し処理
        remaining_cloud = remaining_cloud.select_by_index(inlier_indices, invert=True)

    largest_inlier_cloud = target.select_by_index(largest_inlier_indices)
    only_box_target = target.select_by_index(largest_inlier_indices, invert=True)
    clean_target_radius, ind_radius = only_box_target.remove_radius_outlier(
        nb_points=30,     # 必要な最小近傍点数
        radius=0.01       # 半径
    )

    o3d.io.write_point_cloud("clean_source_radius.ply", clean_source_radius)
    o3d.io.write_point_cloud("clean_target_radius.ply", clean_target_radius)

    source = o3d.io.read_point_cloud("clean_source_radius.ply")
    source, ind = source.remove_radius_outlier(nb_points=30, radius=0.05)
    source.remove_non_finite_points()
    source_np = np.asarray(source.points)
    add_color_normal(source, "box")

    target = o3d.io.read_point_cloud("clean_target_radius.ply")
    target, ind = target.remove_radius_outlier(nb_points=30, radius=0.05)
    target.remove_non_finite_points()
    target_np = np.asarray(target.points)
    add_color_normal(target, "paper")

    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))

    o3d.visualization.draw_geometries([source], point_show_normal=True)
    o3d.visualization.draw_geometries([target], point_show_normal=True)

    k = 5
    kdtree_source = cKDTree(source_np)
    angles = []
    normals_source = np.asarray(source.normals)
    normals_target = np.asarray(target.normals)

    tree = o3d.geometry.KDTreeFlann()
    tree.set_geometry(o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(source_np)))

    # 近傍点ごとの法線を計算して、なす角度を計算
    for target_idx, target_point in enumerate(target_np):
        # target点群の各点bについて、点群Aの近傍点を取得
        # _, idx_source = kdtree_source.query(target_point, k=k)
        # source_neigh = source_np[idx_source]

        # target点群の法線
        normal_target = normals_target[target_idx]

        # 近傍点の法線を取得
        # source_neigh = source.select_by_index(idx_source)
        _, idxs, _ = tree.search_knn_vector_3d(target_point, k)
        # source_neigh.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=10))
        # normals_source_neigh = np.asarray(source_neigh.normals)
        normals_source_neigh = normals_source[idxs]
        normal_source_neigh_average = np.mean(normals_source_neigh, axis=0)

        # # 各近傍点との法線のなす角度を計算
        # angle_neigh =np.empty(k)
        # for i, normal_source in enumerate(normals_source_neigh):
        #     angle = np.arccos(np.clip(np.dot(normal_target, normal_source), -1.0, 1.0))  # 法線ベクトルの内積を利用して角度計算
        #     angle_deg = degrees(angle)  # ラジアンから度に変換
        #     # angles.append(angle_deg)
        #     angle_neigh[i] = angle_deg
        # average_angle = angle_neigh.mean(axis=0)
        # if average_angle > 90:
        #     average_angle = 180 - average_angle
        # angles.append(average_angle)
        angle = np.arccos(np.clip(np.dot(normal_target, normal_source_neigh_average), -1.0, 1.0))  # 法線ベクトルの内積を利用して角度計算
        angle_deg = degrees(angle)  # ラジアンから度に変換
        if angle_deg > 90:
            angle_deg = 180 - angle_deg
        angles.append(angle_deg)

    angles = np.array(angles)
    print(angles)
    condition = (angles > 20) & (angles < 70)
    ratio = np.sum(condition) / len(angles)
    print(ratio)


    # mesh = o3d.io.read_triangle_mesh(directory_path)
    # mesh.compute_vertex_normals()
    # vertices = np.asarray(mesh.vertices)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = mesh.vertices
    # add_color_normal(pcd)
    # normals = np.asarray(mesh.vertex_normals)

    # for i, (vertex, normal) in enumerate(zip(vertices, normals)):
    #     print(f"Vertex {i}: {vertex}, Normal: {normal}")

    # lines = []  # 法線を表す線の始点と終点のペア
    # colors = []  # 各線の色（任意で設定）
    # normal_length = 0.05  # 法線の長さ（調整可能）

    # for i, (vertex, normal) in enumerate(zip(vertices, normals)):
    #     start = vertex  # 法線の始点（頂点）
    #     end = vertex + normal * normal_length  # 法線の終点
    #     lines.append([start, end])
    #     colors.append([1, 0, 0])  # 法線の色を赤色に設定（RGB値）

    # # 法線を矢印として描画
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(np.vstack(lines))  # 線の始点と終点を設定
    # line_set.lines = o3d.utility.Vector2iVector([[i * 2, i * 2 + 1] for i in range(len(vertices))])  # 線の接続情報
    # line_set.colors = o3d.utility.Vector3dVector(colors)  # 各線の色

    # o3d.visualization.draw_geometries([box_pcd])
    # o3d.visualization.draw_geometries([paper_pcd])
    # o3d.visualization.draw_geometries([mesh, line_set])

    # kdtree = o3d.geometry.KDTreeFlann(pcd)
    # radius = 0.1
    # print(pcd_np[:, 2])
