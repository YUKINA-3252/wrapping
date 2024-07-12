import copy
import glob
import math
import numpy as np
import open3d as o3d
import os
import pyransac3d as pyrsc
from scipy.spatial.distance import cdist
import re

def add_color_normal (pcd):
    # pcd.paint_uniform_color(np.random.rand(3))
    pcd.paint_uniform_color(np.array([255/255, 182/255, 193/255]))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(kdt_n)

def register (pcd1, pcd2, size):
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=size*10, max_nn=50)

    # down sampling
    pcd1_d = pcd1.voxel_down_sample(size)
    pcd2_d = pcd2.voxel_down_sample(size)
    pcd1_d.estimate_normals(kdt_n)
    pcd2_d.estimate_normals(kdt_n)

    # Feature amount calculation
    pcd1_f = o3d.pipelines.registration.compute_fpfh_feature(pcd1_d, kdt_f)
    pcd2_f = o3d.pipelines.registration.compute_fpfh_feature(pcd2_d, kdt_f)

    checker = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
               o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

    est_ptp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    est_ptpln = o3d.pipelines.registration.TransformationEstimationPointToPlane()

    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)

    #RANSAC
    result1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(pcd1_d, pcd2_d, pcd1_f, pcd2_f, True,
                                                                                       size * 2, est_ptp, 4, checker, criteria)
    # ICP
    result2 = o3d.pipelines.registration.registration_icp(pcd1, pcd2, size, result1.transformation, est_ptpln)

    return result2.transformation


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


def align_pcds(pcds, size):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    accum_pose = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(accum_pose))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            source = pcds[source_id]
            target = pcds[target_id]
            trans = register(source, target, size)
            GTG_mat = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, size, trans)

            if target_id == source_id + 1:
                accum_pose = trans @ accum_pose
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(accum_pose)))

            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, trans, GTG_mat, uncertain=True))

    solver = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
    option = o3d.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=size / 10, edge_prune_threshold=size / 10, reference_node=0)

    # optimization
    o3d.pipelines.registration.global_optimization(pose_graph, method=solver, criteria=criteria, option=option)

    for pcd_id in range(n_pcds):
        trans = pose_graph.nodes[pcd_id].pose
        pcds[pcd_id].transform(trans)

    return pcds


def group_coordinates(coordinates, num_points_per_group):
    num_points = len(coordinates)
    groups = []

    visited = set()

    for i in range(num_points):
        if i in visited:
            continue

        group = [i]
        visited.add(i)

        while len(group) < num_points_per_group:
            min_distance = np.inf
            min_index = None

            for j in range(num_points):
                if j in visited:
                    continue

                distances = cdist(coordinates[j].reshape(1, -1), coordinates[list(visited)])
                min_distance_to_group = np.min(distances)
                if min_distance_to_group < min_distance:
                    min_distance = min_distance_to_group
                    min_index = j

            if min_index is None:
                break

            group.append(min_index)
            visited.add(min_index)

        groups.append(group)
    return groups


def extract_closest_indices(A, B, k=10):
    extracted_indices = []

    for point_A in A:
        distances = np.linalg.norm(point_A - B, axis=1)
        closest_indices = np.argsort(distances)[:k]
        extracted_indices.append(closest_indices)

    return extracted_indices

if __name__ == "__main__":
    file = ["/home/iwata/wrapping_ws/src/wrapping/wrapping/data/box/merge.pcd", "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/final_02/merge.pcd"]
    pcds = load_pcds_path(file)
    for f in pcds:
        f.remove_non_finite_points()
    pcds_cp = []
    for i, f in enumerate(pcds):
        pcd_cp = copy.deepcopy(f)
        pcds_cp.append(pcd_cp)
    pcds = load_pcds(pcds_cp)

    size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30
    pcd_aligned = align_pcds(pcds, size)
    pcd_aligned[0].paint_uniform_color(np.array([0/255, 206/255, 209/255]))
    add_color_normal(pcd_aligned[1])
    o3d.visualization.draw_geometries([pcd_aligned[0], pcd_aligned[1]])

    np_pcd_0 = np.asarray(pcd_aligned[0].points)
    np_pcd_1 = np.asarray(pcd_aligned[1].points)

    # down sampling
    down_sampled_pcd_0 = pcd_aligned[0].uniform_down_sample(every_k_points = 100)
    down_sampled_pcd_1 = pcd_aligned[1].uniform_down_sample(every_k_points = 100)
    down_sampled_pcd_np_0 = np.asarray(down_sampled_pcd_0.points)
    down_sampled_pcd_np_1 = np.asarray(down_sampled_pcd_1.points)
    # estimate normals
    down_sampled_pcd_0.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    down_sampled_pcd_1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([down_sampled_pcd_0], point_show_normal=True)

    # # estimate normals
    # pcd_aligned[0].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # pcd_aligned[1].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # o3d.visualization.draw_geometries([pcd_aligned[0]], point_show_normal=True)

    extracted_indices = np.asarray(extract_closest_indices(down_sampled_pcd_np_1, down_sampled_pcd_np_0)).reshape([-1])
    extracted_normals = np.asarray(down_sampled_pcd_0.normals)[extracted_indices]
    extracted_normals_average = [np.mean(extracted_normals[i:i+10], axis=0) for i in range(0, len(extracted_normals), 10)]
    normalized_extracted_normals_average = extracted_normals_average / np.linalg.norm(extracted_normals_average, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.abs(np.sum(normalized_extracted_normals_average * np.asarray(down_sampled_pcd_1.normals), axis=1))))
    condition = (angles >= 10) & (angles <= 75)
    r = np.sum(condition) / angles.shape[0]
    print(r)
