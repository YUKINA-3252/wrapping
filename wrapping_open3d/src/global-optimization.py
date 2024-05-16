import copy
import glob
import sys
import numpy as np
import open3d as o3d
import os
import re

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

def merge(pcds):
    all_points=[]
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))

    return merged_pcd

def add_color_normal (pcd):
    pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(kdt_n)

def load_pcds_path(pcd_files):
    pcds = []
    for f in pcd_files:
        pcd = o3d.io.read_point_cloud(f)
        add_color_normal(pcd)
        pcds.append(pcd)

    return pcds

def load_pcds(pcd_files):
    pcds = []
    for i in range(len(pcd_files)):
        pcd = pcd_files[i]
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

def quaternion_to_rotationMatrix(quaternion, translationMatrix):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    d0 = translationMatrix[0]
    d1 = translationMatrix[1]
    d2 = translationMatrix[2]
    rotationMatrix = np.asarray([[1-2*q1*q1-2*q2*q2, 2*q0*q1+2*q3*q2, 2*q0*q2-2*q3*q1, d0],
                                 [2*q0*q1-2*q3*q2, 1-2*q0*q0-2*q2*q2, 2*q1*q2+2*q3*q0, d1],
                                 [2*q0*q2+2*q3*q1, 2*q1*q2-2*q3*q0, 1-2*q0*q0-2*q1*q1, d2],
                                 [0.0, 0.0, 0.0, 1.0]])
    return rotationMatrix

def store_angle (file_path, line_num):
    with open(file_path, "r") as file:
        lines = file.readlines()
        angle = float(lines[line_num].split()[0])
    return angle
def store_trans (file_path, line_num):
    with open(file_path, "r") as file:
        lines = file.readlines()

        line = lines[line_num+1].split()
        pos = np.asarray([float(value) for value in line]) * 0.001
        line = lines[line_num+2].split()
        R = np.asarray([float(value) for value in line])
        R = np.reshape(R, (3, 3))
        trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3] = pos
        trans = trans @ quaternion_to_rotationMatrix(np.array([0.5, -0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]))

n    return trans

def atoi (text):
    return int(text) if text.isdigit() else text
def natural_keys (text):
    return [atoi(c) for c in re.split(r'(\d+)',text)]

directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/202309231424/top"
file = sorted(glob.glob("{}/*.pcd".format(directory_path)), key=natural_keys)
pcds = load_pcds_path(file)
for f in pcds:
    f.remove_non_finite_points()

angle_list = []
for i in range(len(pcds)):
    angle_list.append(store_angle("{}/end_coords.txt".format(directory_path), i*3))
trans_list = []
for i in range(len(pcds)):
    trans_list.append(store_trans("{}/end_coords.txt".format(directory_path), i*3))

pcds_cp = []
for i, f in enumerate(pcds):
    pcd_cp = copy.deepcopy(f)
    pcd_cp.transform(trans_list[i])
    pcds_cp.append(pcd_cp)

pcds = load_pcds(pcds_cp)
o3d.visualization.draw_geometries(pcds, "input pcds")

size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30
pcds_0_to_60 = pcds[6:0:-1]
pcds_0_to_minus60 = pcds[6:-1]
pcd_aligned_0_to_60 = align_pcds(pcds_0_to_60, size)
pcd_aligned_0_to_minus60 = align_pcds(pcds_0_to_minus60, size)
# o3d.visualization.draw_geometries(pcd_aligned_0_to_minus60, "aligned")
pcd_merge = merge(pcd_aligned_0_to_60 + pcd_aligned_0_to_minus60)
o3d.io.write_point_cloud("{}/merge.pcd".format(directory_path), pcd_merge)
add_color_normal(pcd_merge)
o3d.visualization.draw_geometries([pcd_merge], "merged")
