import math
import numpy as np
import open3d as o3d
import os
import pyransac3d as pyrsc

def add_color_normal (pcd):
    pcd.paint_uniform_color(np.random.rand(3))
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

def normalize_vector (x):
    c = np.linalg.norm(x)
    return x / c

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

directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/202308211501"
pcd_merge = o3d.io.read_point_cloud("{}/merge.pcd".format(directory_path))
add_color_normal(pcd_merge)
pcds = [pcd_merge]
o3d.visualization.draw_geometries(pcds, "merged")

face_normal_vector_ans = np.asarray([1.0, 0.0, 0.0])

if "face.pcd" in os.listdir(directory_path):
    pass
else:
    inliers = []

    plane_model_table, inliers_table = pcd_merge.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    inlier_cloud_table = pcd_merge.select_by_index(inliers_table)
    inlier_cloud_table.paint_uniform_color(np.random.rand(3))
    inliers.append(inlier_cloud_table)
    outlier_cloud_table = pcd_merge.select_by_index(inliers_table, invert = True)

    outlier = outlier_cloud_table
    while (True):
        plane_model, inlier = outlier.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
        inlier_cloud = outlier.select_by_index(inlier)
        inlier_cloud.paint_uniform_color(np.random.rand(3))
        inliers.append(inlier_cloud)
        outlier = outlier.select_by_index(inlier, invert = True)
        face_normal_vector = normalize_vector(np.asarray([plane_model[0], plane_model[1], plane_model[2]]))
        if (np.rad2deg(math.acos(np.dot(face_normal_vector_ans, face_normal_vector))) < 15 or np.rad2deg(math.acos(np.dot(face_normal_vector_ans * -1, face_normal_vector))) < 15):
            break
    o3d.visualization.draw_geometries(inliers + [outlier])

    o3d.io.write_point_cloud("{}/face.pcd".format(directory_path), inliers[-1])

face = o3d.io.read_point_cloud("{}/face.pcd".format(directory_path))

model = o3d.geometry.PointCloud()
length_x = 0.22
length_z = 0.11
thickness = 0.003
points = np.asarray(face.points).shape[0]
face_pos = np.asarray([480, 0, -20]) * 0.001

X = ((length_x * length_z * thickness) / points) ** (1/3)
for i in range((int)(thickness / X)*2):
    for x in range((int)(length_x / 2 / X)):
        for y in range((int)(length_z / X)):
            model.points.append([face_pos[0]+face_normal_vector_ans[1]*X*x+face_normal_vector_ans[0]*X*i, face_pos[1]-face_normal_vector_ans[0]*X*x+face_normal_vector_ans[1]*X*i, face_pos[2]+X*y])
            model.points.append([face_pos[0]-face_normal_vector_ans[1]*0.005*x+face_normal_vector_ans[0]*X*i, face_pos[1]+face_normal_vector_ans[0]*X*x+face_normal_vector_ans[1]*X*i, face_pos[2]+X*y])

model.paint_uniform_color(np.random.rand(3))
o3d.visualization.draw_geometries([face, model])

add_color_normal(face)
add_color_normal(model)
size = np.abs((model.get_max_bound() - model.get_min_bound())).max() / 30
align = align_pcds([face, model], size)
o3d.visualization.draw_geometries(align, "aligned")

o3d.visualization.draw_geometries([face, model])
