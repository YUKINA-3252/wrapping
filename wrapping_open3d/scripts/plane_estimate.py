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
    all_points = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))

    return merged_pcd


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
                                 [2*q0*q1-2*q3*q2, 1-2*q0*q0-2*q2*q2, 2*q1*q2+2*qpp3*q0, d1],
                                 [2*q0*q2+2*q3*q1, 2*q1*q2-2*q3*q0, 1-2*q0*q0-2*q1*q1, d2],
                                 [0.0, 0.0, 0.0, 1.0]])
    return rotationMatrix


def rodrigues (vector, axis, rad):
    rotation_matrix = np.array([[np.cos(rad)+axis[0]**2*(1-np.cos(rad)), axis[0]*axis[1]*(1-np.cos(rad))-axis[2]*np.sin(rad), axis[0]*axis[2]*(1-np.cos(rad))+axis[1]*np.sin(rad)],
                                [axis[0]*axis[1]*(1-np.cos(rad))+axis[2]*np.sin(rad), np.cos(rad)+axis[1]**2*(1-np.cos(rad)), axis[1]*axis[2]*(1-np.cos(rad))-axis[0]*np.sin(rad)],
                                [axis[2]*axis[0]*(1-np.cos(rad))-axis[1]*np.sin(rad), axis[1]*axis[2]*(1-np.cos(rad))+axis[0]*np.sin(rad), np.cos(rad)+axis[2]**2*(1-np.cos(rad))]])
    return rotation_matrix @ vector

def coord_transform (pcd, transformation_matrix):
    points = np.array(pcd.points)
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = (transformation_matrix @ points.T).T[:, :3]
    return points


def atoi (text):
    return int(text) if text.isdigit() else text
def natural_keys (text):
    return [atoi(c) for c in re.split(r'(\d+)',text)]


if __name__ == "__main__":
    directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/final_02"

    if "merge.pcd" in os.listdir(directory_path):
        pass
    else:
        file = sorted(glob.glob("{}/*.pcd".format(directory_path)), key=natural_keys)
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
        pcd_merge = merge(pcd_aligned)
        add_color_normal(pcd_merge)
        o3d.io.write_point_cloud("{}/merge.pcd".format(directory_path), pcd_merge)
    pcd_merge = o3d.io.read_point_cloud("{}/merge.pcd".format(directory_path))
    o3d.visualization.draw_geometries([pcd_merge], "merged")

    table_normal_vector_model = np.asarray([0.0, 0.0, 1.0])
    box_front_normal_vector_model = np.asarray([1.0, 0.0, 0.0])

    # # estimate plane
    # if "face.pcd" in os.listdir(directory_path):
    #     pass
    # else:
    #     inliers = []
    #     plane_model_table, inliers_table = pcd_merge.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    #     inlier_cloud_table = pcd_merge.select_by_index(inliers_table)
    #     inlier_cloud_table.paint_uniform_color(np.random.rand(3))
    #     inliers.append(inlier_cloud_table)
    #     outlier_cloud_table = pcd_merge.select_by_index(inliers_table, invert = True)
    #     [a, b, c, d] = plane_model_table

    #     outlier = outlier_cloud_table
    #     o3d.visualization.draw_geometries(inliers + [outlier])
    #     while (True):
    #         plane_model, inlier = outlier.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
    #         inlier_cloud = outlier.select_by_index(inlier)
    #         inlier_cloud.paint_uniform_color(np.random.rand(3))
    #         inliers.append(inlier_cloud)
    #         outlier = outlier.select_by_index(inlier, invert = True)
    #         face_normal_vector = normalize_vector(np.asarray([plane_model[0], plane_model[1], plane_model[2]]))
    #         if (np.rad2deg(math.acos(np.dot(face_normal_vector_model, face_normal_vector))) < 15 or np.rad2deg(math.acos(np.dot(face_normal_vector_model * -1, face_normal_vector))) < 15):
    #             break
    #     # o3d.visualization.draw_geometries(inliers + [outlier])
    #     # o3d.visualization.draw_geometries([inliers[-1]])
    # # face = o3d.io.read_point_cloud("{}/face.pcd".format(directory_path))

    table_cloud = []
    box_front_cloud = []
    rest_cloud = []
    table_normal_vector = []
    pcd = pcd_merge
    while (True):
        plane_model, inlier = pcd.segment_plane(distance_threshold=0.003, ransac_n=3, num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inlier)
        face_normal_vector = normalize_vector(np.asarray([plane_model[0], plane_model[1], plane_model[2]]))
        # determine table
        if (np.rad2deg(math.acos(np.dot(table_normal_vector_model, face_normal_vector))) < 15 or np.rad2deg(math.acos(np.dot(table_normal_vector_model * -1, face_normal_vector))) < 15):
            inlier_cloud.paint_uniform_color(np.random.rand(3))
            table_cloud.append(inlier_cloud)
            table_normal_vector.append(plane_model)
        #determine box front face
        elif (np.rad2deg(math.acos(np.dot(box_front_normal_vector_model, face_normal_vector))) < 15 or np.rad2deg(math.acos(np.dot(box_front_normal_vector_model * -1, face_normal_vector))) < 15):
            inlier_cloud.paint_uniform_color(np.random.rand(3))
            box_front_cloud.append(inlier_cloud)
        else:
            inlier_cloud.paint_uniform_color(np.random.rand(3))
            rest_cloud.append(inlier_cloud)
        pcd = pcd.select_by_index(inlier, invert=True)
        if (len(table_cloud) == 2 and len(box_front_cloud) == 1):
            rest_cloud.append(pcd)
            break

    # determine which table cloud
    min_index = None
    min_value = float('inf')
    for i, sublist in enumerate(table_normal_vector):
        ratio = sublist[3] / sublist[2]
        if ratio < min_value:
            min_value = ratio
            min_index = i
    rest_cloud.append(table_cloud[min_index])
    del table_cloud[min_index]
    # o3d.visualization.draw_geometries(table_cloud + box_front_cloud)
    rest_cloud_pcds = o3d.geometry.PointCloud()
    for pcd in rest_cloud:
        rest_cloud_pcds += pcd
    cl, ind = rest_cloud_pcds.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    rest_cloud_removal = rest_cloud_pcds.select_by_index(ind)
    rest_cloud_removal.paint_uniform_color(np.array([0/255, 206/255, 209/255]))
    o3d.visualization.draw_geometries([rest_cloud_removal])

    # # estimate normal vector
    rest_cloud_removal.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([rest_cloud_removal], point_show_normal=True)

    normals_np = np.asarray(rest_cloud_removal.normals)
    test_normals1 = np.array([0.0, 0.0, 1.0])
    test_normals2 = np.array([0.0, 1.0, 0.0])
    angle1 = np.degrees(np.arccos(np.abs(np.dot(normals_np, test_normals1) / np.linalg.norm(normals_np, axis=1))))
    angle2 = np.degrees(np.arccos(np.abs(np.dot(normals_np, test_normals2) / np.linalg.norm(normals_np, axis=1))))
    angle_max = np.minimum(angle1, angle2)
    print(normals_np)
    print(angle_max)
    angle_max_indices = np.where(angle_max > 25)[0]
    print(len(normals_np), len(angle_max_indices), len(angle_max_indices) / len(normals_np))


    # # extract point cloud
    # model = o3d.geometry.PointCloud()
    # length_x = 0.22
    # length_y = 0.15
    # length_z = 0.11
    # thickness = 0.003
    # points = np.asarray(face.points).shape[0]
    # face_pos = np.asarray([480, 0, -20]) * 0.001

    # X = ((length_x * length_z * thickness) / points) ** (1/3)
    # model_point = np.asarray([[face_pos[0], face_pos[1], face_pos[2]+X*(int)(-length_z/X/2), 1.0],
    #                           [face_pos[0], face_pos[1], face_pos[2]+X*(int)(length_z/X/2), 1.0]])
    # for i in range((int)(thickness / X)*2):
    #     for x in range((int)(length_x / 2 / X)):
    #         for y in range((int)(-length_z / X / 2), (int)(length_z / X / 2)):
    #             model.points.append([face_pos[0]+face_normal_vector_model[1]*X*x+face_normal_vector_model[0]*X*i, face_pos[1]-face_normal_vector_model[0]*X*x+face_normal_vector_model[1]*X*i, face_pos[2]+X*y])
    #             model.points.append([face_pos[0]-face_normal_vector_model[1]*0.005*x+face_normal_vector_model[0]*X*i, face_pos[1]+face_normal_vector_model[0]*X*x+face_normal_vector_model[1]*X*i, face_pos[2]+X*y])

    # model.paint_uniform_color(np.random.rand(3))
    # o3d.io.write_point_cloud("{}/model.pcd".format(directory_path), model)
    # o3d.visualization.draw_geometries([face, model])

    # add_color_normal(face)
# add_color_normal(model)
# size = np.abs((model.get_max_bound() - model.get_min_bound())).max() / 30
# (align, accum_pose_list) = align_pcds([face, model], size)
# o3d.visualization.draw_geometries(align, "aligned")
# o3d.io.write_point_cloud("{}/model_align.pcd".format(directory_path), model)
# model_align = o3d.io.read_point_cloud("{}/model_align.pcd".format(directory_path))
# model = o3d.io.read_point_cloud("{}/model.pcd".format(directory_path))

# face_normal_vector_real_pos = accum_pose_list[0] @ np.append(face_pos, 1.0)
# arrow = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=face_pos)
# arrow.transform(accum_pose_list[0])
# arrow_x_axis_rotation = arrow.get_rotation_matrix_from_axis_angle([1, 0, 0])
# arrow_y_axis_rotation = arrow.get_rotation_matrix_from_axis_angle([0, 1, 0])
# arrow_z_axis_rotation = arrow.get_rotation_matrix_from_axis_angle([0, 0, 1])
# arrow_x_axis_world = np.dot(arrow_x_axis_rotation, [1, 0, 0])
# arrow_y_axis_world = np.dot(arrow_y_axis_rotation, [0, 1, 0])
# arrow_z_axis_world = np.dot(arrow_z_axis_rotation, [0, 0, 1])
# arrow_pos = arrow.get_center()
# obb = o3d.geometry.OrientedBoundingBox(arrow_pos + accum_pose_list[0][:, 0][:3] * length_y / 2 + [-0.01, -0.01, -0.005],
#                                        accum_pose_list[0][:3, :3],
#                                        np.array([length_y + 0.01, length_x + 0.01, length_z + 0.01]),)
# o3d.visualization.draw_geometries([face, model_align, obb, arrow])

# obb_rotation = np.eye(4)
# obb_rotation[:3, :3] = np.vstack((arrow_x_axis_world, arrow_y_axis_world, arrow_z_axis_world))
# obb_rotation[:3, 3] = obb.get_center()[:3]
# obb_org_rotation = np.eye(4)
# obb_org = o3d.geometry.OrientedBoundingBox(np.array([0, 0, 0]),
#                                            np.eye(3),
#                                            np.array([length_y + 0.01, length_x + 0.01, length_z + 0.01]))
# obb_to_obb_org_transformation_matrix = obb_org_rotation @ np.linalg.inv(obb_rotation)
# pcd_merge_org = copy.deepcopy(pcd_merge)
# points_pcd_merge_org = np.array(pcd_merge_org.points)
# points_pcd_merge_org = np.hstack((points_pcd_merge_org, np.ones((points_pcd_merge_org.shape[0], 1))))
# points_pcd_merge_org = (obb_to_obb_org_transformation_matrix @ points_pcd_merge_org.T).T[:, :3]
# pcd_merge_org.points = o3d.utility.Vector3dVector(points_pcd_merge_org)

# min_array = np.asarray(obb_org.get_box_points()).min(axis=0)
# max_array = np.asarray(obb_org.get_box_points()).max(axis=0)
# x_condition = (points_pcd_merge_org[:, 0] >= min_array[0]) & (points_pcd_merge_org[:, 0] <= max_array[0])
# y_condition = (points_pcd_merge_org[:, 1] >= min_array[1]) & (points_pcd_merge_org[:, 1] <= max_array[1])
# z_condition = (points_pcd_merge_org[:, 2] >= min_array[2]) & (points_pcd_merge_org[:, 2] <= max_array[2])
# selected_row = x_condition & y_condition & z_condition
# points_pcd_merge_org_inside = points_pcd_merge_org[selected_row]
# points_pcd_merge_org_outside = points_pcd_merge_org[np.logical_not(selected_row)]
# points_pcd_merge_inside = np.copy(points_pcd_merge_org_inside)
# points_pcd_merge_inside = np.hstack((points_pcd_merge_inside, np.ones((points_pcd_merge_inside.shape[0], 1))))
# points_pcd_merge_inside = (np.linalg.inv(obb_to_obb_org_transformation_matrix) @ points_pcd_merge_inside.T).T[:, :3]
# pcd_merge_inside = o3d.geometry.PointCloud()
# pcd_merge_inside.points = o3d.utility.Vector3dVector(points_pcd_merge_inside)
# points_pcd_merge_outside = np.copy(points_pcd_merge_org_outside)
# points_pcd_merge_outside = np.hstack((points_pcd_merge_outside, np.ones((points_pcd_merge_outside.shape[0], 1))))
# points_pcd_merge_outside = (np.linalg.inv(obb_to_obb_org_transformation_matrix) @ points_pcd_merge_outside.T).T[:, :3]
# pcd_merge_outside = o3d.geometry.PointCloud()
# pcd_merge_outside.points = o3d.utility.Vector3dVector(points_pcd_merge_outside)
# add_color_normal(pcd_merge_inside)
# add_color_normal(pcd_merge_outside)
# o3d.visualization.draw_geometries([obb, pcd_merge_outside])

# inliers = []
# _, inliers_table = pcd_merge_outside.segment_plane(distance_threshold=0.007, ransac_n=3, num_iterations=1000)
# inlier_cloud_table = pcd_merge_outside.select_by_index(inliers_table)
# inlier_cloud_table.paint_uniform_color(np.random.rand(3))
# inliers.append(inlier_cloud_table)
# outlier_cloud_table = pcd_merge_outside.select_by_index(inliers_table, invert = True)
# o3d.visualization.draw_geometries([obb, outlier_cloud_table])
