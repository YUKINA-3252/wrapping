import copy
import numpy as np
import open3d as o3d
import re

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

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

def calc_cog(pcd):
    pcd_np = np.asarray(pcd.points)
    cog = np.mean(pcd_np, axis=0)
    cog.reshape((1, 3))
    cog = np.append(cog, np.array([0.0]), axis=0)
    return cog

if __name__ == "__main__":

    pcd1 = o3d.io.read_point_cloud("save_pcd/sample_pcd_1691894406531966.pcd")
    pcd2 = o3d.io.read_point_cloud("save_pcd/sample_pcd_1691905282011343.pcd")
    pcd3 = o3d.io.read_point_cloud("save_pcd/sample_pcd_1691905369039818.pcd")
    pcd1.remove_non_finite_points()
    pcd2.remove_non_finite_points()
    pcd3.remove_non_finite_points()
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])
    p1 = np.asarray([315.989, 67.4202, 76.0648]) * 0.001
    R1 = np.asarray([[0.866056, 0.000245, 0.499947],
                     [-0.000128, 1.0, -0.000268],
                     [-0.499947, 0.000168, 0.866056]])
    trans1 = np.eye(4)
    trans1[:3, :3] = R1
    trans1[:3, 3] = p1
    trans1 = trans1 @ quaternion_to_rotationMatrix(np.array([0.5, -0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]))
    p2 = np.asarray([359.518, 229.482, 75.9747]) * 0.001
    R2 = np.asarray([[0.75, 0.5, 0.43],
                     [-0.43, 0.866, -0.25],
                     [-0.50, -7.866e-08, 0.866]])
    trans2 = np.eye(4)
    trans2[:3, :3] = R2
    trans2[:3, 3] = p2
    trans2 = trans2 @ quaternion_to_rotationMatrix(np.array([0.5, -0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]))
    p3 = np.asarray([477.694, 347.966, 76.121]) * 0.001
    R3 = np.asarray([[0.433, 0.866, 0.25],
                     [-0.75, 0.50, -0.43],
                     [-0.50, 3.2e-05, 0.866]])
    trans3 = np.eye(4)
    trans3[:3, :3] = R3
    trans3[:3, 3] = p3
    trans3 = trans3 @ quaternion_to_rotationMatrix(np.array([0.5, -0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.0]))
    pcd1_cp = copy.deepcopy(pcd1)
    pcd2_cp = copy.deepcopy(pcd2)
    pcd3_cp = copy.deepcopy(pcd3)
    pcd1_cp.transform(trans1)
    pcd2_cp.transform(trans2)
    pcd3_cp.transform(trans3)
    o3d.visualization.draw_geometries([pcd1_cp, pcd2_cp, pcd3_cp])
    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([target_cp, mesh])
