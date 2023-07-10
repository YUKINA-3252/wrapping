import copy
import numpy as np
import open3d as o3d

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
    source = o3d.io.read_point_cloud("../save_pcd/sample_pcd_1683866827277444.pcd")
    target = o3d.io.read_point_cloud("../save_pcd/sample_pcd_1683866846091469.pcd")
    source.remove_non_finite_points()
    target.remove_non_finite_points()
    threshold = 0.02
    quaternion = np.array([-0.679, 0.685, -0.153, 0.213])
    translationMatrix = np.array([0.253, -0.004, 0.563])
    P_2 = quaternion_to_rotationMatrix(quaternion, translationMatrix)
    P_2 = np.array([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    cog = calc_cog(source)
    cog_t = P_2 @ cog
    P_1 = np.asarray([[1.0, 0.0, 0.0, -1.0 * cog_t[0]],
                    [0.0, 1.0, 0.0, -1.0 * cog_t[1]],
                    [0.0, 0.0, 1.0, -1.0 * cog_t[2]],
                    [0.0, 0.0, 0.0, 1.0]])
    print(P_1)
    R = np.asarray([[np.sqrt(2)/2.0, np.sqrt(2)/-2.0, 0.0, 0.0],
                    [np.sqrt(2)/2.0, np.sqrt(2)/2.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    trans_init = P_2 @ P_1 @ R @ np.linalg.inv(P_1) @ np.linalg.inv(P_2)
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is :")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)
