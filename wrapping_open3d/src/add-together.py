import numpy as np
import open3d as o3d

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

def merge(pcds):
    all_points=[]
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))

    return merged_pcd


path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/202309231424"
pcd1 = o3d.io.read_point_cloud("{}/side/merge.pcd".format(path))
pcd2 = o3d.io.read_point_cloud("{}/top/merge.pcd".format(path))
add_color_normal(pcd1)
add_color_normal(pcd2)
size = np.abs((pcd1.get_max_bound() - pcd1.get_min_bound())).max() / 30
pcd_aligned = align_pcds([pcd1, pcd2], size)
pcd_merge = merge(pcd_aligned)
add_color_normal(pcd_merge)
o3d.visualization.draw_geometries([pcd_merge])
o3d.io.write_point_cloud("{}/merge.pcd".format(path), pcd_merge)
