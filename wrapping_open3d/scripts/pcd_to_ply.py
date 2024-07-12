import numpy as np
import open3d as o3d


def add_color_normal (pcd):
    # pcd.paint_uniform_color(np.random.rand(3))
    pcd.paint_uniform_color(np.array([255/255, 182/255, 193/255]))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals(kdt_n)


if __name__ == "__main__":
    directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/tmp/merge.pcd"
    pcd = o3d.io.read_point_cloud(directory_path)
    add_color_normal(pcd)
    o3d.io.write_point_cloud("merge_top.ply", pcd)
