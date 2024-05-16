#!/usr/bin/env python

import copy
import numpy as np
import open3d as o3d
import os


if __name__ == "__main__":
    directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping_melodic/save_pcd/1707975742578560.pcd"
    pcd = o3d.io.read_point_cloud(directory_path)
    pcd.remove_non_finite_points()
    pcd_np = np.asarray(pcd.points)
    o3d.visualization.draw_geometries([pcd])

    kdtree = o3d.geometry.KDTreeFlann(pcd)
    radius = 0.1
    print(pcd_np[:, 2])
