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
    box_directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/box/merge.pcd"
    source = o3d.io.read_point_cloud(box_directory_path)
    source.remove_non_finite_points()
    source_np = np.asarray(source.points)
    add_color_normal(source, "box")
    o3d.visualization.draw_geometries([source])
