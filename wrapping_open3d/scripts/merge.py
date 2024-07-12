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


def atoi (text):
    return int(text) if text.isdigit() else text
def natural_keys (text):
    return [atoi(c) for c in re.split(r'(\d+)',text)]


if __name__ == "__main__":
    directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping/data/cylinder01"
    file = sorted(glob.glob("{}/*.ply".format(directory_path)), key=natural_keys)

    pcd1 = o3d.io.read_point_cloud(file[0])
    pcd1.remove_non_finite_points()
    add_color_normal(pcd1)
    for i in range(len(file) - 1):
        pcd2 = o3d.io.read_point_cloud(file[i+1])
        pcd2.remove_non_finite_points()
        add_color_normal(pcd2)
        pcd1 += pcd2

    o3d.visualization.draw_geometries([pcd1])
    o3d.io.write_point_cloud("{}/merge.pcd".format(directory_path), pcd1)
