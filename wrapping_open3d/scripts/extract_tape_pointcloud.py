#!/usr/bin/env python

import copy
import numpy as np
import open3d as o3d
import os
import rospy
from std_srvs.srv import Empty


def get_pcd_service():
    rospy.init_node("get_pcd_service_server")
    service_name = "/pointcloud_to_pcd_tape/save_pcd"
    rospy.wait_for_service(service_name)
    service_proxy = rospy.ServiceProxy(service_name, Empty)

    try:
        response = service_proxy()
        rospy.loginfo("Service call successful")
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" %e)


if __name__ == "__main__":
    directory_path = "/home/iwata/wrapping_ws/src/wrapping/wrapping_melodic/save_pcd/tape"
    # delete past pcd data
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            os.unlink(item_path)

    get_pcd_service()

    files = os.listdir(directory_path)
    print(files)
    file_path = os.path.join(directory_path, files[0])
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.remove_non_finite_points()
    # o3d.visualization.draw_geometries([pcd])

    # calculate average of pointcloud coordinates
    points_np = np.asarray(pcd.points)
    average_coords = np.mean(points_np, axis=0)
    print(average_coords)

    # # translate pointcloud coords
    # listener = tf2_ros.TransformListener()
    # target_frame = "/tape_stand"
    # listener.waitForTransform("/head_camera_rgb_optical_frame", target_frame, rospy.Time(), rospy.Duration(1.0))
    # (trans, rot) = listener.lookupTransform("/head_camera_rgb_optical_frame", target_frame, rospy.Time(0))
    # average_coords_trans = listener.trasnformPoint(target_frame, average_coords)
    # print(average_coords_trans)

    # # attention clipper bbox coords is (0.61, 0.26, 0)
    # tape_x = 0.61 + average_x
    # tape_y = 0.26 + average_y

    # print(tape_x, tape_y)
