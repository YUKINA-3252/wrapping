#!/usr/bin/env python

import rospy
from wrapping_melodic.srv import PointCloudAverageSecond
from sensor_msgs.msg import Image
import cv_bridge
import message_filters
import numpy as np
import std_msgs.msg

shifted_line_points_int_multi_array = None
pointcloud_z_average = None

def pointcloud_average_calculate(req):
    global shifted_line_points_int_multi_array
    global pointcloud_z_average
    shifted_line_points_int_multi_array = req.shifted_line_points_int_multi_array
    depth_image_topic = req.depth_image_topic

    while True:
        rospy.Subscriber(depth_image_topic, Image, depth_image_callback)
        if pointcloud_z_average is not None:
            break

    return pointcloud_z_average


def depth_image_callback(msg):
    bridge = cv_bridge.CvBridge()
    depth_image = bridge.imgmsg_to_cv2(msg, 'passthrough')

    if msg.encoding == '16U1':
        depth_image = np.asarray(depth_image, dtype=np.float32)
        depth_image /= 1000.0
    elif msg.encoding != '32FC1':
        rospy.logerr('Unsupported depth encoding: %s' %
                     msg.encoding)

    height, width = depth_image.shape

    global shifted_line_points_int_multi_array
    global pointcloud_z_average
    pointcloud_z_average = np.nanmean([depth_image.reshape(-1)[a * width + b] for a, b in zip(shifted_line_points_int_multi_array[::2], shifted_line_points_int_multi_array[1::2])])


def pointcloud_average_service_server():
    rospy.init_node('pointcloud_average_service_server_second')

    rospy.Service('pointcloud_average_service_second', PointCloudAverageSecond, pointcloud_average_calculate)

    rospy.spin()


if __name__ == "__main__":
    pointcloud_average_service_server()
