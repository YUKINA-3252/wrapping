#!/usr/bin/env python

import rospy
from jsk_recognition_msgs.msg import PaperCorner
from wrapping_melodic.msg import PointCloudAverageService
from wrapping_melodic.srv import PointCloudAverage
from sensor_msgs.msg import Image
import cv_bridge
import message_filters
import numpy as np
import std_msgs.msg

shifted_line_points_int_multi_array = None
shift_width = 5
pointcloud_z_average = None

def pointcloud_average_calculate(req):
    corner_x_topic = req.corner_x_topic
    corner_y_topic = req.corner_y_topic
    depth_image_topic = req.depth_image_topic

    while True:
        sub_corner_x = message_filters.Subscriber(corner_x_topic, PaperCorner, queue_size=1)
        sub_corner_y = message_filters.Subscriber(corner_y_topic, PaperCorner, queue_size=1)
        sub_depth_image = message_filters.Subscriber(depth_image_topic, Image)
        subs = [sub_corner_x, sub_corner_y, sub_depth_image]
        sync = message_filters.ApproximateTimeSynchronizer(subs, queue_size=10, slop=0.1)
        sync.registerCallback(callback)

        if pointcloud_z_average is not None and shifted_line_points_int_multi_array is not None:
            break

    result = PointCloudAverageService()
    result.average = pointcloud_z_average
    result.shifted_line_points_int_multi_array = shifted_line_points_int_multi_array.data

    return result


def bressham_line(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return points


def callback(corner_x_msg, corner_y_msg, depth_msg):
    bridge = cv_bridge.CvBridge()
    depth_image = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

    if depth_msg.encoding == '16U1':
        depth_image = np.asarray(depth_image, dtype=np.float32)
        depth_image /= 1000.0
    elif depth_msg.encoding != '32FC1':
        rospy.logerr('Unsupported depth encoding: %s' %
                     depth_msg.encoding)

    height, width = depth_image.shape

    global shifted_line_points_int_multi_array
    global pointcloud_z_average
    corner_y_msg_idx_list = list(enumerate(corner_y_msg.corner))
    corner_y_msg_min_idx, corner_y_msg_min_value = min(corner_y_msg_idx_list, key=lambda x: x[1])
    corner_y_msg_idx_list.remove((corner_y_msg_min_idx, corner_y_msg_min_value))
    corner_y_msg_2nd_min_idx, corner_y_msg_2nd_min_value = min(corner_y_msg_idx_list, key=lambda x: x[1])

    # get the pixel sequence of the line segment connecting two points
    line_points = bressham_line(corner_y_msg.corner[corner_y_msg_min_idx], corner_x_msg.corner[corner_y_msg_min_idx],
                                corner_y_msg.corner[corner_y_msg_2nd_min_idx], corner_x_msg.corner[corner_y_msg_2nd_min_idx])
    shifted_line_points = []
    for x, y in line_points:
        for dx in range(-shift_width, shift_width + 1):
            shifted_line_points.append((x + dx, y))
    shifted_line_points_int_multi_array = std_msgs.msg.Int32MultiArray(data=[item for sublist in shifted_line_points for item in sublist])
    pointcloud_z_average = np.mean([depth_image.reshape(-1)[a * width + b] for a, b in line_points])


def pointcloud_average_service_server():
    rospy.init_node('pointcloud_average_service_server')

    rospy.Service('pointcloud_average_service', PointCloudAverage, pointcloud_average_calculate)

    rospy.spin()


if __name__ == "__main__":
    pointcloud_average_service_server()
