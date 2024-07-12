#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


rospy.init_node('default_box_marker_array_publisher')
marker_publisher = rospy.Publisher('/default_box_marker_array', MarkerArray, queue_size=10)
while not rospy.is_shutdown():
    box_x_coord = rospy.get_param('~box_x_coord')
    box_y_coord = rospy.get_param('~box_y_coord')
    box_z_coord = rospy.get_param('~box_z_coord')
    marker_array = MarkerArray()
    box_len_x = rospy.get_param("~box_len_x")
    box_len_y = rospy.get_param("~box_len_y")
    box_len_z = rospy.get_param("~box_len_z")
    marker_array.markers = []
    marker = Marker()
    marker.header.frame_id = "WAIST"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "default_box_marker"
    marker.id = 1
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = box_x_coord
    marker.pose.position.y = box_y_coord
    marker.pose.position.z = box_z_coord - box_len_z / 2
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = box_len_x
    marker.scale.y = box_len_y
    marker.scale.z = box_len_z
    marker.color.r = 24 / 255.0
    marker.color.g = 235 / 255.0
    marker.color.b = 249 / 255.0
    marker.color.a = 1.0
    marker_array.markers.append(marker)

    marker_publisher.publish(marker_array)

    rospy.sleep(0.5)
