#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


def find_rectangle_center(x1, y1, x2, y2, width, height):
    center_x = (x1 + x2 - (y2 - y1) / math.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2)) * width / height) / 2
    center_y = (y1 + y2 + (x2 - x1) / math.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2)) * width / height) / 2

    return center_x, center_y

def calc_rotate_angle(dy):
    if dy > 0:
        angle = (math.pi / 2 - math.acos(dy)) * -1
    else:
        angle = math.acos(dy) - math.pi / 2

    return angle


class MarkerArrayPublisher:
    def __init__(self):
        rospy.init_node('box_marker_array_publisher')
        self.box_z_coord = rospy.get_param('~box_z_coord')
        self.subscriber = rospy.Subscriber('/edge_finder_box/output/line', Float32MultiArray, self.coordinate_callback)
        self.marker_publisher = rospy.Publisher('/box_marker_array_publisher', MarkerArray, queue_size=10)
        self.marker_array = MarkerArray()
        self.box_len_x = rospy.get_param("~box_len_x")
        self.box_len_y = rospy.get_param("~box_len_y")
        self.box_len_z = rospy.get_param("~box_len_z")

    def coordinate_callback(self, msg):
        coordinates = msg.data

        self.marker_array.markers = []

        marker = Marker()
        marker.header.frame_id = "WAIST"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "box_marker"
        marker.id = 1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        # center_x, center_y = find_rectangle_center(coordinates[0], coordinates[1], coordinates[3], coordinates[4], 0.2, 0.14)
        rotate_angle = calc_rotate_angle(coordinates[4])
        if coordinates[4] > 0:
            marker.pose.position.x = coordinates[0] - self.box_len_y / 2 * math.sin(rotate_angle)
        else:
            marker.pose.position.x = coordinates[0] + self.box_len_y / 2 * math.sin(rotate_angle)
        marker.pose.position.y = coordinates[1] - self.box_len_y / 2 * math.cos(rotate_angle)
        marker.pose.position.z = self.box_z_coord - self.box_len_z / 2
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = math.sin(rotate_angle / 2)
        marker.pose.orientation.w = math.cos(rotate_angle / 2)
        marker.scale.x = self.box_len_x
        marker.scale.y = self.box_len_y
        marker.scale.z = self.box_len_z
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.3
        self.marker_array.markers.append(marker)

        self.marker_publisher.publish(self.marker_array)

    def run(self):
        rospy.spin()


if __name__ == ('__main__'):
    try:
        marker_array_publisher = MarkerArrayPublisher()
        marker_array_publisher.run()
    except rospy.ROSInterruptException:
        pass
