#!/usr/bin/env python

import rospy
from wrapping_melodic.srv import PointCloudAverage

def pointcloud_average_service_client(depth_image, corner_x, corner_y):
    rospy.wait_for_service('pointcloud_average_service')

    try:
        pointcloud_average_service = rospy.ServiceProxy('pointcloud_average_service', PointCloudAverage)
        respl = pointcloud_average_service(depth_image, corner_x, corner_y)
        return respl.average
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


if __name__ == "__main__":
    if len(sys.
