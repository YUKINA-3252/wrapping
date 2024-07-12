#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import pcl
import pcl.pcl_visualization

def pcl_callback(msg):
    pc = pcl.PointCloud()
    pc.from_message(msg)

    points_np = pc.to_array()
    average_x = points_np[:, 0].mean()

    rospy.loginfo(average_x)
    rospy.signal_shutdown("shut down")


def pcl_listener():
    rospy.init_node("pcl_listener", anonymous=True)
    rospy.Subscriber("/bbox_cloud_extract_detect_tape_stand_output", PointCloud2, pcl_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        pcl_listener()
    except rospy.ROSInterruptException:
        pass
    # rospy.init_node("tf_transform")
    # average_coords = [-0.26081263, -0.03317862,  0.77987728]
    # # translate pointcloud coords
    # listener = tf.TransformListener()
    # target_frame = "/tape_stand"
    # listener.waitForTransform("/head_camera_rgb_optical_frame", target_frame, rospy.Time(), rospy.Duration(1.0))
    # (trans, rot) = listener.lookupTransform("/head_camera_rgb_optical_frame", target_frame, rospy.Time(0))
    # average_coords_trans = listener.transformPoint(target_frame, average_coords)
    # print(average_coords_trans)
