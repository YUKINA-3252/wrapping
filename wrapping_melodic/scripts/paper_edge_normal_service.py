#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
# from wrapping_melodic.srv import PaperEdgeNormalService
import numpy as np


# class ArraySaver:
#     def __init__(self):
#         rospy.init_node('array_saver_server')
#         self.data = None
#         self.subscriber = rospy.Subscriber('/edge_finder_paper/output/line', Float32MultiArray, self.callback)
#         self.srv = rospy.Service('paper_edge_normal_service', PaperEdgeNormalService, self.save_array)


#     def callback(self, msg):
#        self.data = msg.data


#     def save_array(self, req):
#         print("test")
#         # save pcd
#         # rospy.wait_for_service('/pointcloud_to_pcd_head/save_pcd')
#         if self.data is None:
#             return False, "No data received yet"
#         try:
#             save_pcd = rospy.ServiceProxy('/pointcloud_to_pcd_head/save_pcd', std_srvs/Empty)
#             np.davetxt('array_data.txt', np.array(self.data), fmt='%f')
#             return True, "Data saved successfully"
#         except Exception as e:
#             return False, "Error saving data: {}".format(str(e))


# if __name__ == "__main__":
#     array_saver = ArraySaver()
#     rospy.spin()


def callback(data):
    try:
        array_data = np.array(data.data)
        np.savetxt('/home/iwata/wrapping_ws/src/wrapping/wrapping_melodic/scripts/array_data.txt', array_data, fmt='%f')
        rospy.loginfo("Data daved")
        sub.unregister()
    except Exception as e:
        rospy.logerr("Error saving data: {}".format(str(e)))


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.spin()


sub = rospy.Subscriber("/edge_finder_paper/output/line", Float32MultiArray, callback)


if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
