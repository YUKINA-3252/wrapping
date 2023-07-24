#!/usr/bin/env python
import rospy
import tf_conversions
import tf2_ros
import tf
import numpy as np
import geometry_msgs.msg

if __name__ == "__main__":
    rospy.init_node("tf_handcamera")

    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    listener = tf.TransformListener()

    # tfBuffer = tf2_ros.Buffer()
    # listener = tf2_ros.TransformListener(tfBuffer)
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        try:
            (_, rotA) = listener.lookupTransform('LARM_JOINT5_Link','WAIST', rospy.Time(0))
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "LARM_JOINT5_Link"
            t.child_frame_id = "hand_cam"

            t.transform.translation.x = 0.0
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.0
            # (_, rotA) = tfBuffer.lookup_transform('LARM_JOINT5_Link', 'WAIST', rospy.Time(0))
            rotB = [-1.0, 0.0, 0.0, 0.0]
            rotA_np = np.array([[rotA[0], rotA[1], -1.0 * rotA[2], rotA[3]],
                                [rotA[1], rotA[0], rotA[3], -1.0 * rotA[2]],
                                [rotA[2], -1.0 * rotA[3], rotA[0], rotA[1]],
                                [rotA[3], rotA[2], -1.0 * rotA[1], rotA[0]]])
            rotB_np = np.array(rotB)
            rot = np.dot(rotA_np, rotB_np)
            t.transform.rotation.x = rotA[0]
            t.transform.rotation.y = -1.0 * rotA[1]
            t.transform.rotation.z = -1.0 * rotA[2]
            t.transform.rotation.w = -1.0 * rotA[3]

            br.sendTransform(t)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue
            rate.sleep()
