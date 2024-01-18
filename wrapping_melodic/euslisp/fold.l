(load "package://wrapping_melodic/euslisp/box-utils.l")
(load "models/arrow-object.l")
(load "package://hrpsys_ros_bridge_tutorials/euslisp/hironxjsk-interface.l")
(ros::load-ros-package "jsk_recognition_msgs")
(load "package://wrapping_melodic/euslisp/init.l")


(reset-pose)
(detect-pose)

(setq offset 100)


(detect-paper)


;; right side
(send *target-coords-right* :move-to (send (send (send (send *target-paper-arrow* :copy-worldcoords) :rotate pi/2 :z) :rotate -pi/2 :x) :translate (float-vector 0 (* (/ z-of-paper 2) -1) offset) :world) :world)
(solve-ik-rarm *target-coords-right*)

(send *target-coords-right* :rotate (deg2rad 30) :x)
(solve-ik-rarm *target-coords-right*)

(send *target-coords-right* :translate #f(0 30 0) :world)
(solve-ik-rarm *target-coords-right*)

(send *hironxjsk* :start-grasp :rarm)

(send *target-coords-right* :translate #f(0 100 100) :world)
(solve-ik-rarm *target-coords-right*)
