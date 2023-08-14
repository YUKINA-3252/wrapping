(load "models/arrow-object.l")
(load "package://hrpsys_ros_bridge_tutorials/euslisp/hironxjsk-interface.l")
(load "package://wrapping_melodic/euslisp/detect.l")
(ros::load-ros-package "jsk_recognition_msgs")

(hironxjsk-init)
(setq *right-coords-arrow* (arrow))
(setq *left-coords-arrow* (arrow))
(send *right-coords-arrow* :newcoords (send *hironxjsk* :rarm :end-coords :copy-worldcoords))
(send *left-coords-arrow* :newcoords (send *hironxjsk* :larm :end-coords :copy-worldcoords))
(send (send *hironxjsk* :larm :end-coords) :assoc *left-coords-arrow*)
(send (send *hironxjsk* :rarm :end-coords) :assoc *right-coords-arrow*)
(setq object-list (list *hironxjsk* *right-coords-arrow* *left-coords-arrow*))
(objects object-list)


(defun ir2ri (&optional (time 3000))
  (send *ri* :angle-vector (send *hironxjsk* :angle-vector) time)
  (send *ri* :hand-angle-vector :rhand (send *hironxjsk* :hand-angle-vector :rhand))
  (send *ri* :hand-angle-vector :lhand (send *hironxjsk* :hand-angle-vector :lhand))
  (send *ri* :wait-interpolation))
(defun ir2ri-hand ()
    (send *ri* :hand-angle-vector :rhand (send *hironxjsk* :hand-angle-vector :rhand))
    (send *ri* :hand-angle-vector :lhand (send *hironxjsk* :hand-angle-vector :lhand))
    (send *ri* :wait-interpolation))

(defun open-rhand()
  (let ((hand-angle-vector (send *hironxjsk* :hand-angle-vector :rhand)))
    (setf (aref hand-angle-vector 0) 90)
    (setf (aref hand-angle-vector 1) 0)
    (setf (aref hand-angle-vector 2) -90)
    (setf (aref hand-angle-vector 3) 0)
    (send *hironxjsk* :hand-angle-vector :rhand hand-angle-vector)
    (ir2ri-hand)))
(defun open-lhand()
  (let ((hand-angle-vector (send *hironxjsk* :hand-angle-vector :lhand)))
    (setf (aref hand-angle-vector 0) 90)
    (setf (aref hand-angle-vector 1) 0)
    (setf (aref hand-angle-vector 2) -90)
    (setf (aref hand-angle-vector 3) 0)
    (send *hironxjsk* :hand-angle-vector :lhand hand-angle-vector)
    (ir2ri-hand)))

(defun reset-pose ()
  (send *hironxjsk* :reset-manip-pose)
  (send *hironxjsk* :larm :move-end-pos #f(0 0 200) :world)
  (send *hironxjsk* :rarm :move-end-pos #f(0 0 200) :world)
  (send *hironxjsk* :stop-grasp)
  (send *hironxjsk* :head :look-at
        (send (send (send *hironxjsk* :torso :end-coords :copy-worldcoords)
                    :translate #f(800 0 0)) :worldpos))
  (send *irtviewer* :draw-objects)
  (send *ri* :angle-vector (send *hironxjsk* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp)
  (send *ri* :wait-interpolation)
  (setq left-initial-coords (send *hironxjsk* :larm :end-coords :copy-worldcoords))
  (setq right-initial-coords (send *hironxjsk* :rarm :end-coords :copy-worldcoords))
  (setq initial-coords (send *hironxjsk* :angle-vector)))

(defun ri2ir ()
  (send *hironxjsk* :angle-vector (send *ri* :state :potentio-vector)))

(defun solve-ik (target-coords)
  (send *hironxjsk* :larm :inverse-kinematics (send target-coords :copy-worldcoords) :rotation-axis t))

(defun solve-ik-face (num)
  (setq *target-coords* (send (send (elt surface-location num) :copy-worldcoords) :translate (list2f (scale-vector (f2list (elt surface-normal-vector num)) 200)) :world))
  (if (or (eq num 2) (eq num 3))
      (send *target-coords* :rotate pi/2 :z))
  (send *target-coords* :rotate pi :y)
  (send (send *target-coords* :translate #f(0 0 100) :world) :rotate (deg2rad 30) :y)
  (send *hironxjsk* :larm :inverse-kinematics (send *target-coords* :copy-worldcoords) :rotation-axis t))

(reset-pose)
(open-lhand)
(detect)

(setq varlist (list (list 0 2 '+) (list 0 3 '-) (list 1 2 '-) (list 1 3 '+)))
(setq corner-coords (list))
(dotimes (x 4)
  (setq *target-coords* (send (send (elt surface-location (elt (elt varlist x) 0)) :copy-worldcoords) :translate
                              (list2f (scale-vector (f2list (elt surface-normal-vector (elt (elt varlist x) 1))) (/ (y-of-cube *target-bbox*) 2))) :world))
  (send *target-coords* :rotate (funcall (elt (elt varlist x) 2) pi (atan (/ (sqrt (+ (expt (x-of-cube *target-bbox*) 2) (expt (y-of-cube *target-bbox*) 2))) (y-of-cube *target-bbox*)))) :z)
  (send (send (send *target-coords* :rotate pi :x) :translate #f(-200 0 100) :local) :rotate (deg2rad 30) :y)
  (setq corner-coords (append corner-coords (list *target-coords*))))

(defun solve-ik-sphere (deg)
  (setq *target-coords* (send (send (send (send (send (make-coords) :locate (send (send *target-bbox* :copy-worldcoords) :pos) :world)
                                          :translate (float-vector (* -1 (* (cos (deg2rad deg)) (+ (x-of-cube *target-bbox*) 100))) (* (sin (deg2rad deg)) (+ (x-of-cube *target-bbox*) 100)) 0) :world) :rotate (* -1 (deg2rad deg)) :z)
                                    :translate #f(0 0 100) :world) :rotate (deg2rad 30) :y))
  (solve-ik *target-coords*)
  )


(setq xml-data "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<deg
</root>" deg (send *hironxjsk* :larm :end-cords :copy-worldcoords ))
(setq file-path "/p")
(with-open-file (out-stream file-path :direction :output
                                      :if-exists :supersede
                                      :if-does-not-exist :create)
                (format out-stream "~A" xml-data))

(defun write-xml-to-file-with-variables (sdeg file-path)
  (setq xml-data (format nil "<deg>~A</deg>" sdeg))

  (with-open-file (out-stream file-path :direction :output
                                      :if-exists :supersede
                                      :if-does-not-exist :create)
    (format out-stream "~A" xml-data)))

(setq sdeg 30)
(write-xml-to-file-with-variables sdeg "xml_file/end_coords.xml")

(setq *target-coords-arrow* (arrow))
(send *target-coords-arrow* :newcoords (send *target-coords* :copy-worldcoords))
(setq object-list (append object-list *target-coords-arrow*))
(objects object-list)
