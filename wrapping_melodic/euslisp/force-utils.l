(defparameter *wrench* (make-hash-table))
(sethash :larm *wrench* nil)
(sethash :rarm *wrench* nil)


(defun ftsensor-cb (arm msg)
  (sethash arm *wrench* msg)
  )


(defun create-vector (axis sign)
  (case axis
    (:x (float-vector (if (equal sign "plus") 1 -1) 0 0))
    (:y (float-vector 0 (if (equal sign "plus") 1 -1) 0))
    (:z (float-vector 0 0 (if (equal sign "plus") 1 -1)))))


(defun get-force (arm axis &key (latest t))
  (if latest
      (progn
        (sethash arm *wrench* nil)
        (while (null (gethash arm *wrench*))
          (ros::spin-once)
          (unix::usleep 1000)
          )
        )
      )
  (send (gethash arm *wrench*) :wrench :force axis))


(defun force-move (arm axis sign)
  (let ((first-force (get-force arm axis)))
    (while t
      (setq cur-force (get-force arm axis))
      (print (abs (- first-force cur-force)))
      (if (> (abs (- first-force cur-force)) 1)
          (return))
      (send *hironxjsk* arm :move-end-pos (create-vector axis sign) :world)
      (send *ri* :angle-vector (send *hironxjsk* :angle-vector) 500)
      (send *ri* :wait-interpolation))))
