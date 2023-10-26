import cv2
import numpy as np

# Esc key
ESC_KEY = 0x1b
# s key
S_KEY = 0x73
# r key
R_KEY = 0x72
# Maximum number of feature points
MAX_FEATURE_NUM = 500
# Termination conditions for iterative algorithms
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# Interval (1000 / frame rate)
INTERVAL = 30
# data
VIDEO_DATA = 'Screencast 2023-07-05 12:45:54.mp4'

class Motion:
    def __init__(self):
        cv2.namedWindow("motion")
        cv2.setMouseCallback("motion", self.onMouse)
        self.video = cv2.VideoCapture(VIDEO_DATA)
        self.interval = INTERVAL
        self.frame = None
        self.gray_next = None
        self.gray_prev = None
        self.features = None
        self.status = None

    def run(self):

        # Processing of the first frame
        end_flag, self.frame = self.video.read()
        self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        while end_flag:
            self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # Calculate OpticalFlow when feature points are registered
            if self.features is not None:
                # Calculation of optical flow
                features_prev = self.features
                self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
                                                    self.gray_prev, \
                                                    self.gray_next, \
                                                    features_prev, \
                                                    None, \
                                                    winSize = (10, 10), \
                                                    maxLevel = 3, \
                                                    criteria = CRITERIA, \
                                                    flags = 0)

                # Leave only valid feature points
                self.refreshFeatures()

                # Drawing valid feature points on the frame
                if self.features is not None:
                    for feature in self.features:
                        cv2.circle(self.frame, (feature[0][0], feature[0][1]), 4, (15, 241, 255), -1, 8, 0)

            cv2.imshow("motion", self.frame)

            # Prepare for next loop processing
            self.gray_prev = self.gray_next
            end_flag, self.frame = self.video.read()
            if end_flag:
                self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            # interval
            key = cv2.waitKey(self.interval)
            # Press Esc key to exit
            if key == ESC_KEY:
                break
            # Pause by pressing s key
            elif key == S_KEY:
                self.interval = 0
            elif key == R_KEY:
                self.interval = INTERVAL


        # end processing
        cv2.destroyAllWindows()
        self.video.release()


    # Mouse click to specify a feature point
    # Delete an existing feature point if there is an existing feature point in the clicked neighborhood.
    # Add a new feature point if there is no existing feature point in the clicked neighborhood.
    def onMouse(self, event, x, y, flags, param):
        # Other than left click
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # First feature point added
        if self.features is None:
            self.addFeature(x, y)
            return

        # Search radius (pixels)
        radius = 5
        # Search for existing feature points in the neighborhood
        index = self.getFeatureIndex(x, y, radius)

        # Delete existing feature points because there are existing feature points in the clicked neighborhood.
        if index >= 0:
            self.features = np.delete(self.features, index, 0)
            self.status = np.delete(self.status, index, 0)

        # Add a new feature point because there is no existing feature point in the clicked neighborhood
        else:
            self.addFeature(x, y)

        return


    # Get one index of an existing feature point within the specified radius
    # If there is no feature point within the specified radius, respond index = -1
    def getFeatureIndex(self, x, y, radius):
        index = -1

        # Not a single feature point was registered.
        if self.features is None:
            return index

        max_r2 = radius ** 2
        index = 0
        for point in self.features:
            dx = x - point[0][0]
            dy = y - point[0][1]
            r2 = dx ** 2 + dy ** 2
            if r2 <= max_r2:
                # This feature point is within the specified radius
                return index
            else:
                # This feature point is outside the specified radius
                index += 1

        # All feature points are outside the specified radius
        return -1


    # Add a new feature point
    def addFeature(self, x, y):

        # Feature points not yet registered
        if self.features is None:
            # Create ndarray and register coordinates of feature points
            self.features = np.array([[[x, y]]], np.float32)
            self.status = np.array([1])
            # High-precision feature points
            cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)

        # Maximum number of registered feature points exceeded
        elif len(self.features) >= MAX_FEATURE_NUM:
            print("max feature num over: " + str(MAX_FEATURE_NUM))

        # Register additional feature points
        else:
            # Add coordinates of feature points to the end of existing ndarray
            self.features = np.append(self.features, [[[x, y]]], axis = 0).astype(np.float32)
            self.status = np.append(self.status, 1)
            # High-precision feature points
            cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)


    # Leave only valid feature points
    def refreshFeatures(self):
        # Feature points not yet registered
        if self.features is None:
            return

        # Check all status
        i = 0
        while i < len(self.features):

            # Unrecognizable as a feature point
            if self.status[i] == 0:
                # Remove from existing ndbarray
                self.features = np.delete(self.features, i, 0)
                self.status = np.delete(self.status, i, 0)
                i -= 1

            i += 1


if __name__ == '__main__':
    Motion().run()
