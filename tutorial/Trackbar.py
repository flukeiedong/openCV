import cv2
import numpy as np

class Trackbar:

    def __init__(self, image):

        self.image = image

        def empty(a):
            pass

        cv2.namedWindow("TrackBars")
        cv2.resizeWindow("TrackBars", 640, 240)
        cv2.createTrackbar("Hue min", "TrackBars", 0, 179, empty)
        cv2.createTrackbar("Hue max", "TrackBars", 179, 179, empty)
        cv2.createTrackbar("Sat min", "TrackBars", 0, 255, empty)
        cv2.createTrackbar("Sat max", "TrackBars", 255, 255, empty)
        cv2.createTrackbar("Val min", "TrackBars", 0, 255, empty)
        cv2.createTrackbar("Val max", "TrackBars", 255, 255, empty)
        cv2.waitKey(1)

    def updateImage(self, image):
        self.image = image

    def getTrackbarValues(self):
        h_min = cv2.getTrackbarPos("Hue min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val max", "TrackBars")
        print(h_min, h_max, s_min, s_max, v_min, v_max)
        return h_min, h_max, s_min, s_max, v_min, v_max

    def mask(self, imageHSV):
        # h_min, h_max, s_min, s_max, v_min, v_max = self.getTrackbarValues()
        h_min, h_max, s_min, s_max, v_min, v_max = 95, 179, 144, 255, 40, 255
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imageHSV, lower, upper)
        maskResult = cv2.bitwise_and(self.image, self.image, mask=mask)
        return mask, maskResult

