from tutorial.Trackbar import Trackbar
import cv2

PATH = "/Users/los_napath/PycharmProjects/openCV/src/peter.jpg"
img = cv2.imread(PATH)
trackbar = Trackbar(img)