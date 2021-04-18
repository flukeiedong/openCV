import cv2
import numpy as np

PATH = "/Users/los_napath/PycharmProjects/openCV/src/cards.jpg"

img = cv2.imread(PATH)

width, height = 250, 350 # the aspect ratio of a card
pt1 = np.float32([[304, 75], [454, 67], [311, 262], [483, 250]])
pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

matrix = cv2.getPerspectiveTransform(pt1, pt2)
imgResult = cv2.warpPerspective(img, matrix, (width, height))

print("Image size after warp", imgResult.shape)

cv2.imshow("Default", img)
cv2.imshow("Warp Perspective", imgResult)
cv2.imshow("Gray Result", cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY))
cv2.waitKey(0)

