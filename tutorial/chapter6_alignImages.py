import cv2
import numpy as np
from imagePreview import imagePreview

PATH = "/Users/los_napath/PycharmProjects/openCV/src/peter.jpg"

img = cv2.imread(PATH)
img = cv2.resize(img, (200, 200))

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hor = np.hstack((img, img))
ver = np.vstack((img, img))


# print(img.shape, imgGray.shape)
# print("GRAY", imgGray)
# print("IMAGE", img)
#
# cv2.imshow("Horizontal", hor)
# cv2.imshow("Vertical", ver)
# cv2.waitKey(0)


imgStack = imagePreview(2, [img, img, imgGray])
cv2.imshow("Preview", imgStack)
cv2.waitKey(0)
