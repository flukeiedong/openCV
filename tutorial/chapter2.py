import cv2
import numpy as np

PATH = "/Users/los_napath/PycharmProjects/openCV/src/peter.jpg"
kernel = np.ones((5, 5), np.uint8)
print(kernel)

img = cv2.imread(PATH)

# print(img.shape)
# cv2.imshow("Default", img)
img = cv2.resize(img, (360, 360))
# print(img.shape)
# cv2.imshow("Resize", img)

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
imgCanny = cv2.Canny(img, 150, 200)
imgDilation = cv2.dilate(imgCanny, kernel, iterations=1)
imgErode = cv2.erode(imgDilation, kernel, iterations=1)

cv2.imshow("Default", img)
cv2.imshow("Gray", imgGray)
cv2.imshow("Blur", imgBlur)
cv2.imshow("Canny", imgCanny)
cv2.imshow("Dilation", imgDilation)
cv2.imshow("Erosion", imgErode)



cv2.waitKey(0)