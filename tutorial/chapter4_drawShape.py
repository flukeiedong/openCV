import cv2
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
print(img)
print("Image size", img.shape)

# Draw rectangle to the img
# img[200:300, 200:300] = 255, 0, 0

imgWidth = img.shape[1]
imgHeight = img.shape[0]

cv2.line(img, (0, 0), (200, 200), (0, 255, 255), 3)
cv2.line(img, (imgWidth, 0), (0, imgHeight), (0, 255, 0), 3)
cv2.rectangle(img, (200, 200), (400, 400), (255, 0, 255), 7)
cv2.circle(img, (250, 250), 50, (255, 0, 0), 5)
cv2.putText(img, "openCV", (300, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)


cv2.imshow("Image by numpy", img)
cv2.waitKey(0)
