import cv2
import numpy as np
from imagePreview import imagePreview

PATH = "/Users/los_napath/PycharmProjects/openCV/src/shape2d_2.png"


def getContours(image, imageContour = np.zeros((512, 512, 3), np.uint8)):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        print("Area", area)

        if area > 500:
            cv2.drawContours(imageContour, contour, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(contour, True)
            print("Perimeter", peri)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            print("Approx", len(approx))
            print(approx)
            corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            print(x, y, w, h)
            cv2.rectangle(imageContour, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(imageContour, str(corners), (x+(w//2), y+(h//2)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return imageContour


img = cv2.imread(PATH)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
imgContour = getContours(imgCanny, img.copy())
cv2.imshow("Contour", imgContour)


imgPreview = imagePreview(0.5, [[img, imgGray], [imgBlur, imgCanny]])
cv2.imshow("Preview", imgPreview)
cv2.waitKey(0)
