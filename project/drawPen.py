import cv2
import numpy as np
from tutorial import imagePreview
from tutorial.Trackbar import Trackbar

points = []

def paint(image, x, y):
    global points
    points.append([x, y])
    for p in points:

        if p[0] < 50 and p[1] < 50:
            points = []
            break

        cv2.circle(image, (p[0], p[1]), 20, (255, 0, 0), cv2.FILLED)

def getContours(image, imageContour = np.zeros((512, 512, 3), np.uint8)):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        # print("Area", area)

        if area > 500:
            cv2.drawContours(imageContour, contour, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(contour, True)
            # print("Perimeter", peri)
            approx = cv2.approxPolyDP(contour, 0.02*peri, True)
            print("Approx", len(approx))
            # print(approx)
            corners = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            paint(imageContour, x+w//2, y)
            print(x, y, w, h)
            cv2.rectangle(imageContour, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return imageContour

webcam = cv2.VideoCapture(0)
success, img = webcam.read()
trackbar = Trackbar(img)

blueMask = [95, 179, 144, 255, 40, 255]

while True:
    success, img = webcam.read()
    img = cv2.flip(img, 1)
    cv2.circle(img, (50, 50), 50, (255, 0, 255), cv2.FILLED)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    trackbar.updateImage(img)
    mask, maskResult = trackbar.mask(imgHSV)
    imgContour = getContours(mask, img)
    imgContour = imagePreview.resizeWithRatio(imgContour, 600)
    # cv2.imshow("Contour", imgContour)

    imgPreview = imagePreview.imagePreview(0.4, [[img, imgHSV], [mask, maskResult]])
    cv2.imshow("Preview", imgPreview)
    cv2.waitKey(1)
