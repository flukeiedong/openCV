import cv2

PATH = "/Users/los_napath/PycharmProjects/openCV/src/peter.jpg"

img = cv2.imread(PATH)

print("Image size", img.shape)

imgResize = cv2.resize(img, (360, 360))

print("Image after resize", imgResize.shape)

imgCrop = img[100:200, 400:600]

print("Image after crop", imgCrop.shape)

cv2.imshow("Default", img)
cv2.imshow("Resize", imgResize)
cv2.imshow("Crop", imgCrop)
cv2.waitKey(0)

