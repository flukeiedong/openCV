import cv2

PATH = "/Users/los_napath/PycharmProjects/openCV/src/people2.jpeg"

faceCascade = cv2.CascadeClassifier("/Users/los_napath/PycharmProjects/openCV/src/haarcascade_frontalface_default.xml")

# img = cv2.imread(PATH)
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
# print(faces)
#
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
#
# cv2.imshow("Face Detection", img)
# cv2.waitKey(0)

webcam = cv2.VideoCapture(0)

while True:
    success, img = webcam.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.imshow("Face Detection", img)
    cv2.waitKey(1)

