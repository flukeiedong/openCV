import cv2

webcam = cv2.VideoCapture(0)
webcam.set(3, 480)
webcam.set(4, 640)
webcam.set(10, 100)

while True:
    success, img = webcam.read()
    img = cv2.flip(img, 1)
    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
