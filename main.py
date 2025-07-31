import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)

cv2.createTrackbar("Hue MIN", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue MAX", "TrackBars", 19, 179, empty)
cv2.createTrackbar("Saturation MIN", "TrackBars", 110, 255, empty)
cv2.createTrackbar("Saturation MAX", "TrackBars", 240, 255, empty)
cv2.createTrackbar("Value MIN", "TrackBars", 153, 255, empty)
cv2.createTrackbar("Value MAX", "TrackBars", 255, 255, empty)

while True:
    _, img = cap.read()
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue MIN", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue MAX", "TrackBars")
    s_min = cv2.getTrackbarPos("Saturation MIN", "TrackBars")
    s_max = cv2.getTrackbarPos("Saturation MAX", "TrackBars")
    v_min = cv2.getTrackbarPos("Value MIN", "TrackBars")
    v_max = cv2.getTrackbarPos("Value MAX", "TrackBars")

    lower = np.array([h_min, s_min, v_min])  # оттенок; насыщенность; яркость
    upper = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(imgHSV, lower, upper)  # фильтрация в диапазоне (изображение; минимальная граница; максимальная)

    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
