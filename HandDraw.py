import cv2
import mediapipe as mp
import time
import os
import HandTrakerOOP
import numpy as np

folder_path = 'Images'
lst = os.listdir(folder_path)

overlay = []

for image_path in lst:
    image = cv2.imread(f'{folder_path}/{image_path}')
    overlay.append(image)

header = overlay[0]

draw_colour = (255, 0, 255)
brush_thickness = 15
eraser_thickness = 100
xp, yp = 0, 0

image_canvas = np.zeros((720, 1024, 3), np.uint8)
cap = cv2.VideoCapture(0)
cap.set(3, 1024)
cap.set(7, 720)

detector = HandTrakerOOP.HandTracker(detection_confidence=0.85)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img = detector.find_hands(img)
    landmark_positions = detector.find_position(img, draw=False)

    if len(landmark_positions) != 0:
        x1, y1 = landmark_positions[8][1:]
        x2, y2 = landmark_positions[12][1:]

        fingers = detector.fingers_up()

        if fingers[1] and fingers[2]:
            """Selection Mode"""
            xp, yp = 0, 0

            if y1 < 126:
                if 66 < x1 < 179:
                    header = overlay[1]
                    draw_colour = (255, 0, 255)
                elif 325 < x1 < 440:
                    header = overlay[2]
                    draw_colour = (0, 255, 0)
                elif 586 < x1 < 697:
                    header = overlay[3]
                    draw_colour = (0, 69, 255)
                elif 845 < x1 < 958:
                    header = overlay[4]
                    draw_colour = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_colour, cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            """Drawing Mode"""
            cv2.circle(img, (x1, y1), 15, draw_colour, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if draw_colour == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_colour, eraser_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), draw_colour, eraser_thickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_colour, brush_thickness)
                cv2.line(image_canvas, (xp, yp), (x1, y1), draw_colour, brush_thickness)

            xp, yp = x1, y1

    # Преобразуем canvas в черно-белый формат и инвертируем его
    image_gray = cv2.cvtColor(image_canvas, cv2.COLOR_BGR2GRAY)
    _, image_inverse = cv2.threshold(image_gray, 50, 255, cv2.THRESH_BINARY_INV)
    image_inverse = cv2.cvtColor(image_inverse, cv2.COLOR_GRAY2BGR)

    # Убедитесь, что размеры совпадают для операции bitwise_and
    if img.shape != image_inverse.shape:
        image_inverse = cv2.resize(image_inverse, (img.shape[1], img.shape[0]))

    img = cv2.bitwise_and(img, image_inverse)

    # Убедитесь, что размеры совпадают для операции bitwise_or
    if img.shape != image_canvas.shape:
        image_canvas = cv2.resize(image_canvas, (img.shape[1], img.shape[0]))

    img = cv2.bitwise_or(img, image_canvas)

    img[0:126, 0:1024] = header
    # img = cv2.addWeighted(img, 0.5, image_canvas, 0.5, 0)
    cv2.imshow('Image', img)
    cv2.imshow('Canvas', image_canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
