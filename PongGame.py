import cv2
import mediapipe as mp
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

img_board = cv2.imread('ImgPongGame/board.png')
img_ball = cv2.imread('ImgPongGame/ball.png', cv2.IMREAD_UNCHANGED)
img_slider1 = cv2.imread('ImgPongGame/slider1.png', cv2.IMREAD_UNCHANGED)
img_slider2 = cv2.imread('ImgPongGame/slider2.png', cv2.IMREAD_UNCHANGED)
img_game_over = cv2.imread('ImgPongGame/game_over_board.png')

if img_ball.shape[2] != 4:
    img_ball = cv2.cvtColor(img_ball, cv2.COLOR_BGR2BGRA)
    img_ball[:, :, 3] = 255

if img_slider1.shape[2] != 4:
    img_slider1 = cv2.cvtColor(img_slider1, cv2.COLOR_BGR2BGRA)
    img_slider1[:, :, 3] = 255

if img_slider2.shape[2] != 4:
    img_slider2 = cv2.cvtColor(img_slider2, cv2.COLOR_BGR2BGRA)
    img_slider2[:, :, 3] = 255

alpha = 0.5
beta = 0.5
gamma = 0.0

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

detector = HandDetector(detectionCon=0.8, maxHands=2)

ball_position = [100, 100]
speedX = 10
speedY = 10

game_over = False
score = [0, 0]

while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)
    img = cv2.resize(img, (1024, 768))

    hands, img = detector.findHands(img, flipType=False)

    if img.shape == img_board.shape:
        img = cv2.addWeighted(img, alpha, img_board, beta, gamma)
    else:
        resized_game_over = cv2.resize(img_board, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(img, alpha, resized_game_over, beta, gamma)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = img_slider1.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1, 25, 405)

            if hand['type'] == 'Left':
                img = cvzone.overlayPNG(img, img_slider1, (24, y1))
                if 24 < ball_position[0] < 24 + w1 and y1 < ball_position[1] < y1 + h1:
                    speedX = -speedX
                    ball_position[0] += 30
                    score[0] += 1

            if hand['type'] == 'Right':
                img = cvzone.overlayPNG(img, img_slider2, (947, y1))
                if 885 < ball_position[0] < 885 + w1 and y1 < ball_position[1] < y1 + h1:
                    speedX = -speedX
                    ball_position[0] -= 30
                    score[1] += 1

    if ball_position[0] < 10 or ball_position[0] > 950:
        game_over = True

    if game_over:
        img = img_game_over
        cv2.putText(img, str(score[0]) + ':' + str(score[1]), (460, 400), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 5)
    else:
        if ball_position[1] >= 550 or ball_position[1] <= 20:
            speedY = -speedY

        ball_position[0] += speedX
        ball_position[1] += speedY

        img = cvzone.overlayPNG(img, img_ball, ball_position)

        cv2.putText(img, str(score[0]), (300, 720), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (700, 720), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    cv2.imshow('Image', img)

    key = cv2.waitKey(1)

    if key == ord('r'):
        ball_position = [100, 100]
        speedX = 10
        speedY = 10
        game_over = False
        score = [0, 0]
        img_game_over = cv2.imread('ImgPongGame/game_over_board.png')
