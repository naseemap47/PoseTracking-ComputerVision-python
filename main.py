from unittest import result
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

pose_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        for id, lm in enumerate(result.pose_landmarks.landmark):
            height, width, channel = img.shape
            x, y = int(lm.x*width), int(lm.y*height)
            if id==0:
                cv2.circle(
                    img, (x,y),
                    8, (255,0,255),
                    3, cv2.FILLED
                )
        pose_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Image", img)
    cv2.waitKey(1)