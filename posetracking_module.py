import cv2
import mediapipe as mp
import time

def trackpose(pose_id=0, display_fps=True, p_time=0):
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
                if id==pose_id:
                    cv2.circle(
                        img, (x,y),
                        8, (255,0,255),
                        3, cv2.FILLED
                    )
            pose_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        if display_fps:
            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time

            cv2.putText(
                img, str(int(fps)),
                (10,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,255), 3
            )
    
        cv2.imshow("Image", img)
        cv2.waitKey(1)
