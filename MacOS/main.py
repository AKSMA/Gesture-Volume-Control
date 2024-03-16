import numpy as np
import mediapipe as mp
import cv2
import math
import osascript

#Solution API Usage
draw = mp.solutions.drawing_utils
drawstyle = mp.solutions.drawing_styles
hand = mp.solutions.hands
hands = hand.Hands(model_complexity=0,min_detection_confidence=0.6,min_tracking_confidence=0.6)

#Webcam Setup
cam = cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)

#Mediapipe Landmark Model Using OpenCV
while True:
    ret,frame = cam.read()
    rgbframe = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    op = hands.process(rgbframe)
    if op.multi_hand_landmarks:
        for i in op.multi_hand_landmarks:
            print(i)
            draw.draw_landmarks(
                frame,
                i,
                hand.HAND_CONNECTIONS,
                landmark_drawing_spec=draw.DrawingSpec(color = (0,255,0),circle_radius = 1)
                )
    
        #Finding Position of Hand Ladmarks
        lmlist = []
        for id, lm in enumerate(op.multi_hand_landmarks[0].landmark):
            h, w, _ = frame.shape
            lmlist.append([id,int(lm.x*w),int(lm.y*h)])

        if len(lmlist)!=0:
            x1, y1 = lmlist[4][1], lmlist[4][2]
            x2, y2 = lmlist[8][1], lmlist[8][2]
            length = math.hypot(x1 - x2, y1 - y2)

        #Marking Thumb Tip and Index Tip
        cv2.circle(frame, (x1, y1), 15, (255, 255, 255))
        cv2.circle(frame, (x2, y2), 15, (255, 255, 255))
        if(length > 50):
            cv2.line(frame,(x1, y1), (x2, y2), (0, 255, 0), 3)
        else:
            cv2.line(frame,(x1, y1), (x2, y2), (0, 0, 255), 3)

        #Calculating and Set the Volume Level with Reference to Length
        vol = np.interp(length, [50, 220], [0, 100])
        osascript.osascript("set volume output volume {}".format(vol))
        volbar = np.interp(length, [50, 220], [400, 150])
        volper = np.interp(length, [50, 220], [0, 100])

        #Creating Volume Bar
        cv2.rectangle(frame, (50, 150), (80, 400), (0, 0, 0), 3)
        cv2.rectangle(frame, (50, int(volbar)), (80, 400), (0, 0, 0), -1)
        cv2.putText(frame, f'{int(volper)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0 ,0), 3)

    #Displaying the Result
    cv2.imshow("Controller", frame)

    #Exit
    if cv2.waitKey(1) == ord('q'):
        break

#Releasing Camera and Destroying Window
cam.release()
cv2.destroyAllWindows()