import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

###################################################################
brushThickness = 15
eraserThickness = 100
###################################################################

folderPath = "header"
mylist = os.listdir(folderPath)
overlayList = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]

drawColor = (0,255,255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

xp,yp = 0,0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    #1 import the image
    success, img = cap.read()
    img = cv2.flip(img,1)

    #2 find hand landmark
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw = False)

    if len(lmlist) != 0:

        #print(lmlist)

        #tip of index and middle finger
        x1,y1 = lmlist[8][1:]
        x2,y2 = lmlist[12][1:]

    #3 check which finger is up
        fingers = detector.fingersUp()
        #print(fingers)

    #4 if selection mode -> two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            print("Selection mode")
            if y1 < 125:
                if 550 < x1 < 750:
                    header = overlayList[0]
                    drawColor = (0,255,255)
                elif 800 < x1 < 950:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 1000 < x1 < 1100:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1150 < x1 < 1250:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)



    # 5 if drawing mode -> index finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")

            if xp == 0 and yp == 0:
                xp, yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

            else:
                cv2.line(img, (xp,yp),(x1,y1),drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1,y1



    #setting the header image
    img[0:125, 0:1280] = header

    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    cv2.imshow("image",img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)