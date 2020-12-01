#!/usr/bin/python3

import cv2
import time
import logging

import numpy as np

lk_params = dict( winSize  = (5,5),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

CAM=1

cap = cv2.VideoCapture(1)

t0 = time.time()
frames = 0

gain = cap.get(cv2.CAP_PROP_GAIN)
print(f'gain={gain}')
cap.set(cv2.CAP_PROP_GAIN, 0.01)
gain = cap.get(cv2.CAP_PROP_GAIN)
print(f'gain={gain}')


fps = cap.get(cv2.CAP_PROP_FPS)
print(f'fps={fps}')
cap.set(cv2.CAP_PROP_FPS, 10)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'fps={fps}')


c1 = None # corners in previous frame
g1 = None # previous frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('hello')

while True:

    # Capture
    ret, frame = cap.read()

    # Analyse
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=25,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=11)

    if corners is not None:

        # find optical flow
        if c1 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(g1, gray, corners, None, **lk_params)
            similar = sum(st)/float(corners.shape[0])
            if similar < 0.9:
                logging.info(f'movement, s={similar}')

        g1 = gray.copy()
        c1 = corners.copy()

        # draw corners
        cornersInt = np.int0(corners)

        for i in cornersInt:
            x,y = i.ravel()
            cv2.circle(color,(x,y),3,255,-1)


    # Display
    cv2.imshow('camera', color)



    frames += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t1 = time.time()
    if (t1 - t0) > 1.0 :
        fps = frames/(t1-t0)
        #print(f'FPS: {fps}')
        t0 = t1
        frames = 1

cap.release()
cv2.destroyAllWindows()