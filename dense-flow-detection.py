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

cap.set(cv2.CAP_PROP_FPS, 10)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'fps={fps}')



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('hello')

frames = [] # frame history
SMOOTHING = 15 # num frames to calc average over

while True:

    # Capture
    ret, frame = cap.read()
    if ret == False:
        logging.error("Error capturing")
        continue

    # Analyse
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    asFloat = np.float32(gray)
    frames.append(asFloat)

    if len(frames) == SMOOTHING:
        s = sum(frames)
        frames.pop(0)

        smooth = s/SMOOTHING
        smoothAsInt = np.uint8(smooth)
        cv2.imshow(f'smoothing {SMOOTHING}', smoothAsInt)

    
    g1 = gray.copy()

    # Display
    cv2.imshow('camera', color)


    # keyboard support
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()