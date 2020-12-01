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
logging.info(f'fps={fps}')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logging.info(f'Frame size: {frame_width}x{frame_height}')



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('hello')

smooth = None
S = 0.8
smoothDiff = None
DS = 0.9

def makeFileName(t):
    return time.strftime('%Y-%m-%d-%H-%M.avi', time.localtime(t))

CLIP_LEN = 60 # seconds
t0 = time.time()
tEnd = t0 + CLIP_LEN

writer = None
fourcc = cv2.VideoWriter_fourcc(*'X264')

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

    if smooth is None:
        smooth = asFloat
    else:
        smooth = smooth * S + asFloat * (1.0-S)
    
    smoothAsInt = np.uint8(smooth)
    #cv2.imshow(f'smoothing {S}', smoothAsInt)

    diff = abs(smooth - asFloat)
    diffAsInt = np.uint8(diff)
    #cv2.imshow(f'diff to smooth', diffAsInt)
    dscalar = diff.sum() / (diff.shape[0] * diff.shape[1])
    
    if smoothDiff is None:
        smoothDiff = dscalar
    else:
        smoothDiff = DS * smoothDiff + (1.0-DS) * dscalar
    
    print(f'diff ={dscalar:0.3} / {smoothDiff:0.3}')

    

    # Display
    t = time.time()
    ts = time.strftime('%x %X', time.localtime(t))
    print(ts)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(color, ts, (2,15), font, 0.5,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('camera', color)

    if writer is None:
        fn = makeFileName(t0)
        logging.info(f'Starting video file {fn}')
        writer = cv2.VideoWriter(fn, fourcc=fourcc, fps=fps, frameSize=(frame_width, frame_height))
    writer.write(color)

    if t >= tEnd:
        writer.release()
        t0 = tEnd
        tEnd += CLIP_LEN
        writer = None

    # keyboard support
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break


cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()