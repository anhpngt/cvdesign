#! /usr/bin/python3
from __future__ import print_function
import sys
import time

import numpy as np
import cv2

if __name__ == '__main__':
  cap = cv2.VideoCapture(0)
  roiWidth = 100
  roiHeight = 150

  # Process loop-based
  while(True):
    t = time.time()
    ret, framesrc = cap.read()
    if ret == False:
      print('Cannot get video frame')
      continue
    
    # Draw ROI, assuming 640x480 image
    frame = framesrc.copy()
    imHeight, imWidth, _ = frame.shape
    pt1 = (int(imWidth/2 - roiWidth), int(imHeight/2 - roiHeight))
    pt2 = (int(imWidth/2 + roiWidth), int(imHeight/2 + roiHeight))
    cv2.rectangle(frame, pt1, pt2, (0,0,255), 2)

    # Main algorithm
    framehsv = cv2.cvtColor(framesrc, cv2.COLOR_BGR2HSV)


    # Visualize
    cv2.imshow('image', framehsv)
    cv2.waitKey(5)
    print('Process took {} sec'.format(time.time() - t))
    