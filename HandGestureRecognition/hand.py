import sys
import time

import cv2
import numpy as np

if __name__ == '__main__':
  if(len(sys.argv) < 2):
    print('Please input template img.')
    sys.exit()
  
  template = cv2.imread(sys.argv[1], 0)
  template = cv2.resize(template, (70, 100))

  cap = cv2.VideoCapture(0)
  
  while(True):
    t = time.time()
    ret, frame_bgr = cap.read()
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= 0.8)
    result_roi = []
    for pt in zip(*loc[::-1]):
      if len(result_roi) == 0:
        result_roi.append(pt)
      for item in result_roi:
        dist_x = pt[0] - item[0]
        dist_y = pt[1] - item[1]
        dist_sq = dist_x * dist_x + dist_y * dist_y
        if dist_sq > int(h / 2):
          result_roi.append(pt)
          break

    for pt in result_roi:
      cv2.rectangle(frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)


    cv2.imshow('frame', frame_bgr)
    print('Frame took {0:.5f} s'.format(time.time() - t))
    cv2.waitKey(5)