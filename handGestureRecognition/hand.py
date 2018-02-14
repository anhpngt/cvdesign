import sys

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
    ret, frame_bgr = cap.read()
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]

    # All the 6 methods for comparison in a list
    # methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
    #             'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    # Apply template Matching
    res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    loc = np.where(res >= 0.8)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    for pt in zip(*loc[::-1]):
      cv2.rectangle(frame_bgr, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)


    cv2.imshow('frame', frame_bgr)
    cv2.waitKey(5)

