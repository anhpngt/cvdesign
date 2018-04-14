import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
import datetime, time
from matplotlib import pyplot as plt
from PIL import Image

import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import webbrowser
# import pyautogui
import uinput

url = "https://gemioli.com/hooligans/"

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
winH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FPS, 1)

PATH_TO_CKPT = 'models/05_07042018.pb'
PATH_TO_LABELS = 'models/05.pbtxt'
NUM_CLASSES = 2

winName = 'detection'

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Show fps value on image.
def draw_fps_on_image(fps, image_np):
  cv2.putText(image_np, fps, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)

start_time = datetime.datetime.now()
num_frames = 0
gameStarted = 0
posThreshold = 0.7
timeWall = time.time()

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    with uinput.Device([uinput.KEY_UP, uinput.KEY_DOWN,
                        uinput.KEY_LEFT, uinput.KEY_RIGHT,
                        uinput.KEY_ESC, uinput.KEY_SPACE]) as device:
      while True:
        begin_time = time.time()
        ret, image_np = cap.read()
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # User presses 'a' to start the game
        if gameStarted == 0:
          cv2.imshow(winName, image_np)
          if cv2.waitKey(10) & 0xFF == ord('a'):
            gameStarted = 1
            webbrowser.open_new(url)
            time.sleep(5)
            device.emit_click(uinput.KEY_SPACE)
            time.sleep(5)
            device.emit_click(uinput.KEY_ESC)
            time.sleep(5)
            device.emit_click(uinput.KEY_SPACE)
          continue

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_rgb_expanded = np.expand_dims(image_rgb, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_rgb_expanded})
        
        # Controls
        # if np.squeeze(scores)[0] > 0.9 and np.squeeze(classes)[0] == 1 and time.time() - timeWall > 0.5:
        if scores[0, 0] > 0.9 and classes[0, 0] == 1:
          detected_box = boxes[0, 0]
          crow = (detected_box[0] + detected_box[2]) / 2.0 - 0.5
          ccol = (detected_box[1] + detected_box[3]) / 2.0 - 0.5 # center row and col
          cdistsq = crow * crow + 2 * ccol * ccol
          cv2.circle(image_np, (int((ccol+0.5)*winW), int((crow+0.5)*winH)), 8, (0, 0, 255), -1)   
          # if cdistsq > 0.01:
          if True:
            timeWall = time.time()
            cangle = np.arctan2(crow, ccol)
            print('angle:{}'.format(cangle))
            if cangle > -0.6435 and cangle <= 0.6435:
              # pyautogui.hotkey('left')
              device.emit_click(uinput.KEY_LEFT)
              print('LEFT')
            elif cangle > 0.6435 and cangle <= 2.498:
              # pyautogui.hotkey('down')
              device.emit_click(uinput.KEY_DOWN)
              print('DOWN')
            elif cangle > 2.498 or cangle <= -2.498:
              # pyautogui.hotkey('right')
              device.emit_click(uinput.KEY_RIGHT)
              print('RIGHT')
            elif cangle > -2.498 and cangle < -0.6435:
              # pyautogui.hotkey('up')
              device.emit_click(uinput.KEY_UP)
              print('UP')
          # else:
            # print('Centerized ({}, {})'.format(crow, ccol))

        # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        #     image_np,
        #     # np.squeeze(boxes),
        #     # np.squeeze(classes).astype(np.int32),
        #     # np.squeeze(scores),
        #     np.array(boxes[0]),
        #     np.array(classes[0]).astype(np.int32),
        #     np.array(scores[0]),
        #     category_index,
        #     use_normalized_coordinates=True,
        #     min_score_thresh=0.8,
        #     line_thickness=4)

        # Calculate Frames per second (FPS)
        # num_frames += 1
        # elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        # fps = num_frames / elapsed_time
        # if (fps > 0):
        #   draw_fps_on_image("FPS : " + str(int(fps)), image_np)
        
        cv2.imshow(winName, image_np)
    
        if cv2.waitKey(5) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
        print('One process took: {} s'.format(time.time() - begin_time))