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

import webbrowser, pyautogui

url = "https://gemioli.com/hooligans/"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
winH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FPS, 1)

PATH_TO_CKPT = 'models/all_five.pb'
PATH_TO_LABELS = 'models/object-detection.pbtxt'
NUM_CLASSES = 6

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
    while True:
      ret, image_np = cap.read()
      image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

      # User presses 'a' to start the game
      if gameStarted == 0:
        cv2.imshow('object detection', image_np)
        if cv2.waitKey(10) & 0xFF == ord('a'):
          gameStarted = 1
          webbrowser.open_new(url)
          time.sleep(5)
          pyautogui.keyDown('space')
          time.sleep(5)
          pyautogui.keyDown('esc')
          time.sleep(5)
          pyautogui.keyDown('space')

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
      if np.squeeze(scores)[0] > 0.9 and np.squeeze(classes)[0] == 1 and time.time() - timeWall > 0.5:
        detected_box = np.squeeze(boxes)[0]
        cx = (detected_box[1] + detected_box[3]) / 2.0 - 0.5
        cy = (detected_box[0] + detected_box[2]) / 2.0 - 0.5
        cdistsq = cx * cx + cy * cy
        # print('Pos: {}, {}'.format(int(cy*winH), int(cx*winW)))
        cv2.circle(image_np, (int((cx+0.5)*winW), int((cy+0.5)*winH)), 8, (0, 0, 255), -1)   
        if cdistsq > 0.01:
          timeWall = time.time()
          cangle = np.arctan2(cy, cx)
          print('Angle: {}'.format(cangle))
          
          if cangle > -0.785 and cangle < 0.785:
            pyautogui.hotkey('left')
            print('LEFT')
          elif cangle > 0.785 and cangle < 1.571:
            pyautogui.hotkey('down')
            print('DOWN')
          elif cangle > 1.571 or cangle < -1.571:
            pyautogui.hotkey('right')
            print('RIGHT')
          elif cangle > -1.571 and cangle < -0.785:
            pyautogui.hotkey('up')
            print('UP')
        else:
          print('Centerized ({}, {})'.format(cx, cy))

      # Visualization of the results of a detection.
      # vis_util.visualize_boxes_and_labels_on_image_array(
      #     image_np,
      #     np.squeeze(boxes),
      #     # np.array(boxes[0]),
      #     np.squeeze(classes).astype(np.int32),
      #     # np.array(classes[0]).astype(np.int32),
      #     np.squeeze(scores),
      #     # np.array(scores[0]),
      #     category_index,
      #     use_normalized_coordinates=True,
      #     min_score_thresh=0.3,
      #     line_thickness=4)

      # Calculate Frames per second (FPS)
      num_frames += 1
      elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
      fps = num_frames / elapsed_time
      if (fps > 0):
        draw_fps_on_image("FPS : " + str(int(fps)), image_np)
       
      cv2.imshow('object detection', image_np)
  
      if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break