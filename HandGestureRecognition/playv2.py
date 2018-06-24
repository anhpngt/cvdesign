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
import uinput

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
winH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
winW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FPS, 1)

PATH_TO_CKPT = 'models/all_five.pb'
PATH_TO_LABELS = 'models/all_five.pbtxt'
NUM_CLASSES = 6

winName = 'detection'
cv2.namedWindow('detection', cv2.WINDOW_NORMAL)

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

def put_code_on_image(code, color, image_np):
  if len(code) == 0:
    return
  codeStr = ''
  for i in range(len(code)):
    codeStr += str(code[i]) + ' '
  
  ret, baseline = cv2.getTextSize(codeStr, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
  cv2.rectangle(image_np, (15, 430+baseline), (15+ret[0], 430-ret[1]-5), (255, 255, 255), -1)
  cv2.putText(image_np, codeStr, (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

def put_hint_on_image(hint, image_np):
  ret, baseline = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
  cv2.rectangle(image_np, (15, 430+baseline), (15+ret[0], 430-ret[1]-5), (255, 255, 255), -1)
  cv2.putText(image_np, hint, (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

start_time = datetime.datetime.now()
num_frames = 0
detection = []
code = []
wallTime = time.time()

password = [2, 1, 4, 0]
hint = 'Successful! Proceed to next game.'
finished = False
waitCount = 0
successCount = 0
failureCount = 0

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      begin_time = time.time()
      ret, image_np = cap.read()

      if len(code) == len(password) and not finished:
        waitCount += 1
        if waitCount > 80:
          if code == password:
            finished = True
            continue
          else:
            if failureCount < 80:
              put_code_on_image(code, (0, 0, 255), image_np)
              cv2.imshow(winName, image_np)
              cv2.waitKey(5)
              failureCount += 1
              continue
            else:
              code.clear()
              failureCount = 0
              waitCount = 0
              continue
        else:
          put_code_on_image(code, (0, 0, 0), image_np)
          cv2.imshow(winName, image_np)
          cv2.waitKey(5)
          continue

      if finished == True:
        if successCount > 40:
          put_hint_on_image(hint, image_np)
          cv2.imshow(winName, image_np)
          if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
          continue
        else:
          successCount += 1
          put_code_on_image(code, (0, 255, 0), image_np)
          cv2.imshow(winName, image_np)
          cv2.waitKey(5)
          continue

      image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

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
      if scores[0, 0] > 0.9:
        detected_class = int(classes[0, 0])
        detection.append(detected_class)

      if time.time() - wallTime > 2:
        if len(detection) > 40:
          bin_count = np.bincount(detection)
          most_common = bin_count.argmax()
          if most_common != detection[-1]:
            detection = []
            continue
          # print('Output: {} (count: {})'.format(most_common - 1, bin_count[most_common]))
          code.append(most_common - 1)
          detection = []
          print(code)

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          # np.squeeze(boxes),
          # np.squeeze(classes).astype(np.int32),
          # np.squeeze(scores),
          np.array(boxes[0]),
          np.array(classes[0]).astype(np.int32),
          np.array(scores[0]),
          category_index,
          use_normalized_coordinates=True,
          min_score_thresh=0.9,
          line_thickness=4)

      # Calculate Frames per second (FPS)
      num_frames += 1
      elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
      fps = num_frames / elapsed_time
      if (fps > 0):
        draw_fps_on_image("FPS : " + str(int(fps)), image_np)
      
      # Put code on image
      put_code_on_image(code, (255, 0, 0), image_np)

      cv2.imshow(winName, image_np)
      if cv2.waitKey(5) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
      # print('One process took: {} s'.format(time.time() - begin_time))