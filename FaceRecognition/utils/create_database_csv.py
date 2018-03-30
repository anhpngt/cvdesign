from __future__ import print_function
import os
import pandas as pd

database_dir = '../faces_database'
sub_dir = ['att_faces', 'Chi_Siong', 'Kah_Yooi', 'Samuel', 'Tuan_Anh']

output_image_paths = []
output_labels = []

real_label = 100
for dir_item in sub_dir:
  working_path = os.path.join(database_dir, dir_item)

  if not os.path.isdir(working_path):
    print('{} is not a directory!'.format(working_path))
    break

  if dir_item == 'att_faces':
    for face_dir in os.listdir(working_path): # every dir in att_faces
      if os.path.isdir(os.path.join(working_path, face_dir)):
        image_label = int(face_dir[1:])
        for file in os.listdir(os.path.join(working_path, face_dir)):
          image_path = os.path.abspath(os.path.join(working_path, face_dir, file))
          
          # append
          output_image_paths.append(image_path)
          output_labels.append(image_label)
      else:
        print('Skipping {}, not a directory in {}'.format(face_dir, working_path))
  else: # our faces
    for image in os.listdir(working_path):
      image_path = os.path.abspath(os.path.join(working_path, image))

      output_image_paths.append(image_path)
      output_labels.append(real_label)
    
    # to next real face
    real_label += 1

data = pd.DataFrame({'image': output_image_paths, 'label': output_labels})
data.to_csv(os.path.join(database_dir, 'database.csv'), header=None, index=None)
print('Done.')