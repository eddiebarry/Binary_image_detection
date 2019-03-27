# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
import os
import math
import random

import numpy as np
import tensorflow as tf
from PIL import Image


def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def run_label(file_name):
      # file_name = "test_img/test4.jpeg"
      model_file = "model_info/retrained_graph.pb"
      label_file = "model_info/retrained_labels.txt"
      input_height = 224
      input_width = 224
      input_mean = 128
      input_std = 128
      input_layer = "input"
      output_layer = "final_result"
      # parser = argparse.ArgumentParser()
      # parser.add_argument("--image", help="image to be processed")
      # parser.add_argument("--graph", help="graph/model to be executed")
      # parser.add_argument("--labels", help="name of file containing labels")
      # parser.add_argument("--input_height", type=int, help="input height")
      # parser.add_argument("--input_width", type=int, help="input width")
      # parser.add_argument("--input_mean", type=int, help="input mean")
      # parser.add_argument("--input_std", type=int, help="input std")
      # parser.add_argument("--input_layer", help="name of input layer")
      # parser.add_argument("--output_layer", help="name of output layer")
      # args = parser.parse_args()

      # if args.graph:
      #   model_file = args.graph
      # if args.image:
      #   file_name = args.image
      # if args.labels:
      #   label_file = args.labels
      # if args.input_height:
      #   input_height = args.input_height
      # if args.input_width:
      #   input_width = args.input_width
      # if args.input_mean:
      #   input_mean = args.input_mean
      # if args.input_std:
      #   input_std = args.input_std
      # if args.input_layer:
      #   input_layer = args.input_layer
      # if args.output_layer:
      #   output_layer = args.output_layer

      graph = load_graph(model_file)
      t = read_tensor_from_image_file(file_name,
                                      input_height=input_height,
                                      input_width=input_width,
                                      input_mean=input_mean,
                                      input_std=input_std)

      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = graph.get_operation_by_name(input_name);
      output_operation = graph.get_operation_by_name(output_name);

      with tf.Session(graph=graph) as sess:
        start = time.time()
        results = sess.run(output_operation.outputs[0],
                          {input_operation.outputs[0]: t})
        end=time.time()
      results = np.squeeze(results)

      top_k = results.argsort()[-5:][::-1]
      # print(results)

      # lol = input()
      labels = load_labels(label_file)

      # print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
      # template = "{} (score={:0.5f})"
      for i in top_k:
        # print(i)
        # print("\n this is the id \n")
        return (labels[i],results[i])
        # print(template.format(labels[i], results[i]))



 
 
if __name__ == '__main__':
    #image = 'grasshopper.jpg'
    #crop(image, (161, 166, 706, 1050), 'cropped.jpg')
    # run_label();
    # dir_path="~"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # print(dir_path)
    crops_store =dir_path+"/cropped"
    dir_path += "/test_img"


    #100 random crops
    curr_dir = os.listdir(dir_path)
    num_crops = 10
    crop_percentage = 0.5
    threshold = 0.7
    
    for pic_names in curr_dir:
        if(pic_names == ".DS_Store"):
            continue
        print(pic_names)
        # run_label(dir_path+"/" +filename)
        filename = dir_path+"/" + pic_names
        # run_label(filename)

        ## 100 random crops
        max_prob ={}
        fl_loc = {}

        with Image.open(filename) as im:
            for x in xrange(1,num_crops):
                newname = pic_names.replace('.', '_{:03d}.'.format(x))
                w, h = im.size
                dx = dy = math.ceil(min(w,h)*crop_percentage)
                x = random.randint(0, w-dx-1)
                y = random.randint(0, h-dy-1)
                # print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
                im.crop((x,y, x+dx, y+dy))\
                  .save(os.path.join(crops_store, newname))
                # template = "{} (score={:0.5f})"
                file_location = crops_store+"/"+newname
                best_prediction = run_label(file_location)
                print(file_location)
                print(best_prediction[0])
                print(best_prediction[1])
                os.remove(file_location)


                newname = newname.replace('_', "_" + best_prediction[0] + "_")
                print(newname)
                im.crop((x,y, x+dx, y+dy))\
                  .save(os.path.join(crops_store, newname))
                file_location = crops_store+"/"+newname
                
                # print("\n")
                # print(template.format(run_label(crops_store+"/"+newname)))
                # lol = input()

                to_delete =file_location
                delete_current = True

                if(best_prediction[1]>threshold):
                    if(best_prediction[0] in max_prob):
                        if(max_prob[best_prediction[0]] < best_prediction[1]):
                            #updating max probability
                            to_delete = fl_loc[best_prediction[0]]
                            os.remove(to_delete)

                            max_prob[best_prediction[0]] = best_prediction[1]
                            fl_loc[best_prediction[0]] = file_location
                            delete_current = False
                        
                    
                    else:
                        max_prob[best_prediction[0]] = best_prediction[1]
                        fl_loc[best_prediction[0]] = file_location
                        delete_current = False
                    
                    
                
                if(delete_current):
                    os.remove(file_location)
                









        

# if __name__ == '__main__':
#     #image = 'grasshopper.jpg'
#     #crop(image, (161, 166, 706, 1050), 'cropped.jpg')
#     # run_label();
#     # dir_path="~"
#     dir_path = os.path.dirname(os.path.realpath(__file__))
#     print(dir_path)

#     crops_store =dir_path+"/cropped"

#     dir_path += "/test_img"

#     #100 random crops

#     for pic_names in os.listdir(dir_path):
#         print(filename)
#         # run_label(dir_path+"/" +filename)
#         filename = dir_path+"/" + pic_names
#         run_label(filename)
#         ## 100 random crops



