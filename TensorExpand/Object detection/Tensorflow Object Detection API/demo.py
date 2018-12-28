"""
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# 模型下载
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

将脚本放在 models/research/object_detection 路径下运行
sudo su
python3 demo.py xxx/ssd_mobilenet_v1_coco_2017_11_17.tar.gz xxx/image_path

输入的model可以是：
ssd_mobilenet_v1_coco_2017_11_17.tar.gz，
faster_rcnn_nas_coco_2018_01_28.tar.gz，
mask_rcnn_resnet50_atrous_coco_2018_01_28.tar.gz
等等(detection_model_zoo.md 下载的model都可以)
"""

# -*- coding:utf-8 -*-

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")
# from object_detection.utils import ops as utils_ops
from utils import ops as utils_ops

# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Object detection imports
# 以下是从对象检测模块导入的内容。
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as vis_util

from utils import label_map_util
from utils import visualization_utils as vis_util

assert len(sys.argv)>=2,"python3 demo.py xxx/ssd_mobilenet_v1_coco_2017_11_17.tar.gz xxx/image_path"

# Model preparation
# 只需将PATH_TO_FROZEN_GRAPH更改为指向新的.pb文件，就可以在此处加载使用export_inference_graph.py工具导出的任何模型。

# 默认情况下，我们在此处使用“带有Mobilenet的SSD”模型。
# 请参阅检测模型动物园以获取其他模型的列表，这些模型可以开箱即用，具有不同的速度和精度。
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

# What model to download.
model_path=sys.argv[1] # xxx/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

MODEL_NAME=model_path.split(".tar.gz")[0]
MODEL_FILE = MODEL_NAME + '.tar.gz'

DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Download Model
if not os.path.exists(model_path):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
'''
标签映射将索引映射到类别名称，因此当我们的卷积网络预测5时，我们知道这对应于飞机。 
这里我们使用内部实用程序函数，但任何返回字典映射整数到适当的字符串标签的东西都可以
'''
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

def glob_format(path,base_name = False):
    print('--------pid:%d start--------------' % (os.getpid()))
    # fmt_list = ('.xml',)
    fmt_list = ('.jpg', '.jpeg', '.png',)
    fs = []
    if not os.path.exists(path):return fs
    for root, dirs, files in os.walk(path):
        for file in files:
            item = os.path.join(root, file)
            # item = unicode(item, encoding='utf8')
            fmt = os.path.splitext(item)[-1]
            if fmt.lower() not in fmt_list:
                # os.remove(item)
                continue
            if base_name:fs.append(file)  # fs.append(os.path.splitext(file)[0])
            else:fs.append(item)
    print('--------pid:%d end--------------' % (os.getpid()))
    return fs

PATH_TO_TEST_IMAGES_DIR = sys.argv[2] #'test_images'
assert os.path.exists(PATH_TO_TEST_IMAGES_DIR),"image path not exists!"

# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
# import glob
# TEST_IMAGE_PATHS=glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,'*.jpg'))

TEST_IMAGE_PATHS=glob_format(PATH_TO_TEST_IMAGES_DIR)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

for image_path in TEST_IMAGE_PATHS:
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)

  image_np= np.concatenate((np.expand_dims(image_np[:,:,2],-1),
                            np.expand_dims(image_np[:,:,1],-1),
                            np.expand_dims(image_np[:,:,0],-1)),-1) # ==>BGR

  cv2.imwrite(image_path.replace('.jpg','_test.jpg'),
              image_np,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
  # cv2.COLOR_RGB2BGR(image_np)
  # plt.figure(figsize=IMAGE_SIZE)
  # plt.imshow(image_np)
  # plt.show()