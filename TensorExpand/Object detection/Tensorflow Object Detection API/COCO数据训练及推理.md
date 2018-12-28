<font size=4>

[toc]

# bbox
## COCO数据训练及推理

1、下载COCO 数据

```python
sudo apt-get install aria2
aria2c -c http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/train2014.zip 
aria2c -c http://msvocds.blob.core.windows.net/coco2014/val2014.zip 
# 以上3个url链接分别为2014年的 annotations、train data、val data，下载不全

# 2017
http://images.cocodataset.org/zips/train2017.zip 
http://images.cocodataset.org/annotations/annotations_trainval2017.zip

http://images.cocodataset.org/zips/val2017.zip 
http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

http://images.cocodataset.org/zips/test2017.zip 
http://images.cocodataset.org/annotations/image_info_test2017.zip 


# MPII数据集
http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz 
http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip

http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_batch1.tar.gz 
http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_sequences_batch2.tar.gz
```

2、链接数据

```
cd tensorflow/models/research/
ln -s xxxx/coco `pwd`/coco
```
3、转成tfrecode

```python
# 如果使用mask，默认只使用bbox（include_masks=False）
# 设置--include_masks=True

TRAIN_IMAGE_DIR=`pwd`/coco/train2014
VAL_IMAGE_DIR=`pwd`/coco/val2014
TEST_IMAGE_DIR=`pwd`/coco/val2014 # test2014
TRAIN_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_train2014.json
VAL_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_val2014.json
TESTDEV_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_val2014.json # `pwd`/coco/annotations/instances_test2014.json
OUTPUT_DIR=`pwd`/mscoco
python3 object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
```

如果出现`ImportError: No module named 'contextlib2'`

```python
# Python 2
sudo apt-get install python-contextlib2

# Python 3
sudo apt-get install python3-contextlib2
```
## train

```
cd tensorflow/models/research
mkdir 20180823
mv ./mscoco ./20180823

cp -r object_detection/data/mscoco_label_map.pbtxt ./20180823
cp -r object_detection/samples/configs/faster_rcnn_resnet101_coco.config ./20180823
```

在[此处](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)下载[faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)解压放在目录`20180823`

此时目录结构为

```
<20180823>
├── faster_rcnn_resnet101_coco_2018_01_28
│   └── saved_model
│       └── variables
├── mscoco
	└── *.record
├── mscoco_label_map.pbtxt
└── faster_rcnn_resnet101_coco.config
```

修改`faster_rcnn_resnet101_coco.config`配置

```python
第 106 和 107 行内容改为：（表示不微调，从头开始训练模型）
# fine_tune_checkpoint: "20180823/model.ckpt"  （注释掉）
from_detection_checkpoint: false  （设为false）

# or 如果微调，则修改成
fine_tune_checkpoint: "20180823/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt"
from_detection_checkpoint: true

# or 如果要接着自己训练的模型继续训练（如：中途中断了）
fine_tune_checkpoint: "20180823/model/model.ckpt-1102" # 自己训练的模型保存位置
from_detection_checkpoint: true

# --------------------------------
# 修改 114行train_input_reader中的input_path和label_map_path
# input_path："20180823/mscoco/coco_train.record-?????-of-00100"
# label_map_path:"20180823/mscoco_label_map.pbtxt"

# 121行eval_input_reader中的input_path和label_map_path
# input_path："20180823/mscoco/coco_val.record-?????-of-00010"
# label_map_path:"20180823/mscoco_label_map.pbtxt"
```

运行以下命令进行train

```python
cd ../ # (cd ./models/resarch)

# From the tensorflow/models/research/
PIPELINE_CONFIG_PATH='./20180823/faster_rcnn_resnet101_coco.config'
MODEL_DIR='./20180823/model' # 训练保存的model 位置
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
```
参考：https://github.com/tensorflow/models/issues/4856

如果训练过程出现TypeError: can't pickle dict_values objects，需修改https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib.py#L390 将category_index.values()改成list(category_index.values())

## 查看 TensorBoard
由于上面的训练过程不会打印任何训练的信息，需从`tensorboard`中查看

```
tensorboard --logdir='./20180823/model'
```
## 导出训练模型做推理
在确定要导出的候选检查点之后，从`tensorflow/models/research`运行以下命令：

```python
cd tensorflow/models/research
# From tensorflow/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='./20180823/faster_rcnn_resnet101_coco.config'
TRAINED_CKPT_PREFIX='./20180823/model/model.ckpt-1030' 
EXPORT_DIR='./20180823/model/frozen_pb'
python3 object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

## 部署推理服务

```python
# -*- coding:utf-8 -*-
"""
根据实际修改以下内容，其他的基本保持不变：
NUM_CLASSES = 90 # coco是90类（不含背景）
PATH_TO_FROZEN_GRAPH # 指定冻结的pb文件路径
PATH_TO_LABELS # 指定label map 即标签id与标签名对应
PATH_TO_TEST_IMAGES_DIR # 指定要推理的图片路径
"""

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
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Object detection imports
# 以下是从对象检测模块导入的内容。
from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# Model preparation
# 只需将PATH_TO_FROZEN_GRAPH更改为指向新的.pb文件，就可以在此处加载使用export_inference_graph.py工具导出的任何模型。

# 默认情况下，我们在此处使用“带有Mobilenet的SSD”模型。
# 请参阅检测模型动物园以获取其他模型的列表，这些模型可以开箱即用，具有不同的速度和精度。
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

"""
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
"""
NUM_CLASSES = 90#90

"""
# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
"""
PATH_TO_FROZEN_GRAPH='./model/frozen_pb/frozen_inference_graph.pb'
PATH_TO_LABELS='./mscoco_label_map.pbtxt'

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
PATH_TO_TEST_IMAGES_DIR = '../object_detection/test_images'
# TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
import glob
TEST_IMAGE_PATHS=glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR,'*.jpg'))

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
```

# mask
## 转成tfrecode(只是这里与bbox不一样，其他的都一样)
其他不一样的地方就是`config`文件更换（对应起来）
```python
# 如果使用mask，默认只使用bbox（include_masks=False）
# 设置--include_masks=True

TRAIN_IMAGE_DIR=`pwd`/coco/train2014
VAL_IMAGE_DIR=`pwd`/coco/val2014
TEST_IMAGE_DIR=`pwd`/coco/val2014 # test2014
TRAIN_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_train2014.json
VAL_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_val2014.json
TESTDEV_ANNOTATIONS_FILE=`pwd`/coco/annotations/instances_val2014.json # `pwd`/coco/annotations/instances_test2014.json
OUTPUT_DIR=`pwd`/mscoco
python3 object_detection/dataset_tools/create_coco_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --include_masks=True \
      --output_dir="${OUTPUT_DIR}"
```

## train

```
cd tensorflow/models/research
mkdir 20180823
mv ./mscoco ./20180823

cp -r object_detection/data/mscoco_label_map.pbtxt ./20180823
cp -r object_detection/samples/configs/mask_rcnn_resnet50_atrous_coco.config ./20180823
```

在[此处](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)下载[mask_rcnn_resnet50_atrous_coco]()解压放在目录`20180823`


修改`mask_rcnn_resnet50_atrous_coco.config`配置

```python
第 128 和 129 行内容改为：（表示不微调，从头开始训练模型）
# fine_tune_checkpoint: "20180823/model.ckpt"  （注释掉）
from_detection_checkpoint: false  （设为false）

# or 如果微调，则修改成
fine_tune_checkpoint: "20180823/mask_rcnn_resnet50_atrous_coco_2018_01_28/model.ckpt"
from_detection_checkpoint: true

# or 如果要接着自己训练的模型继续训练（如：中途中断了）
fine_tune_checkpoint: "20180823/model/model.ckpt-1102" # 自己训练的模型保存位置
from_detection_checkpoint: true

# --------------------------------
# 修改 141行train_input_reader中的input_path和label_map_path
# input_path："20180823/mscoco/coco_train.record-?????-of-00100"
# label_map_path:"20180823/mscoco_label_map.pbtxt"

# 157行eval_input_reader中的input_path和label_map_path
# input_path："20180823/mscoco/coco_val.record-?????-of-00010"
# label_map_path:"20180823/mscoco_label_map.pbtxt"
```
运行以下命令进行train

```python
# From the tensorflow/models/research/
PIPELINE_CONFIG_PATH='./20180823/mask_rcnn_resnet50_atrous_coco.config'
MODEL_DIR='./20180823/model' # 训练保存的model 位置
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
```

## 导出训练模型做推理
在确定要导出的候选检查点之后，从`tensorflow/models/research`运行以下命令：

```python
cd tensorflow/models/research
# From tensorflow/models/research/

INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH='./20180823/mask_rcnn_resnet50_atrous_coco.config'
TRAINED_CKPT_PREFIX='./20180823/model/model.ckpt-1030' 
EXPORT_DIR='./20180823/model/frozen_pb'
python3 object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```
## 部署推理
参考：`bbox`部分