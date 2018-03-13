# -*- coding:utf-8 -*-
#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
import os
import tensorflow as tf
import numpy as np
import random
import cv2
import math
import sys
from libs.configs import config_v1 # 导入参数配置

FLAGS = tf.app.flags.FLAGS

class ImageReader(object):
  def __init__(self):
    self._decode_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_data)

  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 1
    return image

class ShapesDataset(object):
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"  # 源名称不能包含点
        # Does the class exist already?  class是否已经存在？
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_shapes(self, count, height, width):
        # self.add_class("shapes", 0, 'BG') # 标签0默认为背景
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        for i in range(count):
            comb_image, gt_boxes, mask, mask_= self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           image=comb_image, gt_boxes=gt_boxes,mask=mask, mask_=mask_)

    '''
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = info['image']
        return image
    
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = info['mask']
        class_ids = info['class_ids']
        return mask, class_ids.astype(np.int32)
    '''
    def random_shape(self,height, width):
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def mask2box(self,mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]

    def random_image(self,height, width):
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # images=[]
        mask = []
        class_id = []

        bg_color = bg_color.reshape([1, 1, 3])
        image = np.ones([height, width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)

        N = random.randint(1, 4)
        for _ in range(N):
            image_mask = np.zeros([height, width], dtype=np.uint8)
            shape, color, dims = self.random_shape(height, width)
            x, y, s = dims
            if shape == 'square':
                cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
                cv2.rectangle(image_mask, (x - s, y - s), (x + s, y + s), 1, -1)  # 0、1组成的图片
                # images.append(img)
                mask.append(image_mask)
                class_id.append(1)  # 对应class ID 1

            elif shape == "circle":
                cv2.circle(image, (x, y), s, color, -1)
                cv2.circle(image_mask, (x, y), s, 1, -1)
                # images.append(img)
                mask.append(image_mask)
                class_id.append(2)  # 对应class ID 2

            elif shape == "triangle":
                points = np.array([[(x, y - s),
                                    (x - s / math.sin(math.radians(60)), y + s),
                                    (x + s / math.sin(math.radians(60)), y + s),
                                    ]], dtype=np.int32)
                cv2.fillPoly(image, points, color)
                cv2.fillPoly(image_mask, points, 1)
                # images.append(img)
                mask.append(image_mask)
                class_id.append(3)  # 对应class ID 3

        # images=np.asarray(images,np.float32) # [h,w,c]
        mask = np.asarray(mask, np.uint8).transpose([1, 2, 0])  # [h,w,instance count]
        class_id = np.asarray(class_id, np.uint8)  # [instance count,]

        # Handle occlusions 处理遮挡情况
        count = mask.shape[-1]
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # 如果mask 全为0 也就是意味着完全被遮挡，需丢弃这种mask，否则训练会报错
            # （而实际标准mask时不会出现这种情况的，因为完全遮挡了没办法标注mask）
            if np.sum(mask[:, :, i]) < 1:  # 完全被遮挡
                mask = np.delete(mask, i, axis=-1)
                class_id = np.delete(class_id, i)  # 对应的mask的class id 也需删除

        count = mask.shape[-1] # 完全覆盖的mask会被删除，重新计算mask个数
        bboxes = []  # [instance count,4]
        [bboxes.append(self.mask2box(mask[:, :, i])) for i in range(count)]
        bboxes = np.asarray(bboxes)
        gt_boxes = np.hstack((bboxes, class_id[:, np.newaxis]))  # [instance count,5] 前4列为boxs，最后一列为 class id

        mask_ = np.zeros((height, width), dtype=np.float32)  # [h,w]
        for i in range(count):
            mask_ += mask[:, :, i] * class_id[i]

        mask = np.asarray(mask, np.uint8).transpose([2, 0, 1])  # [instance count,h,w]

        return image, gt_boxes, mask, mask_


# --------参数定义----------
num_images = 500
image_h = 128
image_w = 128
# dataset_train = ShapesDataset()
# dataset_train.load_shapes(train_images, image_h, image_w)

# image=dataset_train.image_info[0]['image']
# print(image.shape)

def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def _to_tfexample_coco_raw(image_id, image_data, label_data,
                           height, width,
                           num_instances, gt_boxes, masks):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(image_data),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/num_instances': _int64_feature(num_instances),  # N
    'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2, classid)
    'label/gt_masks': _bytes_feature(masks),  # of shape (N, height, width)
    'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
  }))

def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'coco_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _add_to_tfrecord(record_dir,num_images,image_h, image_w, split_name):
    """Loads image files and writes files to a TFRecord.
      Note: masks and bboxes will lose shape info after converting to string.
      """
    dataset_train = ShapesDataset()
    dataset_train.load_shapes(num_images, image_h, image_w)

    num_shards = int(num_images /5 ) # 2500
    num_per_shard = int(math.ceil(num_images / float(num_shards)))
    height, width=image_h, image_w

    for shard_id in range(num_shards):
        record_filename = _get_dataset_filename(record_dir, split_name, shard_id, num_shards)
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)

        with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, num_images)
          for i in range(start_ndx, end_ndx):
            if i % 50 == 0:
                sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
                  i + 1, num_images, shard_id))
                sys.stdout.flush()

            img= dataset_train.image_info[i]['image']
            gt_boxes= dataset_train.image_info[i]['gt_boxes']
            masks= dataset_train.image_info[i]['mask']
            mask= dataset_train.image_info[i]['mask_']

            img_raw = img.tostring()
            mask_raw = mask.tostring()
            img_id=i
            example = _to_tfexample_coco_raw(
                img_id,
                img_raw,
                mask_raw,
                height, width, gt_boxes.shape[0],
                gt_boxes.tostring(), masks.tostring())

            tfrecord_writer.write(example.SerializeToString())
          sys.stdout.write('\n')
          sys.stdout.flush()


def run(dataset_dir, dataset_split_name='train2014',num_images=20,image_h=128, image_w=128):
    """Runs the download and conversion operation.
  
    Args:
      dataset_dir: The dataset directory where the dataset is stored.
    """
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    # for url in _DATA_URLS:
    #   download_and_uncompress_zip(url, dataset_dir)

    record_dir = os.path.join(dataset_dir, 'records')
    # annotation_dir = os.path.join(dataset_dir, 'annotations')

    if not tf.gfile.Exists(record_dir):
        tf.gfile.MakeDirs(record_dir)

    # process the training, validation data:
    _add_to_tfrecord(record_dir,
                     num_images,
                     image_h,
                     image_w,
                     dataset_split_name)


    print('\nFinished converting the coco dataset!')

'''
def main():
  if not os.path.isdir('./output/mask_rcnn'):
    os.makedirs('./output/mask_rcnn')
  # FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string("dataset_dir", "./output/mask_rcnn", "save tfrecord path")
  tf.app.flags.DEFINE_string("dataset_name", "coco", "dataset name")
  tf.app.flags.DEFINE_string("dataset_split_name", "train", "dataset split name train or test")
  FLAGS = tf.app.flags.FLAGS


  # if not FLAGS.dataset_name:
  #   raise ValueError('You must supply the dataset name with --dataset_name')
  # if not FLAGS.dataset_dir:
  #   raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'coco':
    run(FLAGS.dataset_dir, FLAGS.dataset_split_name,num_images=)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)

if __name__ == '__main__':
  main()
  # tf.app.run()
'''

def main(_):
  if not os.path.isdir('./output/mask_rcnn'):
    os.makedirs('./output/mask_rcnn')
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  elif FLAGS.dataset_name == 'coco':
    run(FLAGS.dataset_dir, FLAGS.dataset_split_name,num_images=100)
  else:
    raise ValueError(
        'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)

if __name__ == '__main__':
  tf.app.run()
