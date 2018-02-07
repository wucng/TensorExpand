#! /usr/bin/python
# -*- coding: utf8 -*-

'''
参考：http://blog.csdn.net/wc781708249/article/details/78043099

wget https://s3.amazonaws.com/cadl/models/vgg16.tfmodel

直接使用VGG16 做预测 ，直接使用输入层与输出层
'''
import numpy as np
import tensorflow as tf
import download # pip install download
import os
import PIL.Image

path_graph_def = "./vgg16/vgg16.tfmodel"

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    # 导入pb文件
    with open(path_graph_def, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # Name of the tensor for feeding the input image.
        tensor_name_input_image = "images:0"

        # Names of the tensors for the dropout random-values..
        tensor_name_dropout = 'dropout/random_uniform:0'
        tensor_name_dropout1 = 'dropout_1/random_uniform:0'
        prob='prob:0'

        # tensor_name_input_image_ = sess.graph.get_tensor_by_name(tensor_name_input_image)
        # tensor_name_dropout_=sess.graph.get_tensor_by_name(tensor_name_dropout)
        # tensor_name_dropout1_ = sess.graph.get_tensor_by_name(tensor_name_dropout1)
        prob_=sess.graph.get_tensor_by_name(prob)

        def create_feed_dict(image):
            """
            Create and return a feed-dict with an image.
            :param image:
                The input image is a 3-dim array which is already decoded.
                The pixels MUST be values between 0 and 255 (float or int).
            :return:
                Dict for feeding to the graph in TensorFlow.
            """

            # Expand 3-dim array to 4-dim by prepending an 'empty' dimension.
            # This is because we are only feeding a single image, but the
            # VGG16 model was built to take multiple images as input.
            image = np.expand_dims(image, axis=0)

            if False:
                # In the original code using this VGG16 model, the random values
                # for the dropout are fixed to 1.0.
                # Experiments suggest that it does not seem to matter for
                # Style Transfer, and this causes an error with a GPU.
                dropout_fix = 1.0

                # Create feed-dict for inputting data to TensorFlow.
                feed_dict = {tensor_name_input_image: image,
                             tensor_name_dropout: [[dropout_fix]],
                             tensor_name_dropout1: [[dropout_fix]]}
            else:
                # Create feed-dict for inputting data to TensorFlow.
                feed_dict = {tensor_name_input_image: image}

            return feed_dict

        image=np.array(PIL.Image.open('images/willy_wonka_old.jpg').resize((224,224)))
        # image = np.expand_dims(image, axis=0)

        # feed_dict = {tensor_name_input_image_: image,tensor_name_dropout_:[[1.0]],tensor_name_dropout1_:[[1.0]]}
        feed_dict=create_feed_dict(image)
        img_out = sess.run(prob_, feed_dict=feed_dict)
        print(np.argmax(img_out))
