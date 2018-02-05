#! /usr/bin/python
# -*- coding: utf8 -*-

'''
参考：http://blog.csdn.net/wc781708249/article/details/78043099

wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

从pool_3开始传入自己的数据，训练后面的参数，实现模型迁移

'''
import numpy as np
import tensorflow as tf
import download # pip install download
import os
import PIL.Image
import glob
import sys

path_graph_def = "./inception/classify_image_graph_def.pb"

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    # 导入pb文件
    with open(path_graph_def, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:

        # Name of the tensor for feeding the input image.
        tensor_name_input_image = "DecodeJpeg:0"

        # Names of the tensors for the dropout random-values..
        tensor_name_transfer_layer = "pool_3:0"

        # tensor_name_input_image_ = sess.graph.get_tensor_by_name(tensor_name_input_image)
        # tensor_name_dropout_=sess.graph.get_tensor_by_name(tensor_name_dropout)
        # tensor_name_dropout1_ = sess.graph.get_tensor_by_name(tensor_name_dropout1)
        # conv5_ = sess.graph.get_tensor_by_name(conv5)
        # fc6_ = sess.graph.get_tensor_by_name(fc6)
        # fc7_ = sess.graph.get_tensor_by_name(fc7)
        # fc8_ = sess.graph.get_tensor_by_name(fc8) # (?, 1000)

        input2x=sess.graph.get_tensor_by_name(tensor_name_transfer_layer) # (?, 2048)

        # prob_=sess.graph.get_tensor_by_name(prob)
        # print(tensor_name_input_image_.shape,'\n',tensor_name_dropout_.shape,'\n',tensor_name_dropout1_.shape,
        #       '\n',conv5_.shape,'\n',fc6_.shape,'\n',fc7_.shape,'\n',fc8_.shape,'\n',prob_.shape)

        # exit(-1)

        # 修改输出层 传入自己的数据 训练
        x=tf.placeholder(tf.float32,[None,2048])
        y_true = tf.placeholder(tf.float32, shape=[None, 2], name='y_true')

        y_true_cls = tf.argmax(y_true, dimension=1)

        with tf.variable_scope('D'):
            fc = tf.layers.dense(x, 1024, activation=tf.nn.relu, name='layer_fc1')
            y_pred = tf.layers.dense(fc, 2, activation=tf.nn.softmax)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)
        # train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

        y_pred_cls = tf.argmax(y_pred, dimension=1)

        # 创建一个布尔向量，表示每张图像的真实类别是否与预测类别相同。
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_op=tf.train.GradientDescentOptimizer(1e-4).minimize(loss,global_step)
        '''
        trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        gradients_D = trainerD.compute_gradients(loss)

        clipped_gradients_D = [(tf.clip_by_value(_[0], -1, 1), _[1]) for _ in gradients_D]
        train_op = trainerD.apply_gradients(clipped_gradients_D)
        '''
        '''
        tvars = tf.trainable_variables()
        d_params = [v for v in tvars if v.name.startswith('D/')]
        trainerD = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
        d_grads = trainerD.compute_gradients(loss, d_params)#Only update the weights for the discriminator network.
        train_op = trainerD.apply_gradients(d_grads)
        '''

        init = tf.global_variables_initializer()
        # init=tf.initialize_variables()
        sess.run(init)

        batch_size = 32

        # 加载数据
        def load_image(image_path):
            img_path = glob.glob(image_path)  #
            imgs = []
            labels = []
            for path in img_path:
                image = np.array(PIL.Image.open(path).resize((224, 224)))
                imgs.append(image)
                if path.strip().split('/')[-2] == 'ants':
                    labels.append([0, 1])  # 1
                else:
                    labels.append([1, 0])  # 0

            imgs = np.array(imgs)
            labels = np.array(labels)
            return imgs,labels

        imgs, labels=load_image('../../hymenoptera_data/train/*/*.jpg')
        total_batchs = len(imgs) // batch_size

        # --------------------------------

        def _create_feed_dict(image=None):
            """
            Create and return a feed-dict with an image.
            :param image_path:
                The input image is a jpeg-file with this file-path.
            :param image:
                The input image is a 3-dim array which is already decoded.
                The pixels MUST be values between 0 and 255 (float or int).
            :return:
                Dict for feeding to the Inception graph in TensorFlow.
            """
            feed_dict = {tensor_name_input_image: image}

            return feed_dict

        ########################################################################
        # Batch-processing.

        def transfer_values(session=sess, input2x=input2x, image=None):
            """
            Calculate the transfer-values for the given image.
            These are the values of the last layer of the Inception model before
            the softmax-layer, when inputting the image to the Inception model.
            The transfer-values allow us to use the Inception model in so-called
            Transfer Learning for other data-sets and different classifications.
            It may take several hours or more to calculate the transfer-values
            for all images in a data-set. It is therefore useful to cache the
            results using the function transfer_values_cache() below.
            :param image_path:
                The input image is a jpeg-file with this file-path.
            :param image:
                The input image is a 3-dim array which is already decoded.
                The pixels MUST be values between 0 and 255 (float or int).
            :return:
                The transfer-values for those images.
            """

            # Create a feed-dict for the TensorFlow graph with the input image.
            feed_dict = _create_feed_dict(image=image)

            # Use TensorFlow to run the graph for the Inception model.
            # This calculates the values for the last layer of the Inception model
            # prior to the softmax-classification, which we call transfer-values.
            transfer_values = session.run(input2x, feed_dict=feed_dict)

            # Reduce to a 1-dim array.
            transfer_values = np.squeeze(transfer_values)  # [-1,7,7,512]

            return transfer_values

        def process_images(images=None):
            """
            Call the function fn() for each image, e.g. transfer_values() from
            the Inception model above. All the results are concatenated and returned.
            :param fn:
                Function to be called for each image.
            :param images:
                List of images to process.
            :param image_paths:
                List of file-paths for the images to process.
            :return:
                Numpy array with the results.
            """
            # fn=transfer_values(images=images)
            num_images = len(images)

            # Pre-allocate list for the results.
            # This holds references to other arrays. Initially the references are None.
            result = [None] * num_images

            # For each input image.
            for i in range(num_images):
                # Status-message. Note the \r which means the line should overwrite itself.
                msg = "\r- Processing image: {0:>6} / {1}".format(i + 1, num_images)

                # Print the status message.
                sys.stdout.write(msg)
                sys.stdout.flush()

                # Process the image and store the result for later use.

                result[i] = transfer_values(image=images[i])

            # Print newline.
            print()

            # Convert the result to a numpy array.
            result = np.array(result)

            return result
        # -----------------------------------

        # 开始训练
        for epoch in range(20):
            index = np.arange(0, len(imgs), dtype=np.int32)
            np.random.shuffle(index)
            imgs=imgs[index]
            labels=labels[index]

            for step in range(total_batchs):
                batch_x=imgs[step*batch_size:(step+1)*batch_size]
                batch_y = labels[step * batch_size:(step + 1) * batch_size]

                # input_x=sess.run(input2x,{tensor_name_input_image:batch_x}) # [-1,7,7,512]
                input_x=process_images(images=batch_x)

                sess.run(train_op,{x:input_x,y_true:batch_y})


                if step%10==0:
                    acc=sess.run(accuracy,{x:input_x,y_true:batch_y})
                    print('epoch',epoch,'|','step',step,'|','acc',acc)


        # test
        imgs_test, labels_test = load_image('../../hymenoptera_data/val/*/*.jpg')

        # input_x = sess.run(input2x, {tensor_name_input_image: imgs_test[:10]})
        input_x = process_images(images=imgs_test[:10])
        pred_y=sess.run(y_pred_cls,{x:input_x})

        print('pred:',pred_y,'\n','real:',np.argmax(labels_test[:10],1))
