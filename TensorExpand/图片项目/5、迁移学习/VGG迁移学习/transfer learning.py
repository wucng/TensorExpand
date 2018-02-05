#! /usr/bin/python
# -*- coding: utf8 -*-

'''
参考：http://blog.csdn.net/wc781708249/article/details/78043099

wget https://s3.amazonaws.com/cadl/models/vgg16.tfmodel

从pool5开始传入自己的数据，训练后面的参数，实现模型迁移

'''
import numpy as np
import tensorflow as tf
import download # pip install download
import os
import PIL.Image
import glob

path_graph_def = "./vgg16/vgg16.tfmodel"

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    # 导入pb文件
    with open(path_graph_def, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:

        # Name of the tensor for feeding the input image.
        tensor_name_input_image = "images:0"

        # Names of the tensors for the dropout random-values..
        tensor_name_dropout = 'dropout/random_uniform:0'
        tensor_name_dropout1 = 'dropout_1/random_uniform:0'
        conv5='conv5_3/conv5_3:0'
        pool5='pool5:0'
        # fc6='fc6:0' # (?, 4096)
        # fc7='fc7:0' # (?, 4096)
        # fc8='fc8:0' # (?, 1000)
        # prob = 'prob:0' # (?, 1000)

        # tensor_name_input_image_ = sess.graph.get_tensor_by_name(tensor_name_input_image)
        # tensor_name_dropout_=sess.graph.get_tensor_by_name(tensor_name_dropout)
        # tensor_name_dropout1_ = sess.graph.get_tensor_by_name(tensor_name_dropout1)
        # conv5_ = sess.graph.get_tensor_by_name(conv5)
        # fc6_ = sess.graph.get_tensor_by_name(fc6)
        # fc7_ = sess.graph.get_tensor_by_name(fc7)
        # fc8_ = sess.graph.get_tensor_by_name(fc8) # (?, 1000)

        input2x=sess.graph.get_tensor_by_name(pool5) # (?, 7, 7, 512)

        # prob_=sess.graph.get_tensor_by_name(prob)
        # print(tensor_name_input_image_.shape,'\n',tensor_name_dropout_.shape,'\n',tensor_name_dropout1_.shape,
        #       '\n',conv5_.shape,'\n',fc6_.shape,'\n',fc7_.shape,'\n',fc8_.shape,'\n',prob_.shape)

        # exit(-1)

        # 修改输出层 传入自己的数据 训练
        x=tf.placeholder(tf.float32,[None,7,7,512])
        y_=tf.placeholder(tf.float32,[None,2])

        with tf.variable_scope('D'):
            # input_=tf.reshape(x,[-1,7*7*512])
            conv1=tf.layers.conv2d(x,512,7,padding='SAME',activation=tf.nn.relu)
            pool1=tf.layers.max_pooling2d(conv1,2,2,padding='SAME') # [-1,4,4,512]

            conv2 = tf.layers.conv2d(pool1, 512*2, 5, padding='SAME', activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='SAME')  # [-1,2,2,512*2]

            conv3 = tf.layers.conv2d(pool2, 512*2, 3, padding='SAME', activation=tf.nn.relu)
            pool3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='SAME')  # [-1,1,1,512*2]

            output = tf.layers.dense(tf.reshape(pool3,[-1,512*2]), 2, activation=tf.nn.softmax) # [-1,2]
        loss=tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=output)

        train_op=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
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
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()
        # init=tf.initialize_variables()
        sess.run(init)

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
            if len(image.shape)<4:
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
        '''
        image=np.array(PIL.Image.open('images/willy_wonka_old.jpg').resize((224,224)))
        # image = np.expand_dims(image, axis=0)

        # feed_dict = {tensor_name_input_image_: image,tensor_name_dropout_:[[1.0]],tensor_name_dropout1_:[[1.0]]}
        feed_dict=create_feed_dict(image)
        img_out = sess.run(prob_, feed_dict=feed_dict)
        print(np.argmax(img_out))
        '''

        batch_size = 20

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

        # 开始训练
        for epoch in range(20):
            index = np.arange(0, len(imgs), dtype=np.int32)
            np.random.shuffle(index)
            imgs=imgs[index]
            labels=labels[index]

            for step in range(total_batchs):
                batch_x=imgs[step*batch_size:(step+1)*batch_size]
                batch_y = labels[step * batch_size:(step + 1) * batch_size]
                feed_dict = create_feed_dict(batch_x)
                # feed_dict[y_]=batch_y
                input_x=sess.run(input2x,feed_dict) # [-1,7,7,512]

                sess.run(train_op,{x:input_x,y_:batch_y})


                if step%10==0:
                    acc=sess.run(accuracy,{x:input_x,y_:batch_y})
                    print('epoch',epoch,'|','step',step,'|','acc',acc)


        # test
        imgs_test, labels_test = load_image('../../hymenoptera_data/val/*/*.jpg')

        feed_dict = create_feed_dict(imgs_test[:10])
        input_x = sess.run(input2x, feed_dict)
        pred_y=sess.run(output,{x:input_x})
        print('pred:',np.argmax(pred_y,1),'\n','real:',np.argmax(labels_test[:10],1))
