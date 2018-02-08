# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

train=False # 训练还是推理
Confused=True # 是否混淆 最后的分类结果 如：加入噪声让7 识别成1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

with tf.variable_scope('D'):
    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # Construct model
    y_pred = tf.nn.softmax(tf.matmul(x, W)+b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred), reduction_indices=1))

train_op=tf.train.AdamOptimizer().minimize(cost)

init = tf.global_variables_initializer()

saver=tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(init)

    if train:
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                # Fit training using batch data
                # _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

                _,c=sess.run([train_op,cost],feed_dict={x: batch_xs, y: batch_ys})

                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        saver.save(sess,'./models/model.ckpt')

        # test
        acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1)),tf.float32))
        print('test acc',acc.eval({x: mnist.test.images, y: mnist.test.labels}))

        print("Optimization Finished!")
    else:
        saver.restore(sess,'./models/model.ckpt')


        input_image=mnist.test.images[:1]
        # print(input_image.shape)
        real_label=mnist.test.labels[:1]

        if not Confused:
            pred_label=y_pred.eval({x:input_image})

            print('pred',np.argmax(pred_label,1),'\n','real',np.argmax(real_label,1))
            # pred [7]
            # real [7]
        else:

            with tf.get_default_graph().as_default():
                # Add a placeholder variable for the target class-number.
                # This will be set to e.g. 300 for the 'bookcase' class.
                pl_cls_target = tf.placeholder(dtype=tf.int32)

                # Add a new loss-function. This is the cross-entropy.
                # See Tutorial #01 for an explanation of cross-entropy.
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=[pl_cls_target])

                # Get the gradient for the loss-function with regard to
                # the resized input image.
                gradient = tf.gradients(loss, x)


            feed_dict = {x: input_image}
            pred, image = sess.run([y_pred, x],
                                   feed_dict=feed_dict)

            # image = sess.run(x,feed_dict=feed_dict)
            # Convert to one-dimensional array.
            pred = np.squeeze(pred)

            # Predicted class-number.
            cls_source = np.argmax(pred)


            noise=0

            for i in range(100):
                noisy_image = image + noise
                noisy_image = np.clip(a=noisy_image, a_min=0.0, a_max=1.0)

                feed_dict = {x: noisy_image,
                             pl_cls_target: 1} # 分类成1

                # Calculate the predicted class-scores as well as the gradient.
                pred, grad = sess.run([y_pred, gradient],
                                      feed_dict=feed_dict)

                pred = np.squeeze(pred)

                grad = np.array(grad).squeeze()

                grad_absmax = np.abs(grad).max()

                if grad_absmax < 1e-10:
                    grad_absmax = 1e-10
                step_size = 7 / grad_absmax

                # Print statistics for the gradient.
                # msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
                # print(msg.format(grad.min(), grad.max(), step_size))


                score_target = pred[1] # 分成1的概率值
                if score_target < 0.99:
                    noise -= step_size * grad

                    noise = np.clip(a=noise,
                                    a_min=-3.0,
                                    a_max=3.0)
                else:
                    # Abort the optimization because the score is high enough.
                    break
            print(pred)
            # 输出混淆后的分类
            pred_label = y_pred.eval({x: noisy_image})

            print('pred', np.argmax(pred_label, 1), '\n', 'real', np.argmax(real_label, 1))
