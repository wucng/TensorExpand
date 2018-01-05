#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""说明
model 万能模板
数据：mnist
模型建立 Model  只使用线性模型
数据的输入 Inputs
模型保存与提取 Save_and_load_mode
模型可视化 TensorBoard
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import argparse
import sys

class Model(object):
    def __init__(self,X,Y,w,b,learning_rate):
        self.X=X
        self.Y=Y
        self.w=w
        self.b=b
        self.learning_rate=learning_rate

    def inference(self,activation='softmax'):
        if activation=='softmax':
            pred=tf.nn.softmax(tf.matmul(self.X, self.w) + self.b)
        else:
            pred=tf.nn.bias_add(tf.matmul(self.X, self.w),self.b)
        return pred

    def loss(self,pred_value,MSE_error=False,one_hot=True):
        if MSE_error:return tf.reduce_mean(tf.reduce_sum(
            tf.square(pred_value-self.Y),reduction_indices=[1]))
        else:
            if one_hot:
                return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.Y, logits=pred_value))
            else:
                return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.cast(self.Y, tf.int32), logits=pred_value))

    def evaluate(self,pred_value,one_hot=True):
        if one_hot:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.argmax(self.Y, 1))
            # correct_prediction = tf.nn.in_top_k(pred_value, Y, 1)
        else:
            correct_prediction = tf.equal(tf.argmax(pred_value, 1), tf.cast(self.Y, tf.int64))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self,cross_entropy):
        global_step = tf.Variable(0, trainable=False)
        return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy, global_step=global_step)

class Inputs(object):
    def __init__(self,file_path,batch_size,one_hot=True):
        self.file_path=file_path
        self.batch_size=batch_size
        self.mnist=input_data.read_data_sets(self.file_path, one_hot=one_hot)
    def inputs(self):
        batch_xs, batch_ys = self.mnist.train.next_batch(self.batch_size)
        return batch_xs, batch_ys
    def test_inputs(self):
        return self.mnist.test.images,self.mnist.test.labels

class Save_and_load_mode(object):
    def __init__(self,logdir,sess):
        self.saver = tf.train.Saver()
        self.logdir=logdir # 保存模型位置
        self.sess=sess

    def save_model(self,step):
        if not os.path.exists(self.logdir):os.makedirs(self.logdir)
        self.saver.save(self.sess, os.path.join(self.logdir,'model.ckpt'), global_step=step)

    def load_model(self):
        # 验证之前是否已经保存了检查点文件
        ckpt = tf.train.get_checkpoint_state(self.logdir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False

class TensorBoard(object):
    def __init__(self):
        pass

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

    def image_summary(self,name,tensor,max_outputs=10):
        tf.summary.image(name, tensor, max_outputs)

    def hist_summary(self,name,values):
        tf.summary.histogram(name, values)

    def scalar_summary(self,name,tensor):
        tf.summary.scalar(name, tensor)

    def merge_all_summary(self):
        return tf.summary.merge_all()

    def FileWriter_summary(self,log_dir,graph=None):
        return tf.summary.FileWriter(log_dir,graph)


FLAGS = None
def train():
    tb_model=TensorBoard()

    # Input layer
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28*28*1],'x')
        y_ = tf.placeholder(tf.float32, [None,10],'y_')
    with tf.name_scope('input_reshape'):
         image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
         tb_model.image_summary('input', image_shaped_input, 10)


    # Output layer
    with tf.name_scope('line_layer'):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal([28*28*1, 10])) # 二分类
            tb_model.variable_summaries(w)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.random_normal([10]))
            tb_model.variable_summaries(b)

    input_model=Inputs(FLAGS.data_dir,FLAGS.batch_size,one_hot=FLAGS.one_hot)

    model=Model(x,y_,w,b,FLAGS.learning_rate)
    with tf.name_scope('Wx_plus_b'):
        y=model.inference(activation='softmax')
        tb_model.hist_summary('pred',y)

    with tf.name_scope('total_loss'):
        cross_entropy=model.loss(y,MSE_error=False,one_hot=FLAGS.one_hot)
        tb_model.scalar_summary('cross_entropy', cross_entropy)

    train_op=model.train(cross_entropy)

    with tf.name_scope('accuracy'):
        accuracy=model.evaluate(y,one_hot=FLAGS.one_hot)
        tb_model.scalar_summary('accuracy', accuracy)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


    with tf.Session() as sess:
        # Merge all the summaries and write them out to /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tb_model.merge_all_summary()

        if not os.path.exists(os.path.join(FLAGS.log_dir + '/train')): os.makedirs(os.path.join(FLAGS.log_dir + '/train'))
        if not os.path.exists(os.path.join(FLAGS.log_dir + '/test')): os.makedirs(os.path.join(FLAGS.log_dir + '/test'))
        train_writer = tb_model.FileWriter_summary(os.path.join(FLAGS.log_dir + '/train'),sess.graph)
        test_writer = tb_model.FileWriter_summary(os.path.join(FLAGS.log_dir + '/test'))


        save=Save_and_load_mode(FLAGS.log_dir,sess)
        if not save.load_model():init.run()
        for step in range(FLAGS.num_steps):
            batch_xs, batch_ys = input_model.inputs()
            train_op.run({x: batch_xs, y_: batch_ys})

            if step % FLAGS.disp_step == 0:
                acc=accuracy.eval({x: batch_xs, y_: batch_ys})
                print("step", step, 'acc', acc,
                      'loss', cross_entropy.eval({x: batch_xs, y_: batch_ys}))
                train_result = merged.eval({x: batch_xs, y_: batch_ys})
                train_writer.add_summary(train_result, step)


                test_x, test_y = input_model.test_inputs()
                acc = accuracy.eval({x: test_x, y_: test_y})
                print("step", step, 'acc', acc)
                test_result = merged.eval({x: test_x, y_: test_y})
                test_writer.add_summary(test_result, step)

            save.save_model(step)
        """
        # test acc
        test_x,test_y=input_model.test_inputs()
        print('test acc', acc)
        """

def main(_):
    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__=="__main__":
    # 设置必要参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1000,
                        help = 'Number of steps to run trainer.')
    parser.add_argument('--disp_step', type=int, default=100,
                        help='Number of steps to display.')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='Learning rate.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of mini training samples.')
    parser.add_argument('--one_hot', type=bool, default=True,
                        help='One-Hot Encoding.')
    parser.add_argument('--data_dir', type=str, default='./MNIST_data/',
            help = 'Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./log_dir',
                        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

# 启动TensorBoard: tensorboard --logdir=path/to/log-directory
# tensorboard --logdir='log_dir'

