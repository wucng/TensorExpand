参考：
1、[官方文档教程](https://help.aliyun.com/document_detail/50654.html?spm=5176.doc51800.6.566.MdxCQD)
2、[平台使用教程](https://help.aliyun.com/document_detail/49571.html#TensorFlow)
3、[平台使用教程](https://yq.aliyun.com/articles/72841?utm_campaign=wenzhang&utm_medium=article&utm_source=QQ-qun&2017330&utm_content=m_15442)
4、[平台使用教程](http://www.cnblogs.com/iyulang/p/6648603.html)
5、[开通机器学习PAI流程](https://help.aliyun.com/document_detail/53262.html?spm=5176.doc51800.6.577.Tjv21s)


----------
# mnist数据为例
[官方代码](https://help.aliyun.com/document_detail/50654.html?spm=5176.doc35357.6.566.9RA8OM)传入tfrecord数据

```python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import tensorflow as tf

FLAGS = None

def read_image(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label

def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    capacity = 3 * batch_size
    image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size, capacity=capacity, num_threads=10)
    one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
    return image_batch, one_hot_labels

def main(_):

    train_file_path = os.path.join("./", "train.tfrecords")
    test_file_path = os.path.join("./", "test.tfrecords")
    ckpt_path = os.path.join("./", "model.ckpt")

    train_image_filename_queue = tf.train.string_input_producer(
            [train_file_path])
    train_images, train_labels = read_image_batch(train_image_filename_queue, 100)


    test_image_filename_queue = tf.train.string_input_producer(
            [test_file_path])
    test_images, test_labels = read_image_batch(test_image_filename_queue, 100)

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    x = tf.reshape(train_images, [-1, 784])
    y = tf.matmul(x, W) + b
    y_ = tf.to_float(train_labels)

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    x_test = tf.reshape(test_images, [-1, 784])
    y_pred = tf.matmul(x_test, W) + b
    y_test = tf.to_float(test_labels)

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # start queue runner
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Train and Test 
    for i in range(100):
        sess.run(train_step)
        print("step:", i + 1, "accuracy:", sess.run(accuracy))

    save_path = saver.save(sess, ckpt_path)
    print("Model saved in file: %s" % save_path)

    # stop queue runner
    coord.request_stop()
    coord.join(threads)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
```

自己改进，可以直接读取mnist文件 而非tfrecode文件

```python
# -*- coding: utf-8 -*-

import os
import sys
import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):

    mnist=input_data.read_data_sets(FLAGS.buckets,one_hot=True)

    #模型存储名称
    ckpt_path = os.path.join(FLAGS.checkpointDir, "model.ckpt")

    x=tf.placeholder(tf.float32,[None,784])
    y_=tf.placeholder(tf.float32,[None,10])

    # the Variables we need to train
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.matmul(x, W) + b

    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_size=128
    batchs=mnist.train.num_examples // batch_size


    for epoch in range(2):
        for step in range(batchs):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y_:batch_y})

            if step%50==0:
                acc=accuracy.eval({x:batch_x,y_:batch_y})
                print('epoch',epoch,'|','step',step,'|','acc',acc)

        test_acc = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels})
        print('epoch', epoch, '|', 'test_acc', test_acc)

        save_path = saver.save(sess, ckpt_path,global_step=epoch)
        print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #获得buckets路径
    parser.add_argument('--buckets', type=str, default='',
                        help='input data path')
    #获得checkpoint路径
    parser.add_argument('--checkpointDir', type=str, default='',
                        help='output model path')
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(main=main)
```

1、开通对象[存储OSS](https://oss.console.aliyun.com/bucket/oss-cn-shanghai/tensorflow-keras/object?path=)

Object管理--->新建文件夹
先新建一个mnist文件夹，再在该文件夹下建4个文件夹，分别为：
datas（存放数据）、train_code(存放训练脚本)、predict_code(推理脚本)、check_point(输出文件)

![这里写图片描述](http://img.blog.csdn.net/20180125153711174?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


3、大数据--->机器学习 打开PAI
![这里写图片描述](http://img.blog.csdn.net/20180125153752891?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

[进入机器学习](https://pai.base.shuju.aliyun.com/experiment.htm?etag=oxs-base-biz-dmsdp011192097164.em14&Lang=zh_CN&experimentId=43194)


组件--->深度学习（Beta）
![这里写图片描述](http://img.blog.csdn.net/20180125153904485?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

双击TensorFlow 进行参数配置

![这里写图片描述](http://img.blog.csdn.net/20180125153928343?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

执行最下面的运行，右键TensorFlow查看日志，查看运行信息
![这里写图片描述](http://img.blog.csdn.net/20180125153951585?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180125154008349?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

如果 train与predict分开的情况，还需加一个TensorFlow做预测
![这里写图片描述](http://img.blog.csdn.net/20180125154037033?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

预测的TensorFlow参数配置，
![这里写图片描述](http://img.blog.csdn.net/20180125154055455?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Tensorflow_cifar10案例
[文档参考](https://help.aliyun.com/document_detail/51800.html?spm=5176.doc50654.6.567.RLfzXV)

[阿里云机器学习社区](https://yq.aliyun.com/teams/47/type_blog?spm=5176.100244.0.0.oY5yVD)
