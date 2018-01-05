#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
tensorflow 读取csv数据
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# tensorflow 读取 csv、tfrecord数据
class tf_read_csv():
    def __init__(self,file_path,data=None):
        # super(Data_save_load2,self).__init__(file_path,data) # 返回Data_save_load的对象
        # 或者 Data_save_load.__init__(self,file_path,data)
        self.file_path=file_path
        self.data=data

    # 读取 iris.csv
    def __read_data(self,file_queue):
        reader = tf.TextLineReader(skip_header_lines=1)  # skip_header_lines=1 跳过一行
        key, value = reader.read(file_queue)
        defaults = [[0], [0.], [0.], [0.], [0.], ['']] # 各列对应的数值类型
        Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species = tf.decode_csv(value, defaults)
        # Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species 对应数据中的标题列
        # 如果不同只需做相应的修改即可
        preprocess_op = tf.case({
            tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
            tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
            tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
        }, lambda: tf.constant(-1), exclusive=True) # 将文本转成对应的数值

        return tf.stack([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]), preprocess_op  # X,y

    def create_pipeline(self,filename, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        example, label = self.__read_data(file_queue)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

        return example_batch, label_batch

    # 读取 iris.data
    def __read_data2(self, file_queue):
        reader = tf.TextLineReader(skip_header_lines=1)  # skip_header_lines=1 跳过一行
        key, value = reader.read(file_queue)
        defaults = [[0.], [0.], [0.], [0.], ['']] # # 各列对应的数值类型
        SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species = tf.decode_csv(value, defaults)
        # Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species 对应数据中的标题列
        # 如果不同只需做相应的修改即可
        preprocess_op = tf.case({
            tf.equal(Species, tf.constant('Iris-setosa')): lambda: tf.constant(0),
            tf.equal(Species, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
            tf.equal(Species, tf.constant('Iris-virginica')): lambda: tf.constant(2),
        }, lambda: tf.constant(-1), exclusive=True)  # 将文本转成对应的数值

        return tf.stack([SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]), preprocess_op  # X,y

    def create_pipeline2(self,filename, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        example, label = self.__read_data2(file_queue)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

        return example_batch, label_batch

    # 读取titanic_dataset.csv
    def __read_data3(self, file_queue):
        reader = tf.TextLineReader(skip_header_lines=1)  # skip_header_lines=1 跳过一行
        key, value = reader.read(file_queue)

        defaults = [[0], [0.], [''], [''], [0.], [0], [0], [''], [0.0]] # 各列对应的数值类型
        survived, pclass, name, sex, age, sibsp, parch, ticket, fare = tf.decode_csv(value, defaults)
        # survived, pclass, name, sex, age, sibsp, parch, ticket, fare 对应数据中的标题列
        # 如果不同只需做相应的修改即可

        gender = tf.case({tf.equal(sex, tf.constant('female')): lambda: tf.constant(1.),
                          tf.equal(sex, tf.constant('male')): lambda: tf.constant(0.),
                          }, lambda: tf.constant(-1.), exclusive=True) # 将文本转成对应的数值
        features = tf.stack([pclass, gender, age])
        return features, survived   # X,y

    def create_pipeline3(self,filename, batch_size, num_epochs=None):
        file_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        example, label = self.__read_data3(file_queue)

        min_after_dequeue = 1000
        capacity = min_after_dequeue + batch_size
        example_batch, label_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue
        )

        return example_batch, label_batch


if __name__=="__main__":
    # d = tf_read_csv('Iris-train.csv')
    # x_train_batch, y_train_batch = d.create_pipeline(d.file_path, 50, num_epochs=1000)

    # d = tf_read_csv('iris.data')
    # x_train_batch, y_train_batch = d.create_pipeline2(d.file_path, 50, num_epochs=1000)

    d = tf_read_csv('titanic_dataset.csv')
    x_train_batch, y_train_batch = d.create_pipeline3(d.file_path, 50, num_epochs=1000)

    sess = tf.InteractiveSession()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            curr_x_train_batch, curr_y_train_batch = sess.run([x_train_batch, y_train_batch])
            print(curr_y_train_batch)
            exit(0)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
