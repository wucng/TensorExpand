# -*- coding:utf-8 -*-

'''
迁移学习：
只更新添加层的参数，冻结原始模型
初始化参数时 只对对添加层的参数初始化

模型微调：
将原来的模型参数作为初始参数，也进行更新
初始化参数时 只对对添加层的参数初始化

# 附加 ckpt模型做迁移思路
saver = tf.train.Saver(max_to_keep=1)  # 最多保留一个版本
var_list = tf.global_variables()
print(var_list)
var_list_1=[]
for var in var_list:  # 不加载 最后两层的参数，即重新训练
    if 'fc1' in var.name  or 'ouput' in var.name:
        # var_list_1.remove(var)
        continue
    var_list_1.append(var)
print(var_list_1)

saver = tf.train.Saver(max_to_keep=1,var_list=var_list_1)

saver.restore(sess,'save_path')
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re
from tensorflow.python.platform import gfile
import os
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

num_class=10
batch_size=64
model='../20170512-110547/20170512-110547.pb' # 存放ckpt的文件夹,或者pb文件

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

with tf.Graph().as_default():
    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU70%的显存

    with tf.Session(config=config) as sess:
        # Load the model
        print('Loading feature extraction model')
        load_model(model)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") # 输入 shape [batch_size,149,149,3]
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") # 输出 shape [batch_size,128]
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0") # is_training
        embedding_size = embeddings.get_shape()[1]


        # 检测结果
        # images=np.random.random([2,149,149,3])
        # feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        # emb_array = sess.run(embeddings, feed_dict=feed_dict)
        # print(emb_array.shape)

        embeddings=tf.stop_gradient(embeddings) # 做迁移 不更新这层以及之前所有层的参数
        # 或者
        # tvars = tf.trainable_variables() # 找到所以可更新的变量
        # tvars.trainable=False  # 将变量的trainable属性设置为False（不更新参数）

        y_true = tf.placeholder(tf.int64, shape=[None,], name='y_true')
        # y_true_cls = tf.argmax(y_true, dimension=1)
        with tf.variable_scope('D'):
            fc=tf.layers.dense(embeddings,48,activation=tf.nn.sigmoid,name='coding_layer')
            y_pred=tf.layers.dense(fc,num_class,activation=tf.nn.softmax,name='output')

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        global_step = tf.Variable(initial_value=0,
                                  name='global_step', trainable=False)

        y_pred_cls = tf.argmax(y_pred, axis=1)
        # 创建一个布尔向量，表示每张图像的真实类别是否与预测类别相同。
        correct_prediction = tf.equal(y_pred_cls, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 只更新coding_layer与output这两层参数，其他的参数不更新
        tvars = tf.trainable_variables() # 获取所有可以更新的变量
        d_params = [v for v in tvars if v.name.startswith('D/')]

        # 注 默认是更新所以参数 var_list=None
        train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss, global_step,var_list=d_params)

        # init = tf.global_variables_initializer() # 默认初始化所以变量参数
        # 只初始化coding_layer与output这两层参数, global_step 也需初始化
        init=tf.variables_initializer(var_list=d_params+tf.global_variables(scope='global_step'))
        sess.run(init)

        # 加载数据
        mnist=read_data_sets('./MNIST_data',one_hot=False)

        def change_data(batch_x):
            batch_x = np.reshape(batch_x, [-1, 28, 28, 1])
            batch_x = np.concatenate((batch_x, batch_x, batch_x), axis=-1)  # [-1,28,28,3]
            # 转成[-1,149,149,3]
            batch_x=tf.image.resize_image_with_crop_or_pad(batch_x, 149, 149)
            imgs=sess.run(batch_x)
            return np.asarray(imgs,np.float32)

        # 保存所有变量 var_list指定要保持或者是提取的变量，默认是所有变量
        saver=tf.train.Saver(var_list=tf.global_variables()) # var_list=None 也是默认保持所以变量

        for step in range(1000):
            batch_x,batch_y=mnist.train.next_batch(batch_size)
            batch_x=change_data(batch_x)
            batch_y=batch_y.astype(np.int64)
            # print(batch_x.shape,batch_y.shape)
            # exit(-1)

            # feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False}
            feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False, y_true: batch_y}

            _,_=sess.run([train_op,global_step],feed_dict)

            if step % 50==0:
                feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False,y_true:batch_y}
                acc,cost=sess.run([accuracy,loss],feed_dict)
                print('step',step,'|','acc',acc,'|','loss',cost)

        saver.save(sess,'./model/model.ckpt')
