参考：https://github.com/ritchieng/wideresnet-tensorlayer
[toc]

# cifar_wide_resnet_tl.py

```python
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import numpy as np


class CNNEnv:
    def __init__(self):

        # The data, shuffled and split between train and test sets
        self.x_train, self.y_train, self.x_test, self.y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)

        # Reorder dimensions for tensorflow
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)

        # For generator
        self.num_examples = self.x_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # For wide resnets
        self.blocks_per_group = 4
        self.widening_factor = 4

        # Basic info
        self.batch_num = 64
        self.img_row = 32
        self.img_col = 32
        self.img_channels = 3
        self.nb_classes = 10

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_size = batch_size

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]

    def reset(self, first):
        self.first = first
        if self.first is True:
            self.sess.close()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=config)

    def step(self):

        def zero_pad_channels(x, pad=0):
            """
            Function for Lambda layer
            """
            pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
            return tf.pad(x, pattern)

        def residual_block(x, count, nb_filters=16, subsample_factor=1):
            prev_nb_channels = x.outputs.get_shape().as_list()[3]

            if subsample_factor > 1:
                subsample = [1, subsample_factor, subsample_factor, 1]
                # shortcut: subsample + zero-pad channel dim
                name_pool = 'pool_layer' + str(count)
                shortcut = tl.layers.PoolLayer(x,
                                               ksize=subsample,
                                               strides=subsample,
                                               padding='VALID',
                                               pool=tf.nn.avg_pool,
                                               name=name_pool)

            else:
                subsample = [1, 1, 1, 1]
                # shortcut: identity
                shortcut = x

            if nb_filters > prev_nb_channels:
                name_lambda = 'lambda_layer' + str(count)
                shortcut = tl.layers.LambdaLayer(
                    shortcut,
                    zero_pad_channels,
                    fn_args={'pad': nb_filters - prev_nb_channels},
                    name=name_lambda)

            name_norm = 'norm' + str(count)
            y = tl.layers.BatchNormLayer(x,
                                         decay=0.999,
                                         epsilon=1e-05,
                                         is_train=True,
                                         name=name_norm)

            name_conv = 'conv_layer' + str(count)
            y = tl.layers.Conv2dLayer(y,
                                      act=tf.nn.relu,
                                      shape=[3, 3, prev_nb_channels, nb_filters],
                                      strides=subsample,
                                      padding='SAME',
                                      name=name_conv)

            name_norm_2 = 'norm_second' + str(count)
            y = tl.layers.BatchNormLayer(y,
                                         decay=0.999,
                                         epsilon=1e-05,
                                         is_train=True,
                                         name=name_norm_2)

            prev_input_channels = y.outputs.get_shape().as_list()[3]
            name_conv_2 = 'conv_layer_second' + str(count)
            y = tl.layers.Conv2dLayer(y,
                                      act=tf.nn.relu,
                                      shape=[3, 3, prev_input_channels, nb_filters],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      name=name_conv_2)

            name_merge = 'merge' + str(count)
            out = tl.layers.ElementwiseLayer([y, shortcut],
                                             combine_fn=tf.add,
                                             name=name_merge)


            return out

        # Placeholders
        learning_rate = tf.placeholder(tf.float32)
        img = tf.placeholder(tf.float32, shape=[self.batch_num, 32, 32, 3])
        labels = tf.placeholder(tf.int32, shape=[self.batch_num, ])

        x = tl.layers.InputLayer(img, name='input_layer')
        x = tl.layers.Conv2dLayer(x,
                                  act=tf.nn.relu,
                                  shape=[3, 3, 3, 16],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='cnn_layer_first')

        for i in range(0, self.blocks_per_group):
            nb_filters = 16 * self.widening_factor
            count = i
            x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=1)

        for i in range(0, self.blocks_per_group):
            nb_filters = 32 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            count = i + self.blocks_per_group
            x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

        for i in range(0, self.blocks_per_group):
            nb_filters = 64 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            count = i + 2*self.blocks_per_group
            x = residual_block(x, count, nb_filters=nb_filters, subsample_factor=subsample_factor)

        x = tl.layers.BatchNormLayer(x,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name='norm_last')

        x = tl.layers.PoolLayer(x,
                                ksize=[1, 8, 8, 1],
                                strides=[1, 8, 8, 1],
                                padding='VALID',
                                pool=tf.nn.avg_pool,
                                name='pool_last')

        x = tl.layers.FlattenLayer(x, name='flatten')

        x = tl.layers.DenseLayer(x,
                                 n_units=self.nb_classes,
                                 act=tf.identity,
                                 name='fc')

        output = x.outputs

        ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(output, labels))
        cost = ce

        correct_prediction = tf.equal(tf.cast(tf.argmax(output, 1), tf.int32), labels)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_params = x.all_params
        train_op = tf.train.GradientDescentOptimizer(
            learning_rate, use_locking=False).minimize(cost, var_list=train_params)

        self.sess.run(tf.initialize_all_variables())

        for i in range(10):
            batch = self.next_batch(self.batch_num)
            feed_dict = {img: batch[0], labels: batch[1], learning_rate: 0.01}
            feed_dict.update(x.all_drop)
            _, l, ac = self.sess.run([train_op, cost, acc], feed_dict=feed_dict)
            print('loss', l)
            print('acc', ac)


a = CNNEnv()
a.reset(first=False)
a.step()
```
# cifar_wide_resnet_keras.py

```python
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Lambda, Convolution2D, Convolution1D, AveragePooling2D, BatchNormalization, Dropout, Reshape
from keras.engine import merge, Input, Model
from keras.engine.training import collect_trainable_weights
import numpy as np


class CNNEnv:
    def __init__(self):
        # TF fix
        K.set_learning_phase(0)

        # The data, shuffled and split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        # Reorder dimensions for tensorflow
        self.x_train = np.transpose(self.x_train.astype('float32'), (0, 1, 2, 3))
        self.mean = np.mean(self.x_train, axis=0, keepdims=True)
        self.std = np.std(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = np.transpose(self.x_test.astype('float32'), (0, 1, 2, 3))
        self.x_test = (self.x_test - self.mean) / self.std

        # self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1]*self.x_train.shape[2]*self.x_train.shape[3])
        # self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1]*self.x_test.shape[2]*self.x_test.shape[3])
        print('x_train shape:', self.x_train.shape)
        print('x_test shape:', self.x_test.shape)


        # Convert class vectors to binary class matrices
        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        print('y_train shape:', self.y_train.shape)
        print('y_test shape:', self.y_test.shape)

        # For generator
        self.num_examples = self.x_train.shape[0]
        self.index_in_epoch = 0
        self.epochs_completed = 0

        # For wide resnets
        self.blocks_per_group = 4
        self.widening_factor = 4

        # Basic info
        self.batch_num = 64
        self.img_row = 32
        self.img_col = 32
        self.img_channels = 3
        self.nb_classes = 10

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        self.batch_size = batch_size

        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size

        if self.index_in_epoch > self.num_examples:
            # Finished epoch
            self.epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.x_train = self.x_train[perm]
            self.y_train = self.y_train[perm]

            # Start next epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.num_examples
        end = self.index_in_epoch
        return self.x_train[start:end], self.y_train[start:end]


    def step(self):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K.set_session(sess)

        def zero_pad_channels(x, pad=0):
            """
            Function for Lambda layer
            """
            pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
            return tf.pad(x, pattern)

        def residual_block(x, nb_filters=16, subsample_factor=1):
            prev_nb_channels = K.int_shape(x)[3]

            if subsample_factor > 1:
                subsample = (subsample_factor, subsample_factor)
                # shortcut: subsample + zero-pad channel dim
                shortcut = AveragePooling2D(pool_size=subsample)(x)
            else:
                subsample = (1, 1)
                # shortcut: identity
                shortcut = x

            if nb_filters > prev_nb_channels:
                shortcut = Lambda(zero_pad_channels,
                                  arguments={
                                      'pad': nb_filters - prev_nb_channels})(shortcut)

            y = BatchNormalization(axis=3)(x)
            y = Activation('relu')(y)
            y = Convolution2D(nb_filters, 3, 3, subsample=subsample,
                              init='he_normal', border_mode='same')(y)
            y = BatchNormalization(axis=3)(y)
            y = Activation('relu')(y)
            y = Convolution2D(nb_filters, 3, 3, subsample=(1, 1),
                              init='he_normal', border_mode='same')(y)

            out = merge([y, shortcut], mode='sum')

            return out


        # this placeholder will contain our input digits
        img = tf.placeholder(tf.float32, shape=(None, self.img_col, self.img_row, self.img_channels))
        labels = tf.placeholder(tf.float32, shape=(None, self.nb_classes))
        # img = K.placeholder(ndim=4)
        # labels = K.placeholder(ndim=1)

        # Keras layers can be called on TensorFlow tensors:
        x = Convolution2D(16, 3, 3, init='he_normal', border_mode='same')(img)

        for i in range(0, self.blocks_per_group):
            nb_filters = 16 * self.widening_factor
            x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

        for i in range(0, self.blocks_per_group):
            nb_filters = 32 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

        for i in range(0, self.blocks_per_group):
            nb_filters = 64 * self.widening_factor
            if i == 0:
                subsample_factor = 2
            else:
                subsample_factor = 1
            x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid')(x)
        x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

        # Readout layer
        preds = Dense(self.nb_classes, activation='softmax')(x)

        loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        '''
        with sess.as_default():
            for i in range(10):
                batch = self.next_batch(self.batch_num)
                _, l = sess.run([optimizer, loss],
                                feed_dict={img: batch[0], labels: batch[1]})
                print(l)
        '''

        with sess.as_default():
            batch = self.next_batch(self.batch_num)
            _, l = sess.run([optimizer, loss],
                            feed_dict={img: batch[0], labels: batch[1]})
            print('Loss', l)

        acc_value = accuracy(labels, preds)

        '''
        with sess.as_default():
            acc = acc_value.eval(feed_dict={img: self.x_test, labels: self.y_test})
            print(acc)
        '''

a = CNNEnv()
a.step()
```
