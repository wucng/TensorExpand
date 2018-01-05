#! /usr/bin/python
# -*- coding: utf-8 -*-



import tensorflow as tf
import tensorlayer as tl

sess = tf.InteractiveSession()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
                                tl.files.load_mnist_dataset(shape=(-1,784))
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# define the network
network=tf.layers.dropout(x,0.8,name='drop1')
network=tf.layers.dense(network,800,tf.nn.relu, name='relu1')
network=tf.layers.dropout(network,0.5,name='drop2')
network=tf.layers.dense(network,800,tf.nn.relu, name='relu2')
network=tf.layers.dropout(network,0.5,name='drop3')
'''
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, 800, tf.nn.relu, name='relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
'''
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
# network = tl.layers.DenseLayer(network, n_units=10,
#                                 act=tf.identity, name='output')

network=tf.layers.dense(network,10,tf.identity, name='output') # Tensor

network = tl.layers.InputLayer(network, name='network') # Tensor-->TL
# define cost function and metric.
y = network.outputs # TL -->Tensor
# y=network
# cost = tl.cost.cross_entropy(y, y_, name='cost')
cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=y))
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
# train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001
                    ).minimize(cost)#, var_list=train_params)

# initialize all variables in the session
# tl.layers.initialize_global_variables(sess)
tf.global_variables_initializer().run()
# print network information
# network.print_params()
# network.print_layers()

# train the network
'''
feed={x:X_train,y_:y_train}
for step in range(1000):
    train_op.run(feed)
    # sess.run(train_op,feed)
    if step%50==0:
        acc_,cost_=sess.run([acc,cost],feed)
        print('step:',step,'|','acc:',acc_,'|','cost:',cost_,'|')

'''
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
            acc=acc, batch_size=500, n_epoch=5, print_freq=5,
            X_val=X_val, y_val=y_val, eval_train=False)

# evaluation
tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)
# '''
# save the network to .npz file
# tl.files.save_npz(network.all_params , name='model.npz')
sess.close()
