# -*- coding: utf8 -*-
'''
wget https://s3.amazonaws.com/cadl/models/vgg16.tfmodel
'''
import tensorflow as tf
import os

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join('./vgg16/vgg16.tfmodel'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# print all op names
def print_ops():
    create_graph()
    with tf.Session() as sess:
        ops = sess.graph.get_operations()
        for op in ops:
            print(op.name)

print_ops()
