# -*- coding: utf8 -*-
'''
wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
'''
import tensorflow as tf
import os

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join('./inception/classify_image_graph_def.pb'), 'rb') as f:
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

# print op shape
def print_ops_shape():
    tensor_name_transfer_layer = "pool_3:0"
    graph = tf.Graph()
    with graph.as_default():
        create_graph()
        with tf.Session() as sess:
            # ops = sess.graph.get_operations()
            # for op in ops:
            #     print(op.name)
            transfer_layer = graph.get_tensor_by_name(tensor_name_transfer_layer)
            print(transfer_layer.shape)

print_ops_shape()
