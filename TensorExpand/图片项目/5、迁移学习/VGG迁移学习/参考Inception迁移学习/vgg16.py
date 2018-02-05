########################################################################
#
# The pre-trained VGG16 Model for TensorFlow.
#
# This model seems to produce better-looking images in Style Transfer
# than the Inception 5h model that otherwise works well for DeepDream.
#
# See the Python Notebook for Tutorial #15 for an example usage.
#
# Implemented in Python 3.5 with TensorFlow v0.11.0rc0
#
########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################
'''
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/vgg16.py
'''
import numpy as np
import tensorflow as tf
import download # pip install download
import os
import sys

########################################################################
# Various directories and file-names.

# The pre-trained VGG16 model is taken from this tutorial:
# https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py

# The class-names are available in the following URL:
# https://s3.amazonaws.com/cadl/models/synset.txt

# Internet URL for the file with the VGG16 model.
# Note that this might change in the future and will need to be updated.
data_url = "https://s3.amazonaws.com/cadl/models/vgg16.tfmodel"

# Directory to store the downloaded data.
data_dir = "vgg16/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "vgg16.tfmodel"

########################################################################


def maybe_download():
    """
    Download the VGG16 model from the internet if it does not already
    exist in the data_dir. WARNING! The file is about 550 MB.
    """

    print("Downloading VGG16 Model ...")

    # The file on the internet is not stored in a compressed format.
    # This function should not extract the file when it does not have
    # a relevant filename-extensions such as .zip or .tar.gz

    # download.maybe_download_and_extract(url=data_url, download_dir=data_dir)
    download.download(url=data_url,path=data_dir)


########################################################################


class VGG16:
    """
    The VGG16 model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.
    When you create a new instance of this class, the VGG16 model
    will be loaded and can be used immediately without training.
    """

    # Name of the tensor for feeding the input image.
    tensor_name_input_image = "images:0"

    # Names of the tensors for the dropout random-values..
    tensor_name_dropout = 'dropout/random_uniform:0'
    tensor_name_dropout1 = 'dropout_1/random_uniform:0'

    # Names for the convolutional layers in the model for use in Style Transfer.
    '''
    layer_names = ['conv1_1/conv1_1', 'conv1_2/conv1_2',
                   'conv2_1/conv2_1', 'conv2_2/conv2_2',
                   'conv3_1/conv3_1', 'conv3_2/conv3_2', 'conv3_3/conv3_3',
                   'conv4_1/conv4_1', 'conv4_2/conv4_2', 'conv4_3/conv4_3',
                   'conv5_1/conv5_1', 'conv5_2/conv5_2', 'conv5_3/conv5_3']
    '''
    tensor_name_transfer_layer = 'pool5:0'

    def __init__(self):
        # Now load the model from file. The way TensorFlow
        # does this is confusing and requires several steps.

        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():

            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

                # Now self.graph holds the VGG16 model from the proto-buf file.

            # Get a reference to the tensor for inputting images to the graph.
            self.y_pred = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get the tensor for the last layer of the graph, aka. the transfer-layer.
            self.transfer_layer = self.graph.get_tensor_by_name(self.tensor_name_transfer_layer)

            # Get references to the tensors for the commonly used layers.
            # self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]
            self.session = tf.Session(graph=self.graph)

    def close(self):
        """
        Call this function when you are done using the Inception model.
        It closes the TensorFlow session to release its resources.
        """

        self.session.close()

    def _create_feed_dict(self, image):
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
        image = np.expand_dims(image, axis=0)

        if False:
            # In the original code using this VGG16 model, the random values
            # for the dropout are fixed to 1.0.
            # Experiments suggest that it does not seem to matter for
            # Style Transfer, and this causes an error with a GPU.
            dropout_fix = 1.0

            # Create feed-dict for inputting data to TensorFlow.
            feed_dict = {self.tensor_name_input_image: image,
                         self.tensor_name_dropout: [[dropout_fix]],
                         self.tensor_name_dropout1: [[dropout_fix]]}
        else:
            # Create feed-dict for inputting data to TensorFlow.
            feed_dict = {self.tensor_name_input_image: image}

        return feed_dict

    def transfer_values(self, image=None):
        """
        Calculate the transfer-values for the given image.
        These are the values of the last layer of the Inception model before
        the softmax-layer, when inputting the image to the Inception model.
        The transfer-values allow us to use the Inception model in so-called
        Transfer Learning for other data-sets and different classifications.
        It may take several hours or more to calculate the transfer-values
        for all images in a data-set. It is therefore useful to cache the
        results using the function transfer_values_cache() below.
        :param image_path:
            The input image is a jpeg-file with this file-path.
        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).
        :return:
            The transfer-values for those images.
        """

        # Create a feed-dict for the TensorFlow graph with the input image.
        feed_dict = self._create_feed_dict(image=image)

        # Use TensorFlow to run the graph for the Inception model.
        # This calculates the values for the last layer of the Inception model
        # prior to the softmax-classification, which we call transfer-values.
        transfer_values = self.session.run(self.transfer_layer, feed_dict=feed_dict)

        # Reduce to a 1-dim array.
        transfer_values = np.squeeze(transfer_values) # [-1,7,7,512]

        return transfer_values

def process_images(fn, images=None):
    """
    Call the function fn() for each image, e.g. transfer_values() from
    the Inception model above. All the results are concatenated and returned.
    :param fn:
        Function to be called for each image.
    :param images:
        List of images to process.
    :param image_paths:
        List of file-paths for the images to process.
    :return:
        Numpy array with the results.
    """

    # Are we using images or image_paths?
    # using_images = images is not None

    # Number of images.
    num_images = len(images)

    # Pre-allocate list for the results.
    # This holds references to other arrays. Initially the references are None.
    result = [None] * num_images

    # For each input image.
    for i in range(num_images):
        # Status-message. Note the \r which means the line should overwrite itself.
        msg = "\r- Processing image: {0:>6} / {1}".format(i + 1, num_images)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Process the image and store the result for later use.
        result[i] = fn(image=images[i])
    # Print newline.
    print()

    # Convert the result to a numpy array.
    result = np.array(result)

    return result

def feed_data(model,images=None):
    return process_images(fn=model.transfer_values, images=images)

########################################################################

if __name__=='__main__':
    maybe_download()
