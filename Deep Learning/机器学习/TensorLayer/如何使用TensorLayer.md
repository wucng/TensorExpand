# How to use TensorLayer

While research in Deep Learning continues to improve the world, we use a bunch of tricks to implement algorithms with TensorLayer day to day.

Here are a summary of the tricks to use TensorLayer, you can also find more tricks in [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#fqa).

If you find a trick that is particularly useful in practice, please open a Pull Request to add it to the document. If we find it to be reasonable and verified, we will merge it in.

## 1. Installation
 * To keep your TL version and edit the source code easily, you can download the whole repository by excuting `git clone https://github.com/zsdonghao/tensorlayer.git` in your terminal, then copy the `tensorlayer` folder into your project 
 * As TL is growing very fast, if you want to use `pip` install, we suggest you to install the master version 
 * For NLP application, you will need to install [NLTK and NLTK data](http://www.nltk.org/install.html)

## 2. Interaction between TF and TL
 * TF to TL : use [InputLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#input-layer)
 * TL to TF : use [network.outputs](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer)
 * Other methods [issues7](https://github.com/zsdonghao/tensorlayer/issues/7), multiple inputs [issues31](https://github.com/zsdonghao/tensorlayer/issues/31)

## 3. Training/Testing switching
 * Use [network.all_drop](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer) to control the training/testing phase (for [DropoutLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#dropout-layer) only) see [tutorial_mnist.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist.py) and [Understand Basic layer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer)
 * Alternatively, set `is_fix` to `True` in [DropoutLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#dropout-layer), and build different graphs for training/testing by reusing the parameters. You can also set different `batch_size` and noise probability for different graphs. This method is the best when you use [GaussianNoiseLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#gaussian-noise-layer), [BatchNormLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#batch-normalization) and etc. Here is an example:
```python
def mlp(x, is_train=True, reuse=False):
    with tf.variable_scope("MLP", reuse=reuse):
      tl.layers.set_name_reuse(reuse)
      net = InputLayer(x, name='in')
      net = DropoutLayer(net, 0.8, True, is_train, name='drop1')
      net = DenseLayer(net, n_units=800, act=tf.nn.relu, name='dense1')
      net = DropoutLayer(net, 0.8, True, is_train, name='drop2')
      net = DenseLayer(net, n_units=800, act=tf.nn.relu, name='dense2')
      net = DropoutLayer(net, 0.8, True, is_train, name='drop3')
      net = DenseLayer(net, n_units=10, act=tf.identity, name='out')
      logits = net.outputs
      net.outputs = tf.nn.sigmoid(net.outputs)
      return net, logits
x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
net_train, logits = mlp(x, is_train=True, reuse=False)
net_test, _ = mlp(x, is_train=False, reuse=True)
cost = tl.cost.cross_entropy(logits, y_, name='cost')
```


## 4. Get variables and outputs
 * Use [tl.layers.get_variables_with_name](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#get-variables-with-name) instead of using [net.all_params](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#understand-basic-layer)
```python
train_vars = tl.layers.get_variables_with_name('MLP', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_vars)
```
 * This method can also be used to freeze some layers during training, just simply don't get some variables
 * Other methods [issues17](https://github.com/zsdonghao/tensorlayer/issues/17), [issues26](https://github.com/zsdonghao/tensorlayer/issues/26), [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#exclude-some-layers-from-training)

 * Use [tl.layers.get_layers_with_name](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#get-layers-with-name) to get list of activation outputs from a network.
```python
layers = tl.layers.get_layers_with_name(network, "MLP", True)
```
 * This method usually be used for activation regularization.
  
## 5. Pre-trained CNN and Resnet
* Pre-trained CNN
 * Many applications make need pre-trained CNN model
 * TL examples provide pre-trained VGG16, VGG19, Inception and etc : [TL/example](https://github.com/zsdonghao/tensorlayer/tree/master/example)
 * [tl.layers.SlimNetsLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-tf-slim) allows you to use all [Tf-Slim pre-trained models](https://github.com/tensorflow/models/tree/master/slim)
* Resnet
 * Implemented by "for" loop [issues85](https://github.com/zsdonghao/tensorlayer/issues/85)
 * Other methods [by @ritchieng](https://github.com/ritchieng/wideresnet-tensorlayer)

## 6. Data augmentation
* Use TFRecord, see [cifar10 and tfrecord examples](https://github.com/zsdonghao/tensorlayer/tree/master/example); good wrapper: [imageflow](https://github.com/HamedMP/ImageFlow)
* Use python-threading with [tl.prepro.threading_data](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#threading) and [the functions for images augmentation](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#images) see [tutorial_image_preprocess.py](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_image_preprocess.py)

## 7. Batch of data
* If your data size is small enough to feed into the memory of your machine.
 * Use [tl.iterate.minibatches](http://tensorlayer.readthedocs.io/en/latest/modules/iterate.html#tensorlayer.iterate.minibatches) to shuffle and return the examples and labels by the given batchsize.
 * For time-series data, use [tl.iterate.seq_minibatches, tl.iterate.seq_minibatches2, tl.iterate.ptb_iterator and etc](http://tensorlayer.readthedocs.io/en/latest/modules/iterate.html#time-series)
* If your data size is very large
 * Use [tl.prepro.threading_data](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#tensorlayer.prepro.threading_data) to read a batch of data at the beginning of every step
 * Use TFRecord again, see [cifar10 and tfrecord examples](https://github.com/zsdonghao/tensorlayer/tree/master/example)

## 8. Customized layer
* 1. [Write a TL layer directly](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#your-layer)
* 2. Use [LambdaLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#lambda-layer), it can also accept functions with new variables. With this layer you can connect all third party TF libraries and your customized function to TL. Here is an example of using Keras and TL together.

```python
import tensorflow as tf
import tensorlayer as tl
from keras.layers import *
from tensorlayer.layers import *
def my_fn(x):
    x = Dropout(0.8)(x)
    x = Dense(800, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(800, activation='relu')(x)
    x = Dropout(0.5)(x)
    logits = Dense(10, activation='linear')(x)
    return logits

network = InputLayer(x, name='input')
network = LambdaLayer(network, my_fn, name='keras')
...
```



## 9. Sentences tokenization
 * Use [tl.nlp.process_sentence](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#process-sentence) to tokenize the sentences, [NLTK and NLTK data](http://www.nltk.org/install.html) is required
 * Then use [tl.nlp.create_vocab](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#create-vocabulary) to create a vocabulary and save as txt file (it will return a [tl.nlp.SimpleVocabulary object](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#simple-vocabulary-class) for word to id only)
 * Finally use [tl.nlp.Vocabulary](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html#vocabulary-class) to create a vocabulary object from the txt vocabulary file created by `tl.nlp.create_vocab`
 * More pre-processing functions for sentences in [tl.prepro](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#sequence) and [tl.nlp](http://tensorlayer.readthedocs.io/en/latest/modules/nlp.html)

## 10. Dynamic RNN and sequence length
 * Use [tl.layers.retrieve_seq_length_op2](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#compute-sequence-length-2) to automatically compute the sequence length from placeholder, and feed it to the `sequence_length` of [DynamicRNNLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#dynamic-rnn-layer)
 * Apply zero padding on a batch of tokenized sentences as follow:
```python
b_sentence_ids = tl.prepro.pad_sequences(b_sentence_ids, padding='post')
```
 * Other methods [issues18](https://github.com/zsdonghao/tensorlayer/issues/18)

## 11. Common problems
 * Matplotlib issue arise when importing TensorLayer [issues](https://github.com/zsdonghao/tensorlayer/issues/79), [FQA](http://tensorlayer.readthedocs.io/en/latest/user/more.html#visualization)
 
## 12. Other tricks
 * Disable console logging: if you are building a very deep network and don't want to view them in the terminal, disable `print` by `with tl.ops.suppress_stdout():`:
```
print("You can see me")
with tl.ops.suppress_stdout():
    print("You can't see me") # build your graphs here
print("You can see me")
```
## 13. Compatibility with other TF wrappers
TL can interact with other TF wrappers, which means if you find some codes or models implemented by other wrappers, you can just use it !
 * Keras to TL: [KerasLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-keras) (if you find some codes implemented by Keras, just use it. example [here](https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_keras.py))
 * TF-Slim to TL: [SlimNetsLayer](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html#connect-tf-slim) (you can use all Google's pre-trained convolutional models with this layer !!!)
 * I think more libraries will be compatible with TL

## 14. Compatibility with different TF versions
 * [RNN cell_fn](http://tensorlayer.readthedocs.io/en/latest/modules/layers.html): use [tf.contrib.rnn.{cell_fn}](https://www.tensorflow.org/api_docs/python/) for TF1.0+, or [tf.nn.rnn_cell.{cell_fn}](https://www.tensorflow.org/versions/r0.11/api_docs/python/) for TF1.0-
 * [cross_entropy](http://tensorlayer.readthedocs.io/en/latest/modules/cost.html): have to give a unique name for TF1.0+
 
## Useful links
 * TL official sites: [Docs](http://tensorlayer.readthedocs.io/en/latest/), [中文文档](http://tensorlayercn.readthedocs.io/zh/latest/), [Github](https://github.com/zsdonghao/tensorlayer)
 * [Learning Deep Learning with TF and TL ](https://github.com/wagamamaz/tensorflow-tutorial)
 * Follow [zsdonghao](https://github.com/zsdonghao) for further examples

## Author
 - Zhang Rui
 - You
