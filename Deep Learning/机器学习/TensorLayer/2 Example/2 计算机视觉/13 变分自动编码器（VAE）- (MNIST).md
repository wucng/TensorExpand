https://github.com/BUPTLdy/tl-vae


```python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:51:20 2017
@author: ldy
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
from scipy.misc import imsave as ims
from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img
    
class vae():
    def __init__(self, dataset_name, n_z, conditional_vae=False):
        if dataset_name == 'mnist':
            self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
            self.n_samples = self.mnist.train.num_examples

        else:
            print 'Load your dataset.'
        self.batchsize = 128
        self.n_z = n_z
        self.images = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.label = tf.placeholder(tf.float32, [None, 10])
        self.images_flat = tf.reshape(self.images, [-1, 28*28])
        self.conditional_vae = conditional_vae
        
        if conditional_vae==False:
            z_mean, z_stddev = self.encoder(self.images)
            #print z_mean, z_stddev
            
            sample_test = tf.random_normal([5000, self.n_z], 0, 1, dtype=tf.float32)
            self.guessed_z_test = z_mean + (z_stddev * sample_test)
            
            sample = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
            self.guessed_z = z_mean + (z_stddev * sample)
            
            self.generated_images = self.decoder(self.guessed_z)
            generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])
            
            self.z = tf.placeholder(tf.float32, [None, self.n_z])
            self.generated_test_images = self.decoder(self.z, reuse=True)
        else:
            z_mean, z_stddev = self.encoder(self.images, self.label)
            
            sample_test = tf.random_normal([5000, self.n_z], 0, 1, dtype=tf.float32)
            self.guessed_z_test = z_mean + (z_stddev * sample_test)
            
            sample = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
            self.guessed_z = z_mean + (z_stddev * sample)
            
            self.generated_images = self.decoder(self.guessed_z, self.label)
            generated_flat = tf.reshape(self.generated_images, [self.batchsize, 28*28])

            self.z = tf.placeholder(tf.float32, [None, self.n_z])
            self.generated_test_images = self.decoder(self.z, self.label, reuse=True)
            
        
        
        
        
        self.generation_loss = -tf.reduce_sum(self.images_flat * tf.log(1e-8 + generated_flat) + (1-self.images_flat) * tf.log(1e-8 + 1 - generated_flat),1)
        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1, 1)
        
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

    
    def train(self):
        
        np_x_fixed, np_y = self.mnist.test.next_batch(5000)
        np_x_fixed = np_x_fixed.reshape(-1, 28, 28, 1)
        np_x_fixed = (np_x_fixed > 0.5).astype(np.float32)
        
        visualization, visualization_label = self.mnist.train.next_batch(self.batchsize)
        reshaped_vis = visualization.reshape(self.batchsize,28,28)
        ims("results/base.jpg",merge(reshaped_vis[:64],[8,8]))
        
        
        # train
        saver = tf.train.Saver(max_to_keep=2)
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(10):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch, label = self.mnist.train.next_batch(self.batchsize)
                    #print label
                    batch = batch.reshape([-1, 28, 28, 1])
                    if self.conditional_vae:
                        _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch, self.label:label})
                    else:
                        _, gen_loss, lat_loss = sess.run((self.optimizer, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    # dumb hack to print cost every epoch
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss))
                        saver.save(sess, os.getcwd()+"/training/train",global_step=epoch)
                        
                        if self.conditional_vae:
                            generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization.reshape([-1, 28, 28, 1]),self.label: visualization_label})
                        else:
                            generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization.reshape([-1, 28, 28, 1])})

                            
                        generated_test = generated_test.reshape(self.batchsize,28,28)
                        ims("results/"+str(epoch)+".png",merge(generated_test[:64],[8,8]))
                        
                        if self.n_z == 2:
                            if self.conditional_vae:
                                np_q_mu = sess.run(self.guessed_z_test, {self.images: np_x_fixed, self.label: np_y})
                            else:
                                np_q_mu = sess.run(self.guessed_z_test, {self.images: np_x_fixed})
                                
                            cmap = mpl.colors.ListedColormap(sns.color_palette("husl"))
                            f, ax = plt.subplots(1, figsize=(6 * 1.1618, 6))
                            im = ax.scatter(np_q_mu[:, 0], np_q_mu[:, 1], c=np.argmax(np_y, 1), cmap=cmap,
                                        alpha=0.7)
                            ax.set_xlabel('First dimension of sampled latent variable $z_1$')
                            ax.set_ylabel('Second dimension of sampled latent variable mean $z_2$')
                            ax.set_xlim([-10., 10.])
                            ax.set_ylim([-10., 10.])
                            f.colorbar(im, ax=ax, label='Digit class')
                            plt.tight_layout()
                            plt.savefig(os.path.join('./latent_space',
                                                     'posterior_predictive_map_frame_%d.png' % epoch))
                            plt.close()
                            
                            # 
                            nx = ny = 10
                            x_values = np.linspace(-2, 2, nx)
                            y_values = np.linspace(-2, 2, ny)
                            canvas = np.empty((28*ny, 28*nx))
                            for ii, yi in enumerate(x_values):
                                label = np.zeros((1, 10))
                                label[0,ii] = 1
                                #print label
                                for j, xi in enumerate(y_values):
                                    np_z = np.array([[xi, yi]])
                                    if self.conditional_vae:
                                        x_mean = sess.run(self.generated_test_images, {self.z: np_z, self.label:label})
                                    else:
                                        x_mean = sess.run(self.generated_test_images, {self.z: np_z})
                                    canvas[(nx-ii-1)*28:(nx-ii)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
                                ims(os.path.join('./prior_predictive_map_frame/',
                                                    'prior_predictive_map_frame_%d.png' % epoch), canvas)
          # Make the gifs
        if self.n_z == 2:
            os.system(
                'convert -delay 20 -loop 0 {0}/*png {0}/posterior_predictive.gif'
                    .format('latent_space'))
            os.system(
                'convert -delay 20 -loop 0 {0}/*png {0}/prior_predictive.gif'
                    .format('prior_predictive_map_frame'))

            
        
    
    def encoder(self, input_images, label=None, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        with tf.variable_scope('encoder', reuse=reuse):
            tl.layers.set_name_reuse(reuse)            
            h0 = tl.layers.InputLayer(input_images, name='encoder/h0')
            if label !=None:
                cond = tl.layers.InputLayer(label, name = 'encoder/cond')
                h0_flat = tl.layers.FlattenLayer(h0, name='encoder/h0_flat')
                h0_cond = tl.layers.ConcatLayer([h0_flat, cond], name='encoder/h0_cond')
                h0_dense = tl.layers.DenseLayer(h0_cond, 28*28, act=tf.identity, W_init=w_init, name='encoder/h0_dense')
                h0 = tl.layers.ReshapeLayer(h0_dense, shape=tf.shape(h0.outputs), name='encoder/cond_h0')
            h1 = tl.layers.Conv2dLayer(h0, shape = [5, 5, 1, 16], strides=[1, 2, 2, 1], act=lambda x : tl.act.lrelu(x, 0.2),
                                        padding='SAME', W_init=w_init, name='encoder/h1')
            h2 = tl.layers.Conv2dLayer(h1, shape = [5, 5, 16, 32], strides=[1, 2, 2, 1], act=lambda x : tl.act.lrelu(x, 0.2),
                                        padding='SAME', W_init=w_init, name='encoder/h2')
            h2_flat = tl.layers.FlattenLayer(h2, name='encoder/h2_flat')
            
            w_mean = tl.layers.DenseLayer(h2_flat, self.n_z, act=tf.identity, W_init=w_init, name='encoder/w_mean')
            w_stddev = tl.layers.DenseLayer(h2_flat, self.n_z, act=tf.identity, W_init=w_init, name='encoder/w_stddev')
            return w_mean.outputs, w_stddev.outputs
    
    def decoder(self, z, label=None, is_train=True, reuse=False):
        w_init = tf.random_normal_initializer(stddev=0.02)
        with tf.variable_scope("decoder", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            h0 =  tl.layers.InputLayer(z, name='decoder/h0')
            if label != None:
                cond = tl.layers.InputLayer(label, name = 'decoder/cond')
                h0 = tl.layers.ConcatLayer([h0, cond], name = 'decoder/z_cond')
            h1 =  tl.layers.DenseLayer(h0, n_units=7*7*32, W_init=w_init, act=tf.nn.relu, name='decoder/h1')
            h1_reshape =  tl.layers.ReshapeLayer(h1, shape=[-1, 7, 7, 32], name = 'decoder/h1_reshape')
            h2 =  tl.layers.DeConv2dLayer(h1_reshape, act=tf.nn.relu, shape=[5, 5, 16, 32], padding='SAME',
                                          output_shape=[tf.shape(h1_reshape.outputs)[0], 14, 14, 16], strides=[1, 2, 2, 1], W_init=w_init, name='decode/h2')
            h3 = tl.layers.DeConv2dLayer(h2, act=tf.sigmoid, shape=[5,5,1,16], padding='SAME',
                                         output_shape=[tf.shape(h2.outputs)[0], 28, 28, 1], strides=[1, 2, 2, 1], W_init=w_init, name='decode/h3')
            return h3.outputs
            
    
    
            

model = vae('mnist', 2, conditional_vae=True)
model.train()
```
