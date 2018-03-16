import numpy as np

if __name__ == "__main__":
	import tensorflow as tf
	from .ROIPoolingWrapper import *

	with tf.Session() as sess:
		img = np.zeros((1,8,8, 9), np.float32)
		boxes = tf.constant([[0,0,2*16,5*16]], dtype=tf.float32)
		print(boxes.get_shape().as_list())

		yOffset=0
		xOffset=0
		chOffset=0
		img[0,yOffset+0:yOffset+1,xOffset+0:xOffset+1,chOffset+0:chOffset+1]=1;
		#img[:,:,:,:]=1
		p = tf.placeholder(tf.float32, shape=img.shape)

		np.set_printoptions(threshold=5000, linewidth=150)

		pooled=positionSensitiveRoiPooling(p, boxes)
		pooled=tf.Print(pooled,[tf.shape(pooled)],"pooled shape", summarize=100)
		print(sess.run(pooled, feed_dict={p: img}))


		loss = tf.reduce_sum(pooled)

		g = tf.gradients(loss, p)

		print(img)
		print(sess.run(g, feed_dict={p: img})[0])
		print(sess.run(g, feed_dict={p: img})[0][:,:,:,1])


