新的next_batch 使用

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.random((100000, 2))
y_data = [float(x1 + x2 < 1) for (x1, x2) in x_data]  # 小于1则为1，否则则为0
y_data=np.array(y_data,np.float32)[:,np.newaxis]

plt.scatter(x_data[:,0],x_data[:,1],s=5,c=np.arctan2(x_data[:,0],x_data[:,1]),alpha=5)
# plt.scatter(x_data[:,0],y_data)
# plt.scatter(x_data[:,1],y_data)
plt.show()

x_input = tf.placeholder(tf.float32, [None, 2], name="x_input")
y_input = tf.placeholder(tf.float32, [None, 1], name="y_input")

W = tf.Variable(tf.zeros([2, 1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

y = tf.matmul(x_input, W) + b

cross_entropy = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_input))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start=0
    batch_size=128
    end=0
    for step in range(1000):
        end=min(len(x_data),start+batch_size)
        sess.run(train_step, feed_dict={x_input: x_data[start:end], y_input: y_data[start:end]})
        if end==len(x_data):
            start=0
        else:
            start=end
        if step%50==0:
            print(sess.run(W), sess.run(b))

    print('--------------------')
    print(x_data[:10],np.matmul(x_data[:10],sess.run(W))+sess.run(b))


```
