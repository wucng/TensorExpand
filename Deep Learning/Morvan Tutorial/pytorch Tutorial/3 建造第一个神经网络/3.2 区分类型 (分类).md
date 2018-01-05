学习资料:

[本节的全部代码](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/302_classification.py)
[用 Tensorflow 达到同样效果的代码](https://github.com/MorvanZhou/Tensorflow-Tutorial/blob/master/tutorial-contents/301_simple_regression.py)
[我制作的 什么是神经网络 动画简介](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/2-1-NN/)
[PyTorch 官网](http://pytorch.org/)


----------
[toc]

# pytorch 程序
```python
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted

plt.ion()   # something about plotting

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1] # 返回预测的标签值 0,1
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```

# 对应的tensorflow程序

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

# torch.manual_seed(1)    # reproducible
np.random.seed(1)

# make fake data
n_data = np.random.rand(100, 2)

x0 = 2*n_data  # class0 x data (tensor), shape=(100, 2)
y0 = np.zeros(100,np.int32)               # class0 y data (tensor), shape=(100, 1)
x1 = -2*n_data    # class1 x data (tensor), shape=(100, 2)
y1 = np.ones(100,np.int32)                # class1 y data (tensor), shape=(100, 1)
# x = np.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

x=np.vstack((x0,x1)).astype(np.float32)
y=np.vstack((y0[:,np.newaxis],y1[:,np.newaxis])).astype(np.int32)


# torch can only train on Variable, so convert them to Variable
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

hidden=tf.layers.dense(tf.convert_to_tensor(x),10,tf.nn.relu)
predict=tf.layers.dense(hidden,2)

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict,labels=y.flatten()))
optimizer=tf.train.GradientDescentOptimizer(0.02).minimize(loss)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()


plt.ion()   # something about plotting

for t in range(100):
    sess.run(optimizer)

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        # prediction = torch.max(F.softmax(out), 1)[1] #
        prediction=tf.argmax(predict,1).eval().squeeze()
        # pred_y = prediction.data.numpy().squeeze()
        pred_y = prediction.squeeze()
        target_y = y.flatten()
        plt.scatter(x[:, 0], x[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(0, -2, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```

# torch数据转numpy tensorflow程序

```python
import torch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible

# make fake data
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)    # shape (200,) LongTensor = 64-bit integer

x=x.numpy() # tensor转numpy
y=y.numpy() # tensor转numpy

# torch can only train on Variable, so convert them to Variable
# x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.show()

hidden=tf.layers.dense(tf.convert_to_tensor(x),10,tf.nn.relu)
predict=tf.layers.dense(hidden,2)

loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict,labels=y.flatten()))
optimizer=tf.train.GradientDescentOptimizer(0.02).minimize(loss)

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()


plt.ion()   # something about plotting

for t in range(100):
    sess.run(optimizer)

    if t % 2 == 0:
        # plot and show learning process
        plt.cla()
        # prediction = torch.max(F.softmax(out), 1)[1] #
        prediction=tf.argmax(predict,1).eval().squeeze()
        # pred_y = prediction.data.numpy().squeeze()
        pred_y = prediction.squeeze()
        target_y = y.flatten()
        plt.scatter(x[:, 0], x[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
```


