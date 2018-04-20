参考：

- 原论文：Dynamic Routing Between Capsules, https://arxiv.org/abs/1710.09829
- [先读懂CapsNet架构然后用TensorFlow实现：全面解析Hinton的提出的Capsule](https://blog.csdn.net/uwr44uouqcnsuqb60zk2/article/details/78463687)
- [lkolezhuk/CapsNetMNIST](https://github.com/lkolezhuk/CapsNetMNIST)


----------
# 算法结构

![这里写图片描述](https://ss.csdn.net/p?https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/52501image%20%2815%29.png)

![这里写图片描述](https://ss.csdn.net/p?https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/01507v2-a402c9d94e36bc6bdcf8a6fbe06f78b3_hd.jpg)

![这里写图片描述](https://ss.csdn.net/p?https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/93804image%20%2817%29.png)

![这里写图片描述](https://ss.csdn.net/p?https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/55610%E7%BB%98%E5%9B%BE1.jpg)

# 损失函数与最优化
![这里写图片描述](https://ss.csdn.net/p?https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/46209image%20%2816%29.png)

# tensorflow 实现
参考： [lkolezhuk/CapsNetMNIST](https://github.com/lkolezhuk/CapsNetMNIST)

```python
# input： [batch_size, 1152, 1, 8, 1]
# W： [1, 1152, 10, 8, 16]

input = tf.tile(input, [1, 1, 10, 1, 1]) # [batch_size, 1152, 10, 8, 1]
W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1]) # [batch_size, 1152, 10, 8, 16]

# (W: 8x16->16x8（转置） ,input:8x1) --> 16x1  u_hat:[batch_size,1152,10,16,1]
u_hat = tf.matmul(W, input, transpose_a=True) # [batch_size,1152,10,16,1]  只改变最后两个维度
# 上式 等价于 u_hat = tf.matmul(tf.transpose(W,[0,1,2,4,3]),input) 
# tf.transpose(W,[0,1,2,4,3]) # shape [batch_size, 1152, 10, 16, 8]
# input shape [batch_size, 1152, 10, 8, 1]
```

```python
# c_IJ：[128,1152,10,1,1]
# u_hat_stopped: [128,1152,10,16,1]
# 只改变最后一个维度，可以看作 1x1 multiply 16x1
s_J = tf.multiply(c_IJ, u_hat_stopped) # [128,1152,10,16,1]
s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True) # [128,1,10,16,1]
```

## squash

![这里写图片描述](https://ss.csdn.net/p?https://image.jiqizhixin.com/uploads/wangeditor/77f966b8-eba1-4737-a4c2-551e0d4e7610/12310image%20%288%29.png)
```python

def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
return(vec_squashed)
```

## routing

```python
def routing(input, b_IJ):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, 10, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [num_caps_i, num_caps_j, len_u_i, len_v_j]
    W = tf.get_variable('Weight', shape=(1, 1152, 10, 8, 16), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=cfg.stddev))

    # Eq.2, calc u_hat
    # do tiling for input and W before matmul
    # input => [batch_size, 1152, 10, 8, 1]
    # W => [batch_size, 1152, 10, 8, 16]
    input = tf.tile(input, [1, 1, 10, 1, 1])
    W = tf.tile(W, [cfg.batch_size, 1, 1, 1, 1])
    assert input.get_shape() == [cfg.batch_size, 1152, 10, 8, 1]

    # in last 2 dims:
    # [8, 16].T x [8, 1] => [16, 1] => [batch_size, 1152, 10, 16, 1]
    # tf.scan, 3 iter, 1080ti, 128 batch size: 10min/epoch
    # u_hat = tf.scan(lambda ac, x: tf.matmul(W, x, transpose_a=True), input, initializer=tf.zeros([1152, 10, 16, 1]))
    # tf.tile, 3 iter, 1080ti, 128 batch size: 6min/epoch
    u_hat = tf.matmul(W, input, transpose_a=True)
    assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(cfg.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, dim=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == cfg.iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                assert s_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < cfg.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)
```
## 代码流程结构
![这里写图片描述](http://img.blog.csdn.net/20180420110631203?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
