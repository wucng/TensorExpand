#! /usr/bin/python
# -*- coding: utf8 -*-

'''
验证码图像对应4个字符，放在一行（每一行对应一个样本）对应（4列）4个标签 转成one_hot 标签

尝试使用非one_hot标签训练 报错
一般一张图对应1个标签，放在一行（每一行对应一个样本）对应（1列）1个标签 可以使用sparse_softmax_cross_entropy_with_logits 来训练
但这里 验证码图像对应4个字符，放在一行（每一行对应一个样本）对应（4列）4个标签，使用sparse_softmax_cross_entropy_with_logits 来训练报错，
只能转成one_hot 标签 使用softmax_cross_entropy_with_logits，或者使用MSE（也最好转成one_hot） 来训练

'''

from gen_captcha import gen_captcha_text_and_image
from gen_captcha import number
from gen_captcha import alphabet
from gen_captcha import ALPHABET

import numpy as np
import tensorflow as tf

text, image = gen_captcha_text_and_image()
print("验证码图像channel:", image.shape)  # (60, 160, 3)
# 图像大小
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
MAX_CAPTCHA = len(text) # 默认为4
print("验证码文本最长字符数", MAX_CAPTCHA)  # 验证码最长4字符; 我全部固定为4,可以不固定. 如果验证码长度小于4，用'_'补齐


# 把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1) # 按通道求平均 转成灰度图像
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""

# 文本转向量
char_set = number + alphabet + ALPHABET + ['_']  # 如果验证码长度小于4, '_'用来补齐
CHAR_SET_LEN = len(char_set) # 10个数字 大小写字母 26*2  1个'_' 共 10+26*2+1=63


def text2vec(text):
    '''
    文本转成对应的向量， 如：'12aN' 对应的标签值 1 2 36 23 转成one_hot [0 1 0*61 ,0 0 1 0*60,0*34 1 28*0,21*0 1 41*0]
    :param text: 
    :return: 
    '''
    text_len = len(text) # 4
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    # vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN) # [4*63,]
    vector=np.zeros(MAX_CAPTCHA)  # [4,]

    def char2pos2(c):
        '''
        字符转成对应的ascill值，数字0~9 ascill 48~57，大写字母 65~90，小写字母97~122，缩放到0~63 做标签
        :param c: 输入字符
        :return: 数字：0~9，大写字母：10~35，小写字母：36~61，"_":62
        '''
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48 # 数字：0~9，大写字母：17~42，小写字母：47~74
        if k > 9:       # 是字母
            k = ord(c) - 55 # 大写字母：10~35，小写字母：42~67
            if k > 35: # 小写字母
                k = ord(c) - 61 # 小写字母：36~61
                if k > 61:
                    raise ValueError('No Map')
        return k

    def char2pos(c):
        '''
        字符转成对应的ascill值，数字0~9 ascill 48~57，大写字母 65~90，小写字母97~122 并
        缩放到0~63 做标签
        :param c: 输入字符
        :return: 数字：0~9，大写字母：10~35，小写字母：36~61，"_":62
        '''
        if c == '_':
            k = 62
            return k
        # if isinstance(c,str)
        if c.isdigit():  # 数字
            k = ord(c) - 48  # 数字：0~9
        elif c.isupper():  # 大写字母
            k = ord(c) - 55  # 大写字母：10~35
        elif c.islower():  # 小写字母
            k = ord(c) - 61  # 小写字母：36~61
        else:
            raise ValueError('No Map')
        return k
    '''
    # 转one_hot 标签
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1 # 转成one_hot 标签 如：'12aN' 对应的标签 1 2 36 23 转成one_hot [0 1 0*61 ,0 0 1 0*60,0*34 1 28*0,21*0 1 41*0]
    return vector
    '''
    # 使用非one_hot 标签
    for i,c in enumerate(text):
        vector[i]=char2pos(c)

    return vector

# 向量转回文本
def vec2text2(vec):
    '''
    向量转回文本
    :param vec: 如：[0 1 0*61 ,0 0 1 0*60,0*34 1 28*0,21*0 1 41*0]
    :return: [1 2 36 23]==>'12aN'
    '''
    char_pos = vec.nonzero()[0] # [1 65 162 212]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10: # 数字
            char_code = char_idx + ord('0')
        elif char_idx < 36: # 大写字母
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62: # 小写字母
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

def vec2text(vec):
    '''
    向量转回文本
    :param vec: 如：[0 1 0*61 ,0 0 1 0*60,0*34 1 28*0,21*0 1 41*0]
    :return: [1 2 36 23]==>'12aN'
    '''
    # char_pos = vec.nonzero()[0] # [1 65 162 212]
    char_pos=vec.astype(int)
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        # char_idx = c % CHAR_SET_LEN
        # char_idx=c-char_at_pos*CHAR_SET_LEN
        char_idx=c
        if char_idx < 10: # 数字
            char_code = char_idx + 48
        elif char_idx < 36: # 大写字母
            char_code = char_idx +55
        elif char_idx < 62: # 小写字母
            char_code = char_idx + 61
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

# """
#向量（大小MAX_CAPTCHA*CHAR_SET_LEN）用0,1编码 每63个编码一个字符，这样顺利有，字符也有
vec = text2vec("F5Sd")
text = vec2text(vec)
print(text)  # F5Sd
vec = text2vec("SFd5")
text = vec2text(vec)
print(text)  # SFd5
# """


# 生成一个训练batch
def get_next_batch(batch_size=128):
    '''
    生成一个训练batch
    :param batch_size: 每批次训练样本大小
    :return: image shape [batch,60*160]；labels shape [batch ,4*63]
    '''
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH]) # [batch,60*160]
    batch_y = np.zeros([batch_size, MAX_CAPTCHA ]) # [batch ,4]

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3): # 保证生产的图像大小为[60,160,3]
                return text, image
            # assert image.shape == (60, 160, 3), '图像大小不是（60,160,3）'
            # return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image) # 转成灰度图 [60,160,1]

        batch_x[i, :] = image.flatten() / 255.  # (image.flatten()-128)/128  mean为0  [60*160,]
        batch_y[i, :] = text2vec(text) # [4]

    return batch_x, batch_y # [batch,60*160],[batch,4]

####################################################################

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) # [batch,60*160]
Y = tf.placeholder(tf.int64, [None, MAX_CAPTCHA * CHAR_SET_LEN]) # [batch ,4]
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1]) # [batch,60,160,1]

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)) # [batch,60,160,32]
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [batch,30,80,32]
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)) # [batch,30,80,64]
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [batch,15,40,64]
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)) # [batch,15,40,64]
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # [batch,8,20,64]
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]]) # [batch,8*20*64]
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d)) # [batch,1024]
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out) # [batch,4*63]
    # out = tf.nn.softmax(out)
    return out


# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn() # [batch,4*63]
    # loss
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))
    # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=output))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]) # [batch,4,63]
    max_idx_p = tf.argmax(predict, 2) # 按第三维度求最大值 [batch,4]
    # max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    max_idx_l=Y
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        step = 0
        while step<1000:#True:
            batch_x, batch_y = get_next_batch(64)
            batch_y=batch_y.astype(np.int64)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print(step, acc)
                # 如果准确率大于50%,保存模型,完成训练
                if acc > 0.5:
                    saver.save(sess, "crack_capcha.model", global_step=step)
                    break

            step += 1

train_crack_captcha_cnn()