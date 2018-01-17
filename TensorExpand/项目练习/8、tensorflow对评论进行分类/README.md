参考：http://blog.topspeedsnail.com/archives/10399

- TensorFlow文档：http://tensorflow.org
- [使用Python实现神经网络](http://blog.topspeedsnail.com/archives/10377)
- [Ubuntu 16.04 安装 Tensorflow(GPU支持)](http://blog.topspeedsnail.com/archives/10116)
- Andrew Ng斯坦福公开课
- https://github.com/deepmind


----------
本帖展示怎么使用TensorFlow实现文本的简单分类，判断评论是正面的还是负面的。

# 使用的数据集

英文的[电影评论数据](https://www.cs.cornell.edu/people/pabo/movie-review-data/)（其实不管是英文还是中文，处理逻辑都一样）。

![这里写图片描述](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7-2016-11-13-%E4%B8%8B%E5%8D%889.36.26.png)

neg.txt：5331条负面电影评论(http://blog.topspeedsnail.com/wp-content/uploads/2016/11/neg.txt) （每一行对应一条评论）

pos.txt：5331条正面电影评论 (http://blog.topspeedsnail.com/wp-content/uploads/2016/11/pos.txt)（每一行对应一条评论）

由于处理的是字符串，我们首先要想方法把字符串转换为向量/数字表示。一种解决方法是可以把单词映射为数字ID。

第二个问题是每行评论字数不同，而神经网络需要一致的输入(其实有些神经网络不需要，至少本帖需要)，这可以使用词汇表解决。

# 代码部分
安装nltk（自然语言工具库 [Natural Language Toolkit](http://www.nltk.org/)）

```python
pip install nltk
```
下载nltk数据：

```python
$ python
Python 3.5.2 (v3.5.2:4def2a2901a5, Jun 26 2016, 10:47:25) 
[GCC 4.2.1 (Apple Inc. build 5666) (dot 3)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import nltk
>>> # nltk.download()
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
```
测试nltk安装：

```python
>>> from nltk.corpus import brown
>>> brown.words()
['The', 'Fulton', 'County', 'Grand', 'Jury', 'said', ...]
```

# 思路流程
![这里写图片描述](http://img.blog.csdn.net/20180117145427716?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 文件说明
[main.py](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/8%E3%80%81tensorflow%E5%AF%B9%E8%AF%84%E8%AE%BA%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB/main.py)  
输入序列定长 [batch_size,len(lex)]  # lex为词汇表

label：shape [batch_size,2]

模型：DNN 、rnn、1D cnn

loss：softmax_cross_entropy_with_logits

optimizer ： tf.train.AdamOptimizer().minimize(cost_func)

[main_2.py](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/8%E3%80%81tensorflow%E5%AF%B9%E8%AF%84%E8%AE%BA%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB/main_2.py)

输入序列不定长 [batch_size,None]

label:[batch_size,1]

模型：rnn_net_1

loss: legacy_seq2seq.sequence_loss

```python
cost_func = legacy_seq2seq.sequence_loss([predict], [Y], [tf.ones_like(Y, dtype=tf.float32)], len(lex))
cost = tf.reduce_mean(cost_func)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
optimizer = tf.train.AdamOptimizer(0.001).apply_gradients(zip(grads, tvars))
```


[main_输入长度不固定.py](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/8%E3%80%81tensorflow%E5%AF%B9%E8%AF%84%E8%AE%BA%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB/main_%E8%BE%93%E5%85%A5%E9%95%BF%E5%BA%A6%E4%B8%8D%E5%9B%BA%E5%AE%9A.py)

输入序列不定长 [batch_size,None]

label:[batch_size,2]

模型：rnn_net_1(x)

loss：softmax_cross_entropy_with_logits

optimizer ： tf.train.AdamOptimizer().minimize(cost_func)

