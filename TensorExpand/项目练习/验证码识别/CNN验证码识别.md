参考：

1、https://github.com/luyishisi/tensorflow

2、http://blog.topspeedsnail.com/

3、http://blog.topspeedsnail.com/archives/10858

4、https://zhuanlan.zhihu.com/p/25779608

-----
[toc]

博文网址： [urlteam](https://www.urlteam.org/2017/03/tensorflow%E8%AF%86%E5%88%AB%E5%AD%97%E6%AF%8D%E6%89%AD%E6%9B%B2%E5%B9%B2%E6%89%B0%E5%9E%8B%E9%AA%8C%E8%AF%81%E7%A0%81-%E5%BC%80%E6%94%BE%E6%BA%90%E7%A0%81%E4%B8%8E98%E6%A8%A1%E5%9E%8B/) ，github 网址：[tensorflow_cnn](https://github.com/luyishisi/Anti-Anti-Spider/tree/master/1.%E9%AA%8C%E8%AF%81%E7%A0%81/tensorflow_cnn)

新开一个专门存储TensorFlow项目的仓库逐步更新欢迎star ：[tensorflow](https://github.com/luyishisi/tensorflow)

# 项目综述：
用卷积神经网络识别复杂字符验证码

用4层Cnn网络，识别破解python自生成的复杂扭曲验证码，

![这里写图片描述](https://www.urlteam.org/wp-content/uploads/2017/03/92757507-9907-40E9-BEA3-E9B840A8A6D1.jpg)

CNN需要大量的样本进行训练。如果使用数字+大小写字母CNN网络有4*62个输出，只使用数字CNN网络有4*10个输出。因此需要一个脚本自动生成训练集。

说明： 数字 0~9 共10个，大、小写字母各26个，  数字+大小写字母 共10+26*2=62


# 实践流程：
TensorFlow环境搭建：[官网下查看安装教程](https://www.tensorflow.org/versions/r0.12/get_started/index.html)

测试批量验证码生成训练集： [github](https://github.com/luyishisi/Anti-Anti-Spider/blob/master/1.%E9%AA%8C%E8%AF%81%E7%A0%81/tensorflow_cnn/gen_captcha.py)

TensorFlow—cnn 批量生成验证码并用cnn训练： [github](https://github.com/luyishisi/Anti-Anti-Spider/blob/master/1.%E9%AA%8C%E8%AF%81%E7%A0%81/tensorflow_cnn/tensorflow_cnn_train.py)

将训练模型存放于同一目录下，测试结果：[github](https://github.com/luyishisi/Anti-Anti-Spider/blob/master/1.%E9%AA%8C%E8%AF%81%E7%A0%81/tensorflow_cnn/tensorflow_cnn_test_model.py)

98%准确率模型下载：链接: https://pan.baidu.com/s/1cs0LCM 密码: sngx

# 资源简介：
1、本项目由urlteam维护，欢迎star

2、相关的验证码破解系列可以在这里找到：[github](https://github.com/luyishisi/Anti-Anti-Spider/tree/master/1.%E9%AA%8C%E8%AF%81%E7%A0%81)

3、逐步更新TensorFlow系列项目：[github](https://github.com/luyishisi/tensorflow)

4、博客主页：[The world we move forward together](https://www.urlteam.org/)

相关论文：

> [Multi-digit Number Recognition from Street View Imagery using Deep CNN](http://link.zhihu.com/?target=https://arxiv.org/pdf/1312.6082.pdf)

> [CAPTCHA Recognition with Active Deep Learning](http://link.zhihu.com/?target=https://vision.in.tum.de/_media/spezial/bib/stark-gcpr15.pdf)

> [Number plate recognition with Tensorflow](http://matthewearl.github.io/2016/05/06/cnn-anpr/)

> [最初cnn学习自此](http://blog.topspeedsnail.com/archives/10858)
