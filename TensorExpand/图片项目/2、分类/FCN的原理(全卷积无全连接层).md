[网站地址](http://blog.csdn.net/taigw/article/details/51401448)

链接：[经典 CNNs 的 TensorFlow 实现资源汇总](https://www.jianshu.com/p/68cf89138dca)

[tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn)

---
[toc]

FCN将传统CNN中的全连接层转化成一个个的卷积层。如下图所示，在传统的CNN结构中，前5层是卷积层，第6层和第7层分别是一个长度为4096的一维向量，第8层是长度为1000的一维向量，分别对应1000个类别的概率。FCN将这3层表示为卷积层，卷积核的大小(通道数，宽，高)分别为（4096,1,1）、（4096,1,1）、（1000,1,1）。所有的层都是卷积层，故称为全卷积网络。

![http://img.blog.csdn.net/20160514044341551](http://img.blog.csdn.net/20160514044341551)

# 卷积层代替FC层
![这里写图片描述](http://img.blog.csdn.net/20171214233656511?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


# FCN的优点和不足

与传统用CNN进行图像分割的方法相比，FCN有两大明显的优点：一是==可以接受任意大小的输入图像==，而不用要求所有的训练图像和测试图像具有同样的尺寸。二是更加高效，因为避免了由于使用像素块而带来的重复存储和计算卷积的问题。

同时FCN的缺点也比较明显：一是得到的结果还是不够精细。进行8倍上采样虽然比32倍的效果好了很多，但是上采样的结果还是比较模糊和平滑，对图像中的细节不敏感。二是对各个像素进行分类，没有充分考虑像素与像素之间的关系，忽略了在通常的基于像素分类的分割方法中使用的空间规整（spatial regularization）步骤，缺乏空间一致性。

# 代码
参考：[fcn8_vgg](https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn8_vgg.py)

对应的神经网络图
![这里写图片描述](http://img.blog.csdn.net/20180111152955718?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
