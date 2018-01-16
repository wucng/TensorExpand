参考：[Siamese 网络做mnist分类](https://github.com/fengzhongyouxia/TensorExpand/tree/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/Siamese%20%E7%BD%91%E7%BB%9C%E5%81%9Amnist%E5%88%86%E7%B1%BB)


----------
Siamese 网络 类似于自编码器中的encoding部分，因此结合自编码器（encoding、decoding）来对mnist做分类，参考：[Siamese 网络做mnist分类](https://github.com/fengzhongyouxia/TensorExpand/tree/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/Siamese%20%E7%BD%91%E7%BB%9C%E5%81%9Amnist%E5%88%86%E7%B1%BB)

<font color=#FF00FF size=5>整体思路：先对整个编码器网络（encoding、decoding）进行训练，等训练完成后，取encoding部分进行分类预测</font>


为了简化问题，只取0，1,2 三类图像进行分类

# 算法流程

![这里写图片描述](http://img.blog.csdn.net/20180116101314652?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 推理流程

推理时，只需取出encode部分，这部分与Siamese 网络做mnist分类相同
![这里写图片描述](http://img.blog.csdn.net/20180116101355220?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


