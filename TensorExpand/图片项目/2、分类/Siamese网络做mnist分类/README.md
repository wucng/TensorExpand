参考：[5、Siamese 网络做mnist分类](https://github.com/fengzhongyouxia/TensorExpand/tree/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/5%E3%80%81Siamese%20%E7%BD%91%E7%BB%9C%E5%81%9Amnist%E5%88%86%E7%B1%BB)

参考：
http://blog.csdn.net/wc781708249/article/details/78555160#t38


----------

# Siamese 网络
图像–>卷积神经网络–>编码
类似于自编码中的encodeing部分

![这里写图片描述](http://img.blog.csdn.net/20180114212622598?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180114212628822?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# Triplet 损失
A,P,N(其中A,P 来自同一人 不同图像，A，N来自不同人)
loss function: $||f(A)−f(p)||^2+α<=||f(A)−f(N)||^2$

![这里写图片描述](http://img.blog.csdn.net/20180114214808659?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180114214817663?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180114214823443?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180114214829229?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 流程图

![这里写图片描述](http://img.blog.csdn.net/20180114223457440?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 推理流程
![这里写图片描述](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/5%E3%80%81Siamese%20%E7%BD%91%E7%BB%9C%E5%81%9Amnist%E5%88%86%E7%B1%BB/%E6%8E%A8%E7%90%86%E6%B5%81%E7%A8%8B.png)

# 项目分析
受人脸识别验证启发，使用 Siamese 网络 对mnist数据进行分类
[minst数据](https://pan.baidu.com/disk/home?#list/path=/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%95%B0%E6%8D%AE&vmode=list)

为了方便，只取mnist中的 0,1,2 进行分类

# 转成 二分类问题

![这里写图片描述](http://img.blog.csdn.net/20180130094954207?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)                 

![这里写图片描述](http://img.blog.csdn.net/20180130094224139?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)         

验证两张图片是否是同一人，
将图片进行编码，  在将编码 之差输入到新的神经网络中，同一人输出1，否则0    

# 流程图
![这里写图片描述](http://img.blog.csdn.net/20180130100742411?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)     
