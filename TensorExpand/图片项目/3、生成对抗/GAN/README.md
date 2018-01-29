参考：

- https://github.com/fengzhongyouxia/Tensorflow-learning/tree/master/DCGAN-tensorflow

- http://blog.topspeedsnail.com/archives/10977


----------
GAN相关代码实现：

- [DCGAN TensorFlow实现](https://github.com/carpedm20/DCGAN-tensorflow)
- [根据文本描述生成图像](https://github.com/paarthneekhara/text-to-image)（反过来的: [看图说话Show and Tell](https://github.com/tensorflow/models/blob/master/research/im2txt/)）
- [图像补全，叫你在打码](https://github.com/bamos/dcgan-completion.tensorflow)
- [TF-VAE-GAN-DRAW](https://github.com/ikostrikov/TensorFlow-VAE-GAN-DRAW)
- [Auxiliary Classifier GAN](https://github.com/buriburisuri/ac-gan)
- [InfoGAN时间序列数据分类](https://github.com/buriburisuri/timeseries_gan)
- [生成视频](https://github.com/cvondrick/videogan)
- [Generative Models (OpenAI)](https://openai.com/blog/generative-models/)

# 流程图
![这里写图片描述](http://img.blog.csdn.net/20180129143655852?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 一个TensorFlow代码示例（生成明星脸-EBGAN）

使用的数据集：[Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)，这个数据集包含20万明星脸，可用来做人脸检测、人脸特征识别等等任务。

下载地址：Google Drive或[Baidu云](https://pan.baidu.com/s/1eSNpdRG?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#list/path=/)。

[Energy Based Generative Adversarial Networks (EBGAN)](https://arxiv.org/pdf/1609.03126v2.pdf)
