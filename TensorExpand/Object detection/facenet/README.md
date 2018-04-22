参考：

- [davidsandberg/facenet](https://github.com/davidsandberg/facenet)


----------
# Face Recognition using Tensorflow Build Status

这是本文中描述的人脸识别器的TensorFlow实现“[FaceNet：人脸识别和聚类的统一嵌入](https://arxiv.org/abs/1503.03832)”，该项目也使用论文中的想法[Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)来自 [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/) at Oxford.


----------


# 兼容性
代码使用Tensorflow r1.7在Ubuntu 14.04下使用Python 2.7和Python 3.5进行测试。 测试用例可以在[这里](https://github.com/davidsandberg/facenet/tree/master/test)找到，结果可以在[这里](travis-ci.org/davidsandberg/facenet)找到。


----------


# news
添加了代码以[在自己的图像上训练分类器](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images)。 将facenet_train.py重命名为train_tripletloss.py，将facenet_train_classifier.py重命名为train_softmax.py。


----------


# Pre-trained models
[Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py)

注意：如果您使用任何模型，请不要忘记给那些提供训练数据集的人以合适的功劳。


----------


# 灵感
代码深受[OpenFace](https://github.com/cmusatyalab/openface)实现的启发。


----------


# Training data
[CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)数据集已用于训练。 这个训练集包含在面部检测之后共10 575个身份的总共453 453张图像。 如果数据集在训练之前被过滤，则可以看到一些性能改进。 关于如何完成的更多信息将在稍后提供。 表现最好的模型已经在[VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)数据集上进行了训练，该数据集由〜3.3M面和〜9000级组成。


----------


# Pre-processing


----------


## 使用MTCNN进行脸部对齐
上述方法的一个问题似乎是Dlib人脸检测器遗漏了一些硬例子（部分遮挡，轮廓等）。 这使得训练集太“容易”，导致模型在其他基准测试中表现更差。 为了解决这个问题，其他人脸标志探测器已经过测试。已经证明在这种环境中工作得很好的一个人脸标志探测器是[多任务CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)。Matlab / Caffe的实现可以在[这里](https://github.com/kpzhang93/MTCNN_face_detection_alignment)找到，并且这已经用于面对齐，并且具有非常好的结果。MTCNN的Python / Tensorflow实现可以在[这里](https://github.com/davidsandberg/facenet/tree/master/src/align)找到。 该实现不会给出与Matlab / Caffe实现相同的结果，但性能非常相似。


----------
## Running training
目前，通过使用softmax损失对模型进行训练可以获得最佳结果。 有关如何在CASIA-WebFace数据集上使用softmax损失来训练模型的详细信息，请参阅[Inception-ResNet-v1和Classifier的分类器](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1)训练。


----------
# Pre-trained models


----------
## Inception-ResNet-v1 model
提供了一些预训练模型。 他们在Inception-Resnet-v1模型中使用softmax损失进行训练。 数据集已使用[MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align)进行对齐。


----------


## Performance
型号[20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)的LFW的精度为0.99650±0.00252。 有关如何运行测试的说明可以在[LFW上的验证](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw)页面中找到。请注意，模型的输入图像需要使用固定图像标准化进行标准化（使用该选项`--use_fixed_image_standardization ` 当运行`validate_on_lfw.py`）


----------

