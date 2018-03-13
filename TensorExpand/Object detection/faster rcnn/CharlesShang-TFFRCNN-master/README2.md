参考：

- [CharlesShang/TFFRCNN](https://github.com/CharlesShang/TFFRCNN)


----------
这是一个Faster RCNN（TFFRCNN）的实验性Tensor Flow实现，主要基于[smallcorgi](https://github.com/smallcorgi/Faster-RCNN_TF)和[rbgirshick](https://github.com/rbgirshick/py-faster-rcnn)的工作。 我已经在`lib`路径下重新组织了库，使得每个python模块相互独立，因此您可以轻松理解并重新编写代码。

有关R-CNN的详细信息，请参考[Rochen Raston提出的区域提议网络实时对象检测Faster R-CNN，何开明，Ross Girshick，孙孙](https://arxiv.org/pdf/1506.01497v3.pdf)。

# 致谢：

[rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

[smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)

[zplizzi/tensorflow-fast-rcnn](https://github.com/zplizzi/tensorflow-fast-rcnn)

# 要求：软件

Tensorflow的要求（请参阅：Tensorflow）

您可能没有的Python包：cython，python-opencv，easydict（推荐安装：Anaconda）

# 要求：硬件

为了使用VGG16训练Faster R-CNN的端到端版本，GPU内存的3G是足够的（使用CUDNN）

# 安装（足够演示）
克隆faster R-CNN存储库

```
git clone https://github.com/CharlesShang/TFFRCNN.git
```
构建Cython模块

```python
cd TFFRCNN/lib
make # compile cython and roi_pooling_op, you may need to modify make.sh for your platform
```
# Demo
成功完成基本安装后，您将准备好运行演示。

运行演示

```
cd $TFFRCNN
python ./faster_rcnn/demo.py --model model_path
```
该demo使用经过训练可在PASCAL VOC 2007上进行检测的VGG16网络执行检测。

# Download list
1. [VGG16 trained on ImageNet](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM)

2. [VGG16 - TFFRCNN (0.689 mAP on VOC07)](https://drive.google.com/file/d/0B_xFdh9onPagX0JWRlR0cTZ5OGc/view?usp=sharing).

3. [VGG16 - TFFRCNN (0.748 mAP on VOC07)](https://drive.google.com/file/d/0B_xFdh9onPagVmt5VHlCU25vUEE/view?usp=sharing)

4. [Resnet50 trained on ImageNet](https://drive.google.com/file/d/0B_xFdh9onPagSWU1ZTAxUTZkZTQ/view?usp=sharing)

5. [Resnet50 - TFFRCNN (0.712 mAP on VOC07)](https://drive.google.com/file/d/0B_xFdh9onPagbXk1b0FIeDRJaU0/view?usp=sharing)

6. [PVANet trained on ImageNet, converted from caffemodel](https://drive.google.com/open?id=0B_xFdh9onPagQnJBdWl3VGQxam8)

# Training on Pascal VOC 2007
1.下载训练，验证，测试数据和VOCdevkit
```shell
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
2.将所有这些tars提取到一个名为`VOCdevkit`的目录中

```python
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

  3、它应该有这个基本结构

```python
$VOCdevkit/                           # development kit
$VOCdevkit/VOCcode/                   # VOC utility code
$VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
```
4、为PASCAL VOC数据集创建符号链接

```python
cd $TFFRCNN/data
ln -s $VOCdevkit VOCdevkit2007
```
5、Download pre-trained model [VGG16](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) and put it in the path `./data/pretrain_model/VGG_imagenet.npy`

6、Run training scripts 

```python
 cd $TFFRCNN
 python ./faster_rcnn/train_net.py --gpu 0 --weights ./data/pretrain_model/VGG_imagenet.npy --imdb voc_2007_trainval --iters 70000 --cfg  ./experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_train --set EXP_DIR exp_dir
```
7、运行分析

```python
cd $TFFRCNN
    # install a visualization tool
    sudo apt-get install graphviz  
    ./experiments/profiling/run_profiling.sh 
    # generate an image ./experiments/profiling/profile.png
```
# 训练KITTI检测数据集
1、Download the KITTI detection dataset

```
 http://www.cvlibs.net/datasets/kitti/eval_object.php
```
2、将所有这些tar提取到`./TFFRCNN/data/`中，目录结构如下所示：

```
 KITTI
    |-- training
            |-- image_2
                |-- [000000-007480].png
            |-- label_2
                |-- [000000-007480].txt
    |-- testing
            |-- image_2
                |-- [000000-007517].png
            |-- label_2
                |-- [000000-007517].txt
```
3.将KITTI转换成Pascal VOC格式

```python
cd $TFFRCNN
./experiments/scripts/kitti2pascalvoc.py \
--kitti $TFFRCNN/data/KITTI --out $TFFRCNN/data/KITTIVOC
```
4.输出目录如下所示：

```
KITTIVOC
    |-- Annotations
             |-- [000000-007480].xml
     |-- ImageSets
             |-- Main
                 |-- [train|val|trainval].txt
     |-- JPEGImages
             |-- [000000-007480].jpg
```
5、“KITTIVOC”的训练就像Pascal VOC 2007一样

```Shell
python ./faster_rcnn/train_net.py \
--gpu 0 \
--weights ./data/pretrain_model/VGG_imagenet.npy \
--imdb kittivoc_train \
--iters 160000 \
--cfg ./experiments/cfgs/faster_rcnn_kitti.yml \
--network VGGnet_train
```

