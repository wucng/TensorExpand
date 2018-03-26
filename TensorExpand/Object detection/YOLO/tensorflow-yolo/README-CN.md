参考：

- [YOLO训练](http://blog.csdn.net/hrsstudy/article/details/65644517)
- [pjreddie/darknet](https://github.com/pjreddie/darknet)
- https://pjreddie.com/darknet/
- [gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- [longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)
- [mxcl/YOLOKit](https://github.com/mxcl/YOLOKit)
- [llSourcell/YOLO_Object_Detection](https://github.com/llSourcell/YOLO_Object_Detection)
- [experiencor/basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras)
- [philipperemy/yolo-9000](https://github.com/philipperemy/yolo-9000)
- [nilboy/tensorflow-yolo](https://github.com/nilboy/tensorflow-yolo)
- [KJCracks/yololib](https://github.com/KJCracks/yololib)


----------
- [nilboy/tensorflow-yolo](https://github.com/nilboy/tensorflow-yolo)


----------
“YOLO：实时对象检测”的Tensorflow实现（训练和测试）

# tensorflow-yolo
## Require

```
tensorflow-1.0
```
## download pretrained model
yolo_tiny: https://drive.google.com/file/d/0B-yiAeTLLamRekxqVE01Yi1RRlk/view?usp=sharing

```
mv yolo_tiny.ckpt models/pretrain/ 
```
## Train
### Train on pascal-voc2007 data
#### Download pascal-Voc2007 data

- Download the training, validation and test data

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
- 将所有这些tar解压到一个名为`VOCdevkit`的目录中
```python
# 执行以下命令将解压到一个名为VOCdevkit的目录中
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
- 它应该有这个基本结构

```python
$VOCdevkit/                           # development kit
$VOCdevkit/VOCcode/                   # VOC utility code
$VOCdevkit/VOC2007                    # image sets, annotations, etc.
# ... and several other directories ...
```
- 为PASCAL VOC数据集创建符号链接

```
cd $YOLO_ROOT/data
ln -s $VOCdevkit VOCdevkit2007
```
使用符号链接是一个好主意，因为您可能希望在多个项目之间共享相同的PASCAL数据集安装。

#### 将Pascal-voc数据转换为text_record文件

```python
python tools/preprocess_pascal_voc.py
```


----------


# train

```
python tools/train.py -c conf/train.cfg
```
# Train your customer data

- 将您的训练数据转换为text_record文件（对pascal_voc的格式引用）

- 编写你自己的train配置文件

- train（python tools / train.py -c $ your_configure_file）

# test demo

```
python demo.py
```