# [Train Faster-RCNN / Mask-RCNN on COCO](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN)

# 环境
- Python 3; OpenCV.
- TensorFlow >= 1.6 (1.4 or 1.5 can run but may crash due to a TF bug);
- pycocotools: `pip3 install pycocotools`
- Pre-trained ImageNet ResNet model from [tensorpack model zoo](http://models.tensorpack.com/).
- COCO data. It needs to have the following directory structure:

```
# <COCO>目录结构
COCO/DIR/
├── annotations/
	├── instances_train2014.json （必须）
	├── instances_val2014.json（必须）
	├── instances_minival2014.json （可选，如果没有修改下相应的代码即可）
	└── instances_valminusminival2014.json（可选）
├── train2014/
	└── COCO_train2014_*.jpg
└── val2014/
	└── COCO_val2014_*.jpg
```

[instances_minival2014.json](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0) 5k image 'minival'与[instances_valminusminival2014.json](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0)minus minival (~35k images)



如果想__训练自己的数据__需将自己的数据写成__COCO的数据格式__，参考：[Pascal VOC转COCO数据](https://blog.csdn.net/wc781708249/article/details/79615210)，[labelme数据转成COCO数据](https://blog.csdn.net/wc781708249/article/details/79611536)，另外还需修改下[config.py #L66](https://github.com/tensorpack/tensorpack/blob/master/examples/FasterRCNN/config.py#L66)行，改成自己数据类别数（如：COCO 80，VOC 20，不含背景）

# Train

```python
./train.py --config \ # 修改配置信息，也可以在config.py中直接修改
    MODE_MASK=True MODE_FPN=True \
    DATA.BASEDIR=/path/to/COCO/DIR \
    BACKBONE.WEIGHTS=/path/to/ImageNet-R50-Pad.npz \
```
要运行分布式训练，请设置`TRAINER = horovod`并参考[HorovodTrainer文档](http://tensorpack.readthedocs.io/modules/train.html#tensorpack.train.HorovodTrainer)。

```python
python3 train.py --config MODE_MASK=True BACKBONE.WEIGHTS=COCO-R50C4-MaskRCNN-Standard.npz DATA.BASEDIR='./data/COCO' 
# 如果没有valminusminival2014,minival2014 修改config.py #65行，按以下方式修改
_C.DATA.TRAIN=['train2014'] 
_C.DATA.VAL='val2014'
```




# 预测图像（并在窗口中显示输出）：

```python
# 如果在服务器上预测
# 将train.py 中的predict修改为（# 467行）
cv2.imwrite('test.jpg',final) # 加上这句
# viz = np.concatenate((img, final), axis=1)
#tpviz.interactive_imshow(viz)

python3 train.py --predict 000177.jpg --load COCO-R50C4-MaskRCNN-Standard.npz --config MODE_MASK=True

python3 train.py --predict 000177.jpg --load COCO-R50FPN-MaskRCNN-Standard.npz --config MODE_MASK=True MODE_FPN=True

python3 train.py --predict 000177.jpg --load COCO-R50FPN-MaskRCNN-StandardGN.npz --config MODE_MASK=True MODE_FPN=True BACKBONE.NORM=GN FPN.NORM=GN FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head
```
`--config` 配置参考[这里](https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN#results)

![COCO-R50C4-MaskRCNN-Standard.jpg](../images/COCO-R50C4-MaskRCNN-Standard.jpg)


# 评估COCO模型的性能

```python
python3 train.py --evaluate output.json --load COCO-R50C4-MaskRCNN-Standard.npz --config MODE_MASK=True DATA.BASEDIR=./data/COCO/
```

# Citing Tensorpack
```
@misc{wu2016tensorpack,
  title={Tensorpack},
  author={Wu, Yuxin and others},
  howpublished={\url{https://github.com/tensorpack/}},
  year={2016}
}
```
