- https://github.com/tensorflow/models/tree/master/research/object_detection

---
预训练好的模型在[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)


# 安装
参考[here](./Installation.md)

# demo
参考[here](./demo.py)


# 训练数据
参考[here](./安装与训练VOC2007及推理.md) or [here](./COCO数据训练及推理.md)
## 准备数据

Tensorflow Object DetectionAPI使用`TFRecord`文件格式读取数据。 提供了两个示例脚本（`create_pascal_tf_record.py`和`create_pet_tf_record.py`），用于将PASCAL VOC数据集和Oxford-IIIT Pet数据集转换为TFRecords。

原始的2012年PASCAL VOC数据集位于[此处](http://101.96.10.46/host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)。 要下载，提取并将其转换为TFRecords，请运行以下命令：


```python
cd ./models/resarch
mkdir 20181227
cd 20181227
# 将已下载数据链接到这个目录下
ln -s /media/data/VOCdevkit2007 VOCdevkit

修改object_detection/utils/dataset_util.py:75
if not xml: --> if len(xml)==0:
```

```python
# From tensorflow/models/research/
# wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# tar -xvf VOCtrainval_11-May-2012.tar
# or ln -s ../../VOCdevkit ./VOCdevkit # 将已下好的数据链接过来

# 如果使用VOC2012 将下面的VOC2007换成VOC2012，路径对应即可
python3 ../object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=../object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2007 --set=train \
    --output_path=pascal_train.record
    
python3 ../object_detection/dataset_tools/create_pascal_tf_record.py \
    --label_map_path=../object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2007 --set=val \
    --output_path=pascal_val.record
```
您应该在`tensorflow/models/research/20181227`目录中最终得到两个名为`pascal_train.record`和`pascal_val.record`的TFRecord文件。
```python
.
├── pascal_train.record
├── pascal_val.record
└── VOCdevkit
```
PASCAL VOC数据集的标签映射可以在`object_detection/data/pascal_label_map.pbtxt`中找到。

## train
```python
cd ./models/resarch
cp -r object_detection/data/pascal_label_map.pbtxt ./20181227
cp -r object_detection/samples/configs/faster_rcnn_resnet101_voc07.config ./20181227
# 这里以faster_rcnn_resnet101_voc07为例，也可以更换成其他模型
```

在[此处](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)下载[faster_rcnn_resnet101_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz)解压放在目录`20181227`

此时目录结构为
```python
20181227/
├── faster_rcnn_resnet101_coco_2018_01_28
│   ├── checkpoint
│   ├── frozen_inference_graph.pb
│   ├── model.ckpt.data-00000-of-00001
│   ├── model.ckpt.index
│   ├── model.ckpt.meta
│   ├── pipeline.config
│   └── saved_model
│       ├── saved_model.pb
│       └── variables
├── faster_rcnn_resnet101_voc07.config
├── pascal_label_map.pbtxt
├── pascal_train.record
└── pascal_val.record
```
修改`faster_rcnn_resnet101_voc07.config`配置

```python
将PATH_TO_BE_CONFIGURED 都替换成 20181227

第 110 和 111 行内容改为：（表示不微调，从头开始训练模型）
# fine_tune_checkpoint: "20170820/model.ckpt"  （注释掉）
from_detection_checkpoint: false  （设为false）

# or 如果微调，则修改成
fine_tune_checkpoint: "20181227/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt"
from_detection_checkpoint: true

# or 如果要接着自己训练的模型继续训练（如：中途中断了）
fine_tune_checkpoint: "20181227/model/model.ckpt-1102" # 自己训练的模型保存位置
from_detection_checkpoint: true
```

运行以下命令进行train

```python
cd ../ # (cd ./models/resarch)

# From the tensorflow/models/research/
PIPELINE_CONFIG_PATH='./20181227/faster_rcnn_resnet101_voc07.config'
MODEL_DIR='./20181227/model' # 训练保存的model 位置
NUM_TRAIN_STEPS=50000
NUM_EVAL_STEPS=2000
python3 object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --alsologtostderr
```

参考：https://github.com/tensorflow/models/issues/4856

如果训练过程出现`TypeError: can't pickle dict_values objects`，需修改https://github.com/tensorflow/models/blob/master/research/object_detection/model_lib.py#L390  将`category_index.values()`改成`list(category_index.values())`