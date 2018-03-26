参考：

-  [hizhangp/yolo_tensorflow](hizhangp/yolo_tensorflow)
- [YOLO_v1 的 TensorFlow 源码分析](https://blog.csdn.net/qq_34784753/article/details/78803423)


----------
# YOLO_tensorflow
[YOLO](https://arxiv.org/pdf/1506.02640.pdf)的Tensorflow实施，包括训练和测试阶段。

# Installation
- Clone yolo_tensorflow repository

```
$ git clone https://github.com/hizhangp/yolo_tensorflow.git
$ cd yolo_tensorflow
```
- 下载Pascal VOC数据集，并创建正确的目录

```
$ ./download_data.sh
```
- 或
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

```
mv VOCdevkit data/pascal_voc/
```
- Download [YOLO_small](https://drive.google.com/file/d/0B5aC8pI-akZUNVFZMmhmcVRpbTA/view?usp=sharing) weight file and put it in `data/weights`

- Modify configuration in `yolo/config.py`

- Training


修改 `train.py` 140行 `default='0'` 表示使用GPU

修改 `train.py` 80行
```
log_str = '{} Epoch: {}, Step: {}, Learning rate: {}, \
Loss: {:5.3f}\nSpeed: {:.3f}s/iter, Load: {:.3f}s/iter, \
Remain: {}'.format(
	datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
	self.data.epoch,
	int(step),
	round(self.learning_rate.eval(session=self.sess), 6),
	loss,
	train_timer.average_time,
	load_timer.average_time,
	train_timer.remain(step, self.max_iter))
```

修改 `yolo/config.py` 里的文件配置， 如：62行 `BATCH_SIZE = 45` 设置小点（GPU内存不足）


```
$ python train.py
```

- Test

```
python test.py
```
