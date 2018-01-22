参考：http://blog.topspeedsnail.com/archives/10685


----------
[下载模型](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)

解压模型：`tar -zxf inception-2015-12-05.tgz /tmp/imagenet`

在retrain自己的图像分类器之前，我们先来测试一下Google的Inception模型：

[classify_image.py](https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py)

```python
$ python3 classify_image.py --model_dir /tmp/imagenet --image_file bigcat.jpg
# model_dir 模型存放的位置
# image_file 影像存放位置
# 自动下载100多M的模型文件
# 参数的解释, 查看源码文件
```


----------


- https://www.tensorflow.org/tutorials/image_recognition/
- https://tensorflow.google.cn/tutorials/image_retraining


使用examples中的[image_retraining](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py)。

训练：

```python
$ python3 retrain.py --how_many_training_steps 4000 --model_dir /tmp/imagenet --output_graph output_graph.pb --output_labels output_labels.txt --image_dir ./hymenoptera_data/train/

# ls hymenoptera_data/train/
# >>> ants  bees (对应图像的标签)
```
参数解释参考[retrain.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py)源文件。

注： 如果出现 `from tensorflow.contrib.quantize.python import quant_ops`导入失败
可以去复制[quant_ops.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/python/quant_ops.py) 放置在retrain.py同目录下，并修改成`import quant_ops`即可

大概训练了半个小时：
![这里写图片描述](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7-2016-11-29-%E4%B8%8B%E5%8D%8812.48.19.png)

生成的模型文件和labels文件：
![这里写图片描述](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7-2016-11-29-%E4%B8%8B%E5%8D%8812.49.32.png)

使用训练好的模型：

```python
python3 test.py ./hymenoptera_data/val/ants/Hormiga.jpg
```


