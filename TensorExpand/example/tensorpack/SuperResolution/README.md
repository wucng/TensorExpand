[SuperResolution - EnhanceNet](https://github.com/tensorpack/tensorpack/tree/master/examples/SuperResolution)

给定低分辨率图像，训练网络以使用不同的损失函数产生4x分辨率图像。

![这里写图片描述](https://github.com/tensorpack/tensorpack/raw/master/examples/SuperResolution/enhancenet-demo.jpg)

- 左：输入图像（使用双三次插值进行放大）。
- 中：使用作者的实现（仅包含生成器）
- 右：这个实现（带有训练代码）

---
1、下载MS COCO数据集和VGG19模型

```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://models.tensorpack.com/caffe/vgg19.npz
```
2、使用以下方法训练EnhanceNet-PAT：

```python
python enet-pat.py --vgg19 /path/to/vgg19.npz --data train2017.zip

# or: convert to an lmdb first and train with lmdb:
python data_sampler.py --lmdb train2017.lmdb --input train2017.zip --create
python enet-pat.py --vgg19 /path/to/vgg19.npz --data train2017.lmdb
```
训练非常不稳定，通常不会产生良好的效果。 预训练模型也可能在不同类型的图像上失败。 您可以在[此处](http://models.tensorpack.com/SuperResolution/)下载并使用预训练模型。

3、对当前目录中的图像和输出进行推断：

```
python enet-pat.py --apply --load /path/to/model --lowres input.png
```