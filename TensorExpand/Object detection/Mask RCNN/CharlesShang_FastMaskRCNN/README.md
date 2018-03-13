参考：

- [CharlesShang/FastMaskRCNN](https://github.com/CharlesShang/FastMaskRCNN)


----------
Mask RCNN in TensorFlow

# 安装
默认使用python2（python3 会报错）

1、克隆git
```python
git clone https://github.com/CharlesShang/FastMaskRCNN.git
```
2、 进入 `./libs/datasets/pycocotools` ， 运行 `make`

```python
# 改成python3 能编译 但后续还是会报错
默认是python2编译，修改Makefile文件，`python==>python3` 这样便可以使用python3编译
```


3、下载[COCO](http://cocodataset.org/#download)数据集，将其放入`./data`，然后运行
`cd ./FastMaskRCNN `
`python download_and_convert_data.py`来构建 tf-records。 这需要一段时间。

4、下载pretrained resnet50模型，
`wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz`
解压缩，放入`./data/pretrained_models/`

5、转到`./libs`并运行`make`

```python
# 改成python3 能编译 但后续还是会报错
python3 需修改Makefile  python==>python3
setup.py 84行 加上括号
50行 cudaconfig.iteritems() ==>cudaconfig.items()
```

6、运行`python ./train/train.py`进行训练

7、肯定有一些错误，请回报给我们，让我们一起解决它们。


------




# Mask RCNN
Mask RCNN in TensorFlow

This repo attempts to reproduce this amazing work by Kaiming He et al. : 
[Mask R-CNN](https://arxiv.org/abs/1703.06870)

## Requirements

- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [COCO dataset](http://mscoco.org/dataset/#download)
- [Resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

## How-to
1. Go to `./libs/datasets/pycocotools` and run `make`
2. Download [COCO](http://mscoco.org/dataset/#download) dataset, place it into `./data`, then run `python download_and_convert_data.py` to build tf-records. It takes a while.
3. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz`, unzip it, place it into `./data/pretrained_models/`
4. Go to `./libs` and run `make`
5. run `python train/train.py` for training
6. There are certainly some bugs, please report them back, and let's solve them together.

## TODO:
- [x] ROIAlign
- [x] COCO Data Provider
- [x] Resnet50
- [x] Feature Pyramid Network
- [x] Anchor and ROI layer
- [x] Mask layer
- [x] Speedup anchor layer with cython
- [x] Combining all modules together.
- [x] Testing and debugging (in progress)
- [ ] Training / evaluation on COCO
- [ ] Add image summary to show some results
- [ ] Converting ResneXt
- [ ] Training >2 images

## Call for contributions
- Anything helps this repo, including **discussion**, **testing**, **promotion** and of course **your awesome code**.

## Acknowledgment
This repo borrows tons of code from 
- [TFFRCNN](https://github.com/CharlesShang/TFFRCNN)
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) 
- [faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- [tf-models](https://github.com/tensorflow/models)

## License
See [LICENSE](https://github.com/CharlesShang/FastMaskRCNN/blob/master/LICENSE) for details.

