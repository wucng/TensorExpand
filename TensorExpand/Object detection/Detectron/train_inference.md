<font size=4>

[toc]


# 设置数据集
参考：https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data

如果是`docker`，使用`-v`挂载到相应目录，如：

```
nvidia-docker run -it --name detectron -v `pwd`/work:/detectron/work -v `pwd`/COCO/:/detectron/detectron/datasets/data/coco aa21b75bb20c bash
```
# 配置文件修改（按GPU个数修改）
参考：https://github.com/facebookresearch/Detectron/issues/267#issuecomment-377339845
```python
# 1 GPU:
 BASE_LR: 0.0025
 MAX_ITER: 60000
 STEPS: [0, 30000, 40000]
# 2 GPUs:
 BASE_LR: 0.005
 MAX_ITER: 30000
 STEPS: [0, 15000, 20000]
# 4 GPUs:
 BASE_LR: 0.01
 MAX_ITER: 15000
 STEPS: [0, 7500, 10000]
# 8 GPUs:
 BASE_LR: 0.02
 MAX_ITER: 7500
 STEPS: [0, 3750, 5000]
```


# train(mask,bbox)
参考：https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md

```python
CUDA_VISIBLE_DEVICES=2,3 python2 tools/train_net.py \
    --cfg configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \ # 参数配置（详细信息要查看该文件）
    TRAIN.WEIGHTS R-101.pkl \ # 如果事先下载了可以在这指定，否则会在线下载
    OUTPUT_DIR /tmp/detectron-output \ # 模型输出
    NUM_GPUS 2 # 如果有多个GPU 只需修改这个，如2块 设置为2
```

<font size=4 color=##EE##>注：如果下载的`COCO` 数据集里面没有`coco_minival2014`目录，需新建一个，然后将要测试的图片放在该目录下即可，如果要在一个模型上做微调可以将`TRAIN.WEIGHTS`指定到最新的模型上（如：`e2e_faster_rcnn_R-50-FPN.pkl`），而不是最初的`R-101.pkl`(基于imagenet的模型)

```
coco
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ coco_minival2014
|_ ...
|_ annotations
   |_ instances_train2014.json
   |_ ...
```

修改`configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml`

```python
# from
MAX_ITER: 180000
STEPS: [0, 120000, 160000]

DATASETS: ('coco_2014_train', 'coco_2014_valminusminival')

# to
MAX_ITER: 18
STEPS: [0, 12, 16]

DATASETS: ('coco_2014_train', )
```
---
```python
CUDA_VISIBLE_DEVICES=2,3 python2 ../tools/train_net.py \
    --cfg ../configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml \
    TRAIN.WEIGHTS e2e_mask_rcnn_R-101-FPN_2x.pkl \
    OUTPUT_DIR ./result \
    NUM_GPUS 2 # 如果有多个GPU 只需修改这个，如2块 设置为2
```

# inference
使用上面训练的模型文件做推理
```python
python2 ../tools/infer_simple.py \
    --cfg ../configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml \ # 网络结构配置，必须与下面的wts对应
    --output-dir ./result \ # 图片保存的位置
    --image-ext jpg \ # 图片的后缀名
    --wts ./model_final.pkl \ # 在线提取的权重参数，如果预先下载好了，可以通过这里直接指定其路径
    --output-ext jpg \
    images # 要推理的图片目录（这里是`images/*.jpg`）
 
# 默认是保存成PDF格式，如果要改变保存的格式，如保存成.jpg,可以修改output-ext：
# --output-ext jpg
```
# keypoint训练
修改coco目录结构为如下形式：（`keypoints annotations` 在[此处](https://s3-us-west-2.amazonaws.com/detectron/coco/coco_annotations_minival.tgz)下载）
```
coco
|_ coco_train2014
|  |_ <im-1-name>.jpg
|  |_ ...
|  |_ <im-N-name>.jpg
|_ coco_val2014
|_ coco_minival2014
|_ ...
|_ annotations
   |_ person_keypoints_train2014.json
   |_ person_keypoints_val2014.json
   |_ person_keypoints_minival2014.json
   |_ person_keypoints_valminusminival2014.json
```
---
```python
CUDA_VISIBLE_DEVICES=2,3 python2 tools/train_net.py \
    --cfg configs/getting_started/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml \ # 参数配置（详细信息要查看该文件）
    TRAIN.WEIGHTS e2e_keypoint_rcnn_R-50-FPN_s1x.pkl \ # 如果事先下载了可以在这指定，否则会在线下载
    OUTPUT_DIR /tmp/detectron-output \ # 模型输出
    NUM_GPUS 2 # 如果有多个GPU 只需修改这个，如2块 设置为2

# -----------------------------
CUDA_VISIBLE_DEVICES=2,3 python2 ../tools/train_net.py \
    --cfg ../configs/getting_started/e2e_keypoint_rcnn_R-50-FPN_s1x.yaml \
    TRAIN.WEIGHTS e2e_keypoint_rcnn_R-50-FPN_s1x.pkl \
    OUTPUT_DIR ./result \
    NUM_GPUS 2
```