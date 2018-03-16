参考：

-  [MarvinTeichmann/tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn)
- [shekkizh/FCN.tensorflow](https://github.com/shekkizh/FCN.tensorflow)


----------

完整代码[点击此处](https://github.com/fengzhongyouxia/TensorExpand/tree/master/TensorExpand/Object%20detection/FCN/shekkizh_FCN.tensorflow/FCN-tensorflow)

----------


# Pascal VOC 2007数据下载

```python
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```

```python
# 执行以下命令将解压到一个名为VOCdevkit的目录中
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```
## 数据预览
1、VOC2007/Annotations

类别名与对象的矩形框位置

2、VOC2007/JPEGImages

![这里写图片描述](http://img.blog.csdn.net/20180312111000760?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

3、VOC2007/SegmentationClass

![这里写图片描述](http://img.blog.csdn.net/20180312111028031?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

4、VOC2007/SegmentationObject

![这里写图片描述](http://img.blog.csdn.net/20180312111039831?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


----------
# Pascal_VOC_2007_data.py
解析图片与mask路径

```python
class ShapesDataset(object):

    def __init__(self, SegmentationClass='./VOCdevkit/VOC2007/SegmentationClass/*.png',
                 JPEGImages='./VOCdevkit/VOC2007/JPEGImages'):
        '''
        :param SegmentationClass: Mask 路径
        :param JPEGImages: JPG路径 
        '''
        self.image_info = []
        self.SegmentationClass=SegmentationClass
        self.JPEGImages=JPEGImages
        # self.height=224
        # self.width = 224

    def add_image(self, f, annotation_file, **kwargs):
        image_info = {
            "image": f, # JPG 路径
            "annotation": annotation_file, # 对应mask路径  
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def load_shapes(self,count):
        '''加载图像和对应的Mask的路径（没有事先解析成numpy）'''
        Class_path = glob.glob(self.SegmentationClass)
        np.random.shuffle(Class_path)  # 数据随机打乱
        for num, path in enumerate(Class_path):
            # 进度输出
            sys.stdout.write('\r>> Converting image %d/%d' % (
                num + 1, len(Class_path)))
            sys.stdout.flush()

            file_name = path.split('/')[-1].split('.')[0]
            # 对应JPG
            image_path = os.path.join(self.JPEGImages, file_name + '.jpg')

            self.add_image(f=image_path, annotation_file=path)
            if (num+1)==count:
                break

        sys.stdout.write('\n')
        sys.stdout.flush()
```
# BatchDatsetReader.py
修改的地方

```python
# self.annotations = np.array(
#[np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files]) # [h,w,1]

        self.annotations = np.array([self._transform(filename['annotation']) for filename in self.files]) # [h,w,3]
        self.annotations=np.expand_dims(np.argmax(self.annotations,axis=-1),-1) # [h,w,1]
```

# FCN_Pascal_VOC_2007_data.py
基本上是`FCN_shape_data.py`

修改的地方

```python
NUM_OF_CLASSESS = 21 # 因为Pascal_VOC_2007是20个类，加上背景 21

# 其他的基本不变
```
# train

```python
python3 FCN_Pascal_VOC_2007_data.py
```
# test
```python
python3 FCN_Pascal_VOC_2007_data.py --mode visualize
```

