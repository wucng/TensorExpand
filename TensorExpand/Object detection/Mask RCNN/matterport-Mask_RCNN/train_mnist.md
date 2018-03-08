参考：

- [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN)
- [使用Keras和Tensorflow设置和安装Mask RCNN](http://blog.csdn.net/wc781708249/article/details/79438972)
- [train_shapes.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/train_shapes.ipynb)


----------
[toc]

完整代码在[这里](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/Object%20detection/Mask%20RCNN/matterport-Mask_RCNN/train_shape_simplify.py)，如果觉得有用，可以点击star，谢谢支持！

通过改写train_shapes.ipynb中的代码实现训练mnist数据
只需修改ShapesConfig与ShapesDataset类

# 注意：
1、传入的图片大小必须是2^6的倍数，如：64,128,256……(64*n n=1,2,3,……)，且必须是3波段，如果进入`config.py`修改波段，后面运算会报错（不解）

2、图片对应的掩膜图片大小为对应原图的大小，背景像素为0，对象内的像素为1 即0、1像素图片，图像shape[h,w,3] ;mask shape [h,w,instance count]

3、mnist图片大小28x28x1，类别数0~9 10类 加上背景 共11类（这种模式[图像分割]必须算上背景作为一个类别）**注**：背景默认class为0，0~9 数字对应的标签为1~10

4、先将mnist图片从28x28 resize到32x32 ，再组成4x4的图片，新的图片大小32x4=128

5、masks:[height, width, instance count]  这里height=width=128，instance count=16（4x4每个格都有对象）

6、class_ids: a 1D array of class IDs of the instance masks. [instance count,] 要与mask对应起来

7、输入：图片[h,w,3]，图片每个对象的掩膜[h,w,instance count]，每个对象的class_id[instance count,] (从1开始标记，0默认为背景)

```python
# -*- coding: utf8 -*-
'''
注意：
1、传入的图片大小必须是2^6的倍数，如：64,128,256……(64*n n=1,2,3,……)
2、图片对应的掩膜图片大小为对应原图的大小，背景像素为0，对象内的像素为1 即0、1像素图片
3、mnist图片大小28x28x1，类别数0~9 10类 加上背景 共11类（这种模式[图像分割]必须算上背景作为一个类别）
4、先将mnist图片从28x28 resize到32x32 ，再组成4x4的图片，新的图片大小32x4=128
5、masks:[height, width, instance count]  这里height=width=128，instance count=16（4x4每个格都有对象）
6、class_ids: a 1D array of class IDs of the instance masks. [instance count,] 要与mask对应起来
'''
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets

mnist=read_data_sets('./MNIST_data',one_hot=False)
def Image_Processing(image):
    '''缩放到32x32，像素值转成0、1'''
    image=np.reshape(image,[28,28]) # [28,28]
    img=cv2.resize(image,(32,32)) # [32,32]

    img_mask = np.round(img) # 转成对应的掩膜 像素值0、1
    return img,img_mask

def random_comb(mnist):
    num_images=mnist.train.num_examples
    indexs=random.choices(np.arange(0,num_images),k=16) # 随机选择16个索引值
    indexs=np.asarray(indexs,np.uint8).reshape([4,4])
    class_ids = mnist.train.labels[indexs.flatten()]
    comb_image=np.zeros([32*4,32*4],np.float32)
    mask=[]
    for i in range(4):
        for j in range(4):
            image_mask = np.zeros([32 * 4, 32 * 4], np.uint8)
            img_data=mnist.train.images[indexs[i,j]]
            img, img_mask=Image_Processing(img_data)
            comb_image[i*32:(i+1)*32,j*32:(j+1)*32]=img
            image_mask[i*32:(i+1)*32,j*32:(j+1)*32]=img_mask
            mask.append(image_mask)

    return comb_image,np.asarray(mask).transpose([1,2,0]),class_ids

comb_image,mask,class_ids=random_comb(mnist)

print(class_ids)

plt.subplot(121)
plt.imshow(comb_image,'gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(mask[:,:,8],'gray')
plt.title(class_ids[8])
plt.axis('off')

plt.show()
```
class_ids ： [2 1 3 3 0 6 8 4 3 4 2 7 4 2 0 5]
![这里写图片描述](http://img.blog.csdn.net/20180307141258934?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 修改ShapesConfig
只修改了

```python
NUM_CLASSES = 1 + 10  # background + 0~9 shapes
```
其他的自行修改以适应自己的数据
# 修改ShapesDataset
这个是关键
需改写load_shapes，load_image，load_mask三个函数

## load_shapes

```python
    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images. 
        count: number of images to generate. 数量
        height, width: the size of the generated images. 尺寸
        """
        # Add classes 添加类别
        # self.add_class("shapes", 1, "square")
        # self.add_class("shapes", 2, "circle")
        # self.add_class("shapes", 3, "triangle")
        # self.add_class("shapes", 0, 'BG') # 标签0默认为背景
        [self.add_class("shapes", i+1, str(i)) for i in range(10)] # 0~9 对应标签1~10  ；标签0默认为背景

        # Add images
        for i in range(count):
            comb_image, mask, class_ids=self.random_comb(mnist,height,width)

            # 输入图片默认是3波段的
            comb_images=np.zeros([height,width,3],np.float32)
            comb_images[:,:,0]=comb_image
            comb_images[:, :, 1] = comb_image
            comb_images[:, :, 2] = comb_image
            comb_image=comb_images # [128,128,3]  转成3波段

            mask=np.asarray(mask).transpose([1, 2, 0]) # mask shape [128,128,16]
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           image=comb_image,mask=mask,class_ids=class_ids)
```

## load_image

```python
    def load_image(self, image_id):
        info = self.image_info[image_id]
        image=info['image']
        return image
```

## load_mask

```python
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        掩膜与类名对应起来"""

        info = self.image_info[image_id]
        mask=info['mask']
        class_ids=info['class_ids']
        return mask, class_ids.astype(np.int32)
```

其他地方不需修改

运行的结果：

![这里写图片描述](http://img.blog.csdn.net/20180308083620810?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180308083630683?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

mAP:  0.39781853563103564

识别的效果不是很理想，可以通过调节参数（如：学习率），迭代次数，样本数等 提升模型精度


# 使用labelme生成mask掩码数据集
参考：[Mask RCNN训练自己的数据集](http://blog.csdn.net/l297969586/article/details/79140840) 


----------
github地址：https://github.com/wkentaro/labelme 
安装方式：详情参考[官网](https://github.com/wkentaro/labelme)安装

```python
# Ubuntu 14.04
sudo apt-get install python-qt4 pyqt4-dev-tools
sudo pip install labelme # python2 works

# Ubuntu 16.04
sudo apt-get install python-qt5 pyqt5-dev-tools
sudo pip3 install labelme
```
启动命令	终端或cmd输入`labelme`

![这里写图片描述](http://img.blog.csdn.net/20180308085133453?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

注：每个对象对应一个mask（图中2个对象，对应2个mask）

Windows保存时会弹出以下错误（可能是我电脑问题吧）

![这里写图片描述](http://img.blog.csdn.net/20180308085441851?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

解决方法参考：
[Unable to save: 'bytes' object has no attribute](%27encode%27https://github.com/wkentaro/labelme/issues/25) 

参考[官网](https://github.com/wkentaro/labelme)重新安装


----------
## json2mask
编辑好的label会保存成json文件，接下来需要从该文件中解析出mask
参考 [labelme_draw_json](https://github.com/wkentaro/labelme/blob/master/scripts/labelme_draw_json)
重新改写：

```python
#!/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    data = json.load(open(json_file))

    img = utils.img_b64_to_array(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])

    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)

    # lbl_names[0] 默认为背景，对应的像素值为0
    # 解析图片中的对象 像素值不为0（0 对应背景）
    mask=[]
    class_id=[]
    for i in range(1,len(lbl_names)): # 跳过第一个class（默认为背景）
        mask.append((lbl==i).astype(np.uint8)) # 解析出像素值为1的对应，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
        class_id.append(i) # mask与clas 一一对应

    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0]) # 转成[h,w,instance count]
    class_id=np.asarray(class_id,np.uint8) # [instance count,]
    class_name=lbl_names[1:] # 不需要包含背景

    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(lbl_viz)

    plt.subplot(223)
    plt.imshow(mask[:,:,0],'gray')
    plt.title(class_name[0]+'\n id '+str(class_id[0]))
    plt.axis('off')

    plt.subplot(224)
    plt.imshow(mask[:,:,1],'gray')
    plt.title(class_name[1] + '\n id ' + str(class_id[1]))
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()

```

![这里写图片描述](http://img.blog.csdn.net/20180308102447751?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 使用arcgis画图像掩膜数据
（略过）

如果用过arcgis的知道，可以使用arcgis在图像上画shp文件，再用gdal转成mask

**遥感图像可以使用这种方法**，该方法需用到遥感图像中的坐标信息，一般的图片没法使用！

## shp2mask
![这里写图片描述](http://img.blog.csdn.net/20180308151005057?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


