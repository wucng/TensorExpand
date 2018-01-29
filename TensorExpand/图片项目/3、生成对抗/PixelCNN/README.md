参考：[15、生成妹子图（PixelCNN）](https://github.com/fengzhongyouxia/TensorExpand/tree/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/15%E3%80%81%E7%94%9F%E6%88%90%E5%A6%B9%E5%AD%90%E5%9B%BE%EF%BC%88PixelCNN%EF%BC%89)


参考：http://blog.topspeedsnail.com/archives/10660


----------


本文使用TensorFlow实现论文《[Conditional Image Generation with PixelCNN Decoders](https://arxiv.org/abs/1606.05328)》，它是基于PixelCNN架构的模型，最早出现在《[Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759)》一文。

# 使用的图片数据
《[OpenCV之使用Haar Cascade进行对象识别](http://blog.topspeedsnail.com/archives/10511)》

- https://pan.baidu.com/s/1kVSA8z9 (密码: atqm)
- https://pan.baidu.com/s/1ctbd9O (密码: kubu)

# 数据预处理

```python
import os
 
old_dir = 'images'
new_dir = 'girls'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
 
count = 0
for (dirpath, dirnames, filenames) in os.walk(old_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            new_filename = str(count) + '.jpg'
            os.rename(os.sep.join([dirpath, filename]), os.sep.join([new_dir, new_filename])) # 剪切到新目录并重命名
            print(os.sep.join([dirpath, filename]))
            count += 1
print("Total Picture: ", count)
```

使用《[open_nsfw: 基于Caffe的成人图片识别模型](http://blog.topspeedsnail.com/archives/9440)》剔除掉和妹子图不相关的图片，给open_nsfw输入要检测的图片，它会返回图片评级（0-1），等级越高，图片越黄越暴力。使用OpenCV应该也不难。

为了减小计算量，我把图像缩放为64×64像素：

```python
import os
import cv2
import numpy as np
 
image_dir = 'girls'
new_girl_dir = 'little_girls'
if not os.path.exists(new_girl_dir):
    os.makedirs(new_girl_dir)
 
for img_file in os.listdir(image_dir):
    img_file_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_file_path)
    if img is None:
        print("image read fail")
        continue
    height, weight, channel = img.shape
    if height < 200 or weight < 200 or channel != 3: 
        continue
    # 你也可以转为灰度图片(channel=1)，加快训练速度
    # 把图片缩放为64x64
    img = cv2.resize(img, (64, 64))
    new_file = os.path.join(new_girl_dir, img_file)
    cv2.imwrite(new_file, img)
    print(new_file)
```
去除重复图片：

```python
import os
import cv2
import numpy as np


# 判断两张图片是否完全一样（使用哈希应该要快很多）
def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1 is None or img2 is None:
        return False
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False


# 去除重复图片
file_list = os.listdir('little_girls')
try:
    for img1 in file_list:
        print(len(file_list))
        for img2 in file_list:
            if img1 != img2:
                if is_same_image('little_girls/' + img1, 'little_girls/' + img2) is True:
                    print(img1, img2)
                    os.remove('little_girls/' + img1)
        file_list.remove(img1)
except Exception as e:
    print(e)
```

# PixelCNN生成妹纸图完整代码

下面代码只实现了unconditional模型（无条件），没有实现conditional和autoencoder模型。详细信息，请参看论文。



# 卷积层基本结构
![这里写图片描述](http://img.blog.csdn.net/20180119142449603?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 流程图
![这里写图片描述](http://img.blog.csdn.net/20180119151320314?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


补充练习：使用[OpenCV提取图像中的脸](http://blog.topspeedsnail.com/archives/10511)，然后使用上面模型进行训练，看看能生成什么。

- [Deep Generative Image Models using a Laplacian Pyramid of Adversarial Networks](https://arxiv.org/pdf/1506.05751v1.pdf)
- https://github.com/awentzonline/image-analogies
- https://github.com/ericjang/draw
