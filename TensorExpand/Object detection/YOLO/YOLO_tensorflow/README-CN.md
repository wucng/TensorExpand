参考：

- [YOLO训练](http://blog.csdn.net/hrsstudy/article/details/65644517)
- [pjreddie/darknet](https://github.com/pjreddie/darknet)
- https://pjreddie.com/darknet/
- [gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- [longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)
- [mxcl/YOLOKit](https://github.com/mxcl/YOLOKit)
- [llSourcell/YOLO_Object_Detection](https://github.com/llSourcell/YOLO_Object_Detection)
- [experiencor/basic-yolo-keras](https://github.com/experiencor/basic-yolo-keras)
- [philipperemy/yolo-9000](https://github.com/philipperemy/yolo-9000)
- [nilboy/tensorflow-yolo](https://github.com/nilboy/tensorflow-yolo)
- [KJCracks/yololib](https://github.com/KJCracks/yololib)


----------
- [gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)


----------
YOLO：实时对象检测'的Tensorflow实现

# YOLO_tensorflow
## 1.Introduction
这是YOLO：实时对象检测的Tensorflow实现

它现在只能使用预训`YOLO_small`＆`YOLO_tiny`网络进行预测。

（YOLO人脸检测来自 https://github.com/quanhua92/darknet ）

我从darknet的（.weight）文件中提取了权重值。

我的代码不支持训练。 使用darknet进行训练。

原始代码（C实现）＆paper：http://pjreddie.com/darknet/yolo/


----------
## 2.Install
- （1）下载代码 git clone https://github.com/gliese581gg/YOLO_tensorflow.git

- （2）从YOLO下载权重文件

YOLO_small : https://drive.google.com/file/d/0B2JbaJSrWLpza08yS2FSUnV2dlE/view?usp=sharing

YOLO_tiny : https://drive.google.com/file/d/0B2JbaJSrWLpza0FtQlc3ejhMTTA/view?usp=sharing

YOLO_face : https://drive.google.com/file/d/0B2JbaJSrWLpzMzR5eURGN2dMTk0/view?usp=sharing

- （3）将下载的`YOLO_(version).ckpt`放入代码`weight`文件夹中


----------
## 3.Usage
 - （1）直接使用默认设置（在控制台上显示，显示输出图像，不输出文件写入）
 

```
python YOLO_(small or tiny)_tf.py -fromfile (input image filename)
```
- （2）直接使用自定义设置

```python
python YOLO_(small or tiny)_tf.py argvs

where argvs are

-fromfile (input image filename) : input image file
-disp_console (0 or 1) : whether display results on terminal or not
-imshow (0 or 1) : whether display result image or not
-tofile_img (output image filename) : output image file
-tofile_txt (output txt filename) : output text file (contains class, x, y, w, h, probability)
```
- （3）在其他脚本上导入

```
import YOLO_(small or tiny)_tf
yolo = YOLO_(small or tiny)_tf.YOLO_TF()

yolo.disp_console = (True or False, default = True)
yolo.imshow = (True or False, default = True)
yolo.tofile_img = (output image filename)
yolo.tofile_txt = (output txt filename)
yolo.filewrite_img = (True or False, default = False)
yolo.filewrite_txt = (True of False, default = False)

yolo.detect_from_file(filename)
yolo.detect_from_cvmat(cvmat)
```

## 4.Requirements
- Tensorflow
- Opencv2


----------
## 5.Copyright
根据原始代码的LICENSE文件，

- 我和原作者对任何损害不承担任何责任
- 不要在商业上使用它！


----------
## 6.Changelog
2016/02/15：首次上传！

2016/02/16：增加了YOLO_tiny，修复了当两个框都检测到有效对象时忽略网格中的一个框的错误

2016/08/26：上传的重量文件转换器！ （darknet weight -> tensorflow ckpt）

2017/02/21：增加了YOLO_face（感谢https://github.com/quanhua92/darknet）


----------


