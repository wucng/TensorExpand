参考：

- [Retinanet训练Pascal VOC 2007](http://blog.csdn.net/wc781708249/article/details/79643484#t7) 


----------

> 尝试使用普通的分类（类别分类）和回归（定位框）的方法解决问题

# 数据分析

参考[Retinanet训练Pascal VOC 2007](http://blog.csdn.net/wc781708249/article/details/79643484#t7) 将Pascal VOC 2007转成CSV数据

- CSV目录结构如下：

```python
<CSV>
|———— annotations.csv # 必须
|———— classes.csv # 必须
|
|____ JPEGImages  # 这样 annotations.csv可以使用图片的相对路径       
         └─ *.jpg
```
- `annotations.csv` 每行对应一个对象，格式如下：

```python
path/to/image.jpg,x1,y1,x2,y2,class_name

/data/imgs/img_001.jpg,837,346,981,456,cow
/data/imgs/img_002.jpg,215,312,279,391,cat # 02图片第一个对象
/data/imgs/img_002.jpg,22,5,89,84,bird # 02图片第二个对象
/data/imgs/img_003.jpg,,,,, # 背景图片，没有任何要检查的对象
```
- `classes.csv` 类名与id对应，索引从0开始。不要包含背景类，因为它是隐式的。具体格式如下：

```python
class_name,id

cow,0
cat,1
bird,2
# 因为这里的背景对应的类别为 空 ，
# 而mask RCNN 与 FCN（RFCN）都是使用0来表示背景
```
