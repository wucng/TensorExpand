参考：

- [使用Keras和Tensorflow设置和安装Mask RCNN](http://blog.csdn.net/wc781708249/article/details/79438972) 
- [Mask R-CNN](https://github.com/matterport/Mask_RCNN) for object detection and instance segmentation on Keras and TensorFlow


----------
[toc]


----------
# MaskRCNN识别Pascal VOC 2007

完整程序在[这里](https://github.com/fengzhongyouxia/TensorExpand/tree/master/TensorExpand/Object%20detection/Mask%20RCNN/Mask_RCNN%20Exercise)

------

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

# 解析出image，mask，class_name(class_id)

1、解析xml文件
```python
import os

file_name=r'C:\Users\Administrator\Desktop\009901.xml'

fp=open(file_name)

for p in fp:
    if '<object>' in p:
        print(next(fp).split('>')[1].split('<')[0])
    if '<bndbox>' in p:
        print(next(fp).split('>')[1].split('<')[0])
        print(next(fp).split('>')[1].split('<')[0])
        print(next(fp).split('>')[1].split('<')[0])
        print(next(fp).split('>')[1].split('<')[0])

```

```python
'''
person
104
76
199
337
……
```

2、从xml中解析出class，及矩形位置

```python
def analyze_xml(file_name):
    '''
    从xml文件中解析class，对象位置
    :param file_name: xml文件位置
    :return: class，每个类别的矩形位置
    '''
    fp=open(file_name)

    class_name=[]

    rectangle_position=[]

    for p in fp:
        if '<object>' in p:
            class_name.append(next(fp).split('>')[1].split('<')[0])

        if '<bndbox>' in p:
            rectangle = []
            [rectangle.append(int(next(fp).split('>')[1].split('<')[0])) for _ in range(4)]

            rectangle_position.append(rectangle)

    # print(class_name)
    # print(rectangle_position)

    fp.close()

    return class_name,rectangle_position
```

3、解析所有的类别（生成class id）

```python
def analyze_xml_class(file_names,class_name = []):
    '''解析xml的所有类别'''
    for file_name in file_names:
        with open(file_name) as fp:
            for p in fp:
                if '<object>' in p:
                    class_name.append(next(fp).split('>')[1].split('<')[0])
       
```
4、将每张图片的image，mask，class_id 存放到pickle文件中，便于后续加载

```python
for num,path in enumerate(Object_path):

    # 进度输出
    sys.stdout.write('\r>> Converting image %d/%d' % (
        num + 1, len(Object_path)))
    sys.stdout.flush()

    file_name=path.split('/')[-1].split('.')[0]
    Annotations_path_=os.path.join(Annotations_path,file_name+'.xml') # 对应的xml文件
    class_name,rectangle_position=analyze_xml(Annotations_path_)

    # 解析对象的mask[h,w,m] m为对象的个数，0、1组成的1波段图像
    mask_1=cv2.imread(path,0)

    masks=[]
    for rectangle in rectangle_position:
        mask=np.zeros_like(mask_1,np.uint8)
        mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = mask_1[rectangle[1]:rectangle[3],
                                                                      rectangle[0]:rectangle[2]]

        # 计算矩形中点像素值
        mean_x=(rectangle[0]+rectangle[2])//2
        mean_y=(rectangle[1]+rectangle[3])//2

        end=min((mask.shape[1],int(rectangle[2])+1))
        start=max((0,int(rectangle[0])-1))

        flag=True
        for i in range(mean_x,end):
            x_=i;y_=mean_y
            pixels = mask_1[y_, x_]
            if pixels!=0 and pixels!=220: # 0 对应背景 220对应边界线
                mask=(mask==pixels).astype(np.uint8)
                flag=False
                break
        if flag:
            for i in range(mean_x,start,-1):
                x_ = i;y_ = mean_y
                pixels = mask_1[y_, x_]
                if pixels != 0 and pixels != 220:
                    mask = (mask == pixels).astype(np.uint8)
                    break

        # 统一大小 64*n
        # mask=cv2.resize(mask,(h,w)) # 后面进行统一缩放

        masks.append(mask)
    # mask转成[h,w,m]格式
    masks=np.asarray(masks,np.uint8).transpose([1,2,0]) # [h,w,m]
    # class name 与class id 对应
    class_id=[]
    [class_id.append(class_dict[i]) for i in class_name]
    class_id=np.asarray(class_id,np.uint8) # [m,]

    mask_1=None

    # images 原图像
    image = cv2.imread(os.path.join(Image_path, file_name + '.jpg'))
    # image = cv2.resize(image, (h, w)) # /255.  # 不需要转到0.~1. 程序内部会自动进行归一化处理

    # 图像与mask都进行缩放
    image, _, scale, padding=resize_image(image, min_dim=IMAGE_MIN_DIM, max_dim=IMAGE_MAX_DIM, padding=True)
    masks=resize_mask(masks, scale, padding)

    '''
    # 可视化结果
    num_masks=masks.shape[-1]
    for i in range(num_masks):
        plt.subplot(1,num_masks+1,i+2)
        plt.imshow(masks[:,:,i],'gray')
        plt.axis('off')
        plt.title(class_name_dict[class_id[i]])

    plt.subplot(1, num_masks + 1, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    # '''

    object_data.append([image,masks,class_id])
    if num>0 and num%200==0:
        with open('./data/data_'+str(num)+'.pkl','wb') as fp:
            pickle.dump(object_data,fp)
            object_data=[]
            object_data.append([class_dict])

    if num==len(Object_path)-1 and object_data!=None:
        with open('./data/data_' + str(num) + '.pkl', 'wb') as fp:
            pickle.dump(object_data, fp)
            object_data = None

sys.stdout.write('\n')
sys.stdout.flush()

```

# 重写ShapesConfig类

```python
class ShapesConfig(Config):
    # 命名配置
    NAME = "shapes"

    # 输入图像resing
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # 使用的GPU数量。 对于CPU训练，请使用1
    GPU_COUNT = 1

    IMAGES_PER_GPU = int(1024 * 1024 * 4 // (IMAGE_MAX_DIM * IMAGE_MAX_DIM * 12))+1

    batch_size = GPU_COUNT * IMAGES_PER_GPU
    STEPS_PER_EPOCH = int(train_images / batch_size * (3 / 4))

    VALIDATION_STEPS = STEPS_PER_EPOCH // (1000 // 50)

    NUM_CLASSES = 1 + 20  # 必须包含一个背景（背景作为一个类别）

    scale = 1024 // IMAGE_MAX_DIM
    RPN_ANCHOR_SCALES = (32 // scale, 64 // scale, 128 // scale, 256 // scale, 512 // scale)  # anchor side in pixels

    RPN_NMS_THRESHOLD = 0.6  # 0.6

    RPN_TRAIN_ANCHORS_PER_IMAGE = 256 // scale

    MINI_MASK_SHAPE = (56 // scale, 56 // scale)

    TRAIN_ROIS_PER_IMAGE = 200 // scale

    DETECTION_MAX_INSTANCES = 100 * scale * 2 // 3

    DETECTION_MIN_CONFIDENCE = 0.6
```

# 重写ShapesDataset类

```python
class ShapesDataset(utils.Dataset):

    def __init__(self,pkl_path,class_map=None):
        super(ShapesDataset, self).__init__(class_map)
        self.pkl_path=pkl_path

    def load_shapes(self, count, height, width):
        data=self.load_pkl()
        class_dict=data[0][0] # 所有类别字典 从1开始
        # self.add_class("shapes", 0, 'BG') # 标签0默认为背景 utils中已经添加了 这里不用再添加

        # self.add_class("shapes", 1, "square")
        # self.add_class("shapes", 2, "circle")
        # self.add_class("shapes", 3, "triangle")
        # 必须从1、2、3、……开始添加，否则会出错 ，下面这种情况会导致标签对不上
        # [self.add_class('shapes',class_dict[i],i) for i in list(class_dict.keys())] # 无序添加会导致标签对应不上

        # class id反算出class name
        class_name_dict = dict(zip(class_dict.values(), class_dict.keys()))

        [self.add_class('shapes',i,class_name_dict[i]) for i in range(1,21)] # 共20类


        '''
        for i in range(count):
            comb_image, mask, class_ids = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           image=comb_image, mask=mask, class_ids=class_ids)
        '''

        for i in range(1,len(data)):
            comb_image=data[i][0]
            mask=data[i][1]
            class_ids=data[i][2]
            self.add_image("shapes", image_id=i-1, path=None,
                           width=height, height=width,
                           image=comb_image, mask=mask, class_ids=class_ids)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = info['image']
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = info['mask']
        class_ids = info['class_ids']
        return mask, class_ids.astype(np.int32)

    '''
    def random_shape(self,height, width):
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self,height, width):
        # Pick random background color
        bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        # images=[]
        mask = []
        class_id = []

        bg_color = bg_color.reshape([1, 1, 3])
        image = np.ones([height, width, 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)

        N = random.randint(1, 4)
        for _ in range(N):
            image_mask = np.zeros([height, width], dtype=np.uint8)
            shape, color, dims = self.random_shape(height, width)
            x, y, s = dims
            if shape == 'square':
                cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
                cv2.rectangle(image_mask, (x - s, y - s), (x + s, y + s), 1, -1)  # 0、1组成的图片
                # images.append(img)
                mask.append(image_mask)
                class_id.append(1)  # 对应class ID 1

            elif shape == "circle":
                cv2.circle(image, (x, y), s, color, -1)
                cv2.circle(image_mask, (x, y), s, 1, -1)
                # images.append(img)
                mask.append(image_mask)
                class_id.append(2)  # 对应class ID 2

            elif shape == "triangle":
                points = np.array([[(x, y - s),
                                    (x - s / math.sin(math.radians(60)), y + s),
                                    (x + s / math.sin(math.radians(60)), y + s),
                                    ]], dtype=np.int32)
                cv2.fillPoly(image, points, color)
                cv2.fillPoly(image_mask, points, 1)
                # images.append(img)
                mask.append(image_mask)
                class_id.append(3)  # 对应class ID 3

        # images=np.asarray(images,np.float32) # [h,w,c]
        mask = np.asarray(mask, np.uint8).transpose([1, 2, 0])  # [h,w,instance count]
        class_id = np.asarray(class_id, np.uint8)  # [instance count,]

        # Handle occlusions 处理遮挡情况
        count = mask.shape[-1]
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            # 如果mask 全为0 也就是意味着完全被遮挡，需丢弃这种mask，否则训练会报错
            # （而实际标注mask时不会出现这种情况的，因为完全遮挡了没办法标注mask）
            if np.sum(mask[:, :, i]) < 1:  # 完全被遮挡
                mask = np.delete(mask, i, axis=-1)
                class_id = np.delete(class_id, i)  # 对应的mask的class id 也需删除
        return image, mask, class_id
    '''
    def load_pkl(self):
        with open(self.pkl_path, 'rb') as fp:
            data = pickle.load(fp)
        return data
```

# 最终结果

mAP:  0.6333333358168602

![这里写图片描述](http://img.blog.csdn.net/20180313163353833?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](http://img.blog.csdn.net/20180313163405498?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)