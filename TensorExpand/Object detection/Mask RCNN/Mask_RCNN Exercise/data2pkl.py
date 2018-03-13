# -*- coding:utf8 -*-

import cv2
import glob
import os
import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
import pickle
import sys

base_path=os.getcwd()

# 调整图片大小
# h,w=512,512 # 64*8 Mask rcnn 要求图像必须 大小64*n，波段数为3
IMAGE_MIN_DIM =512
IMAGE_MAX_DIM =512


Object_path=glob.glob('./VOCdevkit/VOC2007/SegmentationObject/*.png')
np.random.shuffle(Object_path) # 数据随机打乱

Image_path='./VOCdevkit/VOC2007/JPEGImages'
Annotations_path='./VOCdevkit/VOC2007/Annotations'

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

def analyze_xml_class(file_names,class_name = []):
    '''解析xml的所有类别'''
    for file_name in file_names:
        with open(file_name) as fp:
            for p in fp:
                if '<object>' in p:
                    class_name.append(next(fp).split('>')[1].split('<')[0])


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


class_all_name=[]
analyze_xml_class(glob.glob(os.path.join(Annotations_path,'*.xml')),class_all_name)
class_set=set(class_all_name) # 转成set去除重复的
class_all_name=None
class_list=sorted(list(class_set)) # 排序
class_set=None

class_dict=dict(zip(class_list,range(1,len(class_list)+1))) # 类别从1开始，0 默认为背景
# 按key值排序
# sorted(class_dict.keys())

# class id反算出class name
class_name_dict=dict(zip(class_dict.values(),class_dict.keys()))


object_data=[]
object_data.append([class_dict])

# 生成的pickle数据存放在data文件下
if not os.path.exists('./data'):
    os.mkdir('./data')

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

