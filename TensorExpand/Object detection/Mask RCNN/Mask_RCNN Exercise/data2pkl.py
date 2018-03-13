# -*- coding:utf8 -*-

import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

base_path=os.getcwd()

# 调整图片大小
h,w=512,512 # 64*8 Mask rcnn 要求图像必须 大小64*n，波段数为3

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


class_all_name=[]
analyze_xml_class(glob.glob(os.path.join(Annotations_path,'*.xml')),class_all_name)
class_set=set(class_all_name) # 转成set去除重复的
class_all_name=None
sorted(class_set) # 排序

class_dict=dict(zip(class_set,range(1,len(class_set)+1))) # 类别从1开始，0 默认为背景
class_set=None
# 按key值排序
# sorted(class_dict.keys())

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
        x_=(rectangle[0]+rectangle[2])//2
        y_=(rectangle[1]+rectangle[3])//2
        pixels = mask_1[y_, x_]
        mask=(mask==pixels).astype(np.uint8)

        # 统一大小 64*n
        mask=cv2.resize(mask,(h,w))

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
    image = cv2.resize(image, (h, w))/255.  # 转到0.~1.

    '''
    # 可视化结果
    num_masks=masks.shape[-1]
    for i in range(num_masks):
        plt.subplot(1,num_masks+1,i+2)
        plt.imshow(masks[:,:,i],'gray')
        plt.axis('off')

    plt.subplot(1, num_masks + 1, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    '''


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

