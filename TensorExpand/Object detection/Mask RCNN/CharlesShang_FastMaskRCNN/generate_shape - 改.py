# -*- coding:utf-8 -*-

import numpy as np
import cv2
import random
import math
import matplotlib.pyplot as plt

def random_shape( height, width):
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

def mask2box(mask):
    '''从mask反算出其边框
    mask：[h,w]  0、1组成的图片
    1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
    '''
    # np.where(mask==1)
    index=np.argwhere(mask==1)
    rows=index[:,0]
    clos=index[:,1]
    # 解析左上角行列号
    left_top_r=np.min(rows) # y
    left_top_c = np.min(clos) # x

    # 解析右下角行列号
    right_bottom_r=np.max(rows)
    right_bottom_c = np.max(clos)

    # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
    # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
    return [left_top_c, left_top_r, right_bottom_c, right_bottom_r] # [x1,y1,x2,y2]

def random_image( height, width):
    # Pick random background color
    bg_color = np.array([random.randint(0, 255) for _ in range(3)])
    # images=[]
    mask=[]
    class_id=[]

    bg_color = bg_color.reshape([1, 1, 3])
    image = np.ones([height, width, 3], dtype=np.uint8)
    image = image * bg_color.astype(np.uint8)

    N = random.randint(1, 4)
    for _ in range(N):
        image_mask = np.zeros([height, width], dtype=np.uint8)
        shape, color, dims = random_shape(height, width)
        x, y, s = dims
        if shape == 'square':
            cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
            cv2.rectangle(image_mask, (x - s, y - s), (x + s, y + s), 1, -1) # 0、1组成的图片
            # images.append(img)
            mask.append(image_mask)
            class_id.append(1) # 对应class ID 1

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
    mask=np.asarray(mask,np.uint8).transpose([1, 2, 0]) # [h,w,instance count]
    class_id=np.asarray(class_id,np.uint8) # [instance count,]

    # Handle occlusions 处理遮挡情况
    count = mask.shape[-1]
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count - 2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # 如果mask 全为0 也就是意味着完全被遮挡，需丢弃这种mask，否则训练会报错
        # （而实际标准mask时不会出现这种情况的，因为完全遮挡了没办法标注mask）
        if np.sum(mask[:, :, i])<1: # 完全被遮挡
            mask=np.delete(mask,i,axis=-1)
            class_id=np.delete(class_id,i) # 对应的mask的class id 也需删除

    bboxes=[] # [instance count,4]
    [bboxes.append(mask2box(mask[:,:,i])) for i in range(count)]
    bboxes = np.asarray(bboxes)
    gt_boxes = np.hstack((bboxes, class_id[:, np.newaxis])) # [instance count,5] 前4列为boxs，最后一列为 class id

    mask_ = np.zeros((height, width), dtype=np.float32) # [h,w]
    for i in range(count):
        mask_+=mask[:,:,i]*class_id[i]

    mask = np.asarray(mask, np.uint8).transpose([2, 0,1])  # [instance count,h,w]

    return image,gt_boxes,mask,mask_

image,gt_boxes,mask,mask_=random_image(128,128)

print(image.shape)
print(gt_boxes.shape)
print(mask.shape)
print(mask_.shape)

plt.subplot(121)
plt.imshow(image)

plt.subplot(122)
plt.imshow(mask_)

plt.show()

