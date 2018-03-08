#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    class_name=lbl_names[1:] # 不需要包含背景 lbl_names[0] 默认为背景

    plt.subplot(221)
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(222)
    plt.imshow(lbl_viz)
    plt.axis('off')

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
