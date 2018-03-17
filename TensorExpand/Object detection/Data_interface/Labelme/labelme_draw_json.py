# -*- coding:utf-8 -*-
# !/usr/bin/env python

import argparse
import json
import matplotlib.pyplot as plt

from labelme import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    args = parser.parse_args()

    json_file = args.json_file

    data = json.load(open(json_file))  # 加载json文件

    img = utils.img_b64_to_array(data['imageData'])  # 解析原图片数据
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data[
        'shapes'])  # 解析'shapes'中的字段信息，解析出每个对象的mask与对应的label   lbl存储 mask，lbl_names 存储对应的label
    # lal 像素取值 0、1、2 其中0对应背景，1对应第一个对象，2对应第二个对象
    # 使用该方法取出每个对象的mask mask=[] mask.append((lbl==1).astype(np.uint8)) # 解析出像素值为1的对象，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
    # lbl_names  ['background','cat_1','cat_2']

    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl_viz)
    plt.show()


if __name__ == '__main__':
    main()

''' 其他
data['imageData'] # 原图数据 str
data['shapes'] # 每个对像mask及label  list
len(data['shapes']) # 返回对象个数 int
data['shapes'][0]['label'] # 返回第一个对象的标签 str
data['shapes'][0]['points'] # 返回第一个对象的边界点 list
data['shapes'][0]['points'][0] # 返回第一个对象的边界点第一个点 list

data['imagePath'] # 原图路径 str
data['fillColor'] # 填充颜色（边界内部） list
data['lineColor'] # 边界线颜色  list
'''