import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cv2,os
import PIL.Image
import PIL.ImageDraw

def create_pascal_label_colormap():
    """
    PASCAL VOC 分割数据集的类别标签颜色映射label colormap

    返回:
        可视化分割结果的颜色映射Colormap
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """
    添加颜色到图片，根据数据集标签的颜色映射 label colormap

    参数:
        label: 整数类型的 2D 数组array, 保存了分割的类别标签 label

    返回:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    """
    输入图片和分割 mask 的可视化.
    """
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

def vis_segmentation2(image, seg_map):
    """
    输入图片和分割 mask 的统一可视化.
    """
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.figure()
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.show()

def change_format(contour):
    contour2=[]
    length=len(contour)
    for i in range(0,length,2):
        contour2.append([contour[i],contour[i+1]])
    return np.asarray(contour2,np.int32)

def polygons_to_mask(img_shape, polygons):
    '''边界点生成mask'''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=np.uint8)
    return mask

LABEL_NAMES = np.asarray(['background', 'person']) # 假设只有两类
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

filename='pudong_test_000000005.jpg'
img=cv2.imread(os.path.join('./images',filename))

img = img[:,:,::-1]

seg=[[522, 263, 538, 251, 543, 236, 553, 241, 555, 252, 555, 269, 559, 298, 554, 324, 550, 331, 556, 344,
      555, 361, 544, 348, 542, 328, 543, 316, 540, 315, 531, 331, 529, 332, 529, 339, 532, 346, 525, 348,
      516, 340, 523, 324, 528, 315, 528, 306, 523, 293, 523, 281, 522, 263]]

img_shape=img.shape[:2]
mask=polygons_to_mask(img_shape,change_format(seg[0]))

vis_segmentation(img, mask)

vis_segmentation2(img,mask)
print('Done.')