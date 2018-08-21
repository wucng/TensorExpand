# -*- coding:utf-8 -*-
"""参考：https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
各种可视化程序
"""
import cv2
import numpy as np
import os
import PIL.Image
import PIL.ImageDraw
import pycocotools.mask as mask_util

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)

bbox= [516.0, 236.0, 43.0, 125.0]
seg=[[522, 263, 538, 251, 543, 236, 553, 241, 555, 252, 555, 269, 559, 298, 554, 324, 550, 331, 556, 344, 555, 361, 544, 348, 542, 328, 543, 316, 540, 315, 531, 331, 529, 332, 529, 339, 532, 346, 525, 348, 516, 340, 523, 324, 528, 315, 528, 306, 523, 293, 523, 281, 522, 263]]

filename='pudong_test_000000005.jpg'

def change_format(contour):
    contour2=[]
    length=len(contour)
    for i in range(0,length,2):
        contour2.append([contour[i],contour[i+1]])
    return np.asarray(contour2,np.int32)

def vis_class(img, pos, class_str, font_scale=0.35):
    """Visualizes the class."""
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    txt = class_str
    font = cv2.FONT_HERSHEY_SIMPLEX
    ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
    # Place text background.
    back_tl = x0, y0 - int(1.3 * txt_h)
    back_br = x0 + txt_w, y0
    cv2.rectangle(img, back_tl, back_br, _GREEN, -1)
    # Show text.
    txt_tl = x0, y0 - int(0.3 * txt_h)
    cv2.putText(img, txt, txt_tl, font, font_scale, _GRAY, lineType=cv2.LINE_AA)
    return img
def vis_bbox(img, bbox, thick=2):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img
def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

def polygons_to_mask(img_shape, polygons):
    '''边界点生成mask'''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=np.uint8)*255
    return mask

def vis_mask2(img,seg,border_thick=1):
    img_shape=img.shape[:2]
    mask=polygons_to_mask(img_shape,change_format(seg[0]))
    _, contours, _ = cv2.findContours(
        mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)
    return img

img=cv2.imread(os.path.join('./images',filename))

img=vis_class(img,bbox,'person')
img=vis_bbox(img,bbox)

img=vis_mask2(img,seg)
# or
# cv2.imshow('img2',img)
# img_shape=img.shape
# mask=polygons_to_mask(img_shape,change_format(seg[0]))
# img=cv2.addWeighted(img,1.,mask,1,1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()