# -*- coding:utf-8 -*-
"""
参考：https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2,os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)
def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines

def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }
    return keypoints, keypoint_flip_map

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints, _ = get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, dataset_keypoints.index('right_shoulder')] +
        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(
        kps[2, dataset_keypoints.index('right_shoulder')],
        kps[2, dataset_keypoints.index('left_shoulder')])
    mid_hip = (
        kps[:2, dataset_keypoints.index('right_hip')] +
        kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(
        kps[2, dataset_keypoints.index('right_hip')],
        kps[2, dataset_keypoints.index('left_hip')])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[2, nose_idx] > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]),
            color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(
            kp_mask, tuple(mid_shoulder), tuple(mid_hip),
            color=colors[len(kp_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = kps[0, i1], kps[1, i1]
        p2 = kps[0, i2], kps[1, i2]
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

keypoints=["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
                          "left_knee", "right_knee", "left_ankle", "right_ankle"]

filename='pudong_test_000000285.jpg'

img=cv2.imread(os.path.join('./images',filename))
kps=[519, 257, 2, 0, 0, 0, 515, 255, 2, 0, 0, 0, 508, 255, 2, 522, 264, 2,
     502, 265, 2, 528, 278, 2, 509, 289, 2, 532, 292, 2, 525, 296, 2, 522, 303, 2,
     507, 305, 2, 519, 326, 2, 509, 330, 2, 518, 347, 2, 512, 350, 2]

def kp_connections2(keypoints):
    kp_lines = [
        [keypoints.get('left_eye'), keypoints.get('right_eye')],
        [keypoints.get('left_eye'), keypoints.get('nose')],
        [keypoints.get('right_eye'), keypoints.get('nose')],
        [keypoints.get('right_eye'), keypoints.get('right_ear')],
        [keypoints.get('left_eye'), keypoints.get('left_ear')],
        [keypoints.get('right_shoulder'), keypoints.get('right_elbow')],
        [keypoints.get('right_elbow'), keypoints.get('right_wrist')],
        [keypoints.get('left_shoulder'), keypoints.get('left_elbow')],
        [keypoints.get('left_elbow'), keypoints.get('left_wrist')],
        [keypoints.get('right_hip'), keypoints.get('right_knee')],
        [keypoints.get('right_knee'), keypoints.get('right_ankle')],
        [keypoints.get('left_hip'), keypoints.get('left_knee')],
        [keypoints.get('left_knee'), keypoints.get('left_ankle')],
        [keypoints.get('right_shoulder'), keypoints.get('left_shoulder')],
        [keypoints.get('right_hip'), keypoints.get('left_hip')],
    ]
    return kp_lines

def vis_keypoints2(img,kps):
    data={}
    for i,key in enumerate(keypoints):
        data[key]=tuple(kps[i*3:(i+1)*3][:2])
        # 画点
        cv2.circle(img,data[key],2,(0,255,0),-1)
    # 画线
    for line in kp_connections2(data):
        if sum(line[0]) and sum(line[1]):
            cv2.line(img,line[0],line[1],(0,0,255),1,cv2.LINE_AA)
    return img

img=vis_keypoints2(img,kps)

cv2.imshow('keypoints',img)
cv2.waitKey(0)
cv2.destroyAllWindows()