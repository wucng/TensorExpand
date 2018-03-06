# -*- coding: utf8 -*-

'''参考：utils.py 的download_trained_weights'''

import urllib.request
import shutil

COCO_MODEL_URL='https://image.baidu.com/search/down?tn=download&word=download&ie=' \
               'utf8&fr=detail&url=https%3A%2F%2Ftimgsa.baidu.com%2Ftimg%3Fimage%26quality%3D80%26size%3Db9999_' \
               '10000%26sec%3D1520315813170%26di%3Df0cf0f7c8e041be005d78971c5c239d5%26imgtype%3D0%26src%3Dhttp%253A%25' \
               '2F%252Fwww.bapimi.com%252Fuploads%252F0_2501250005x2627376053_23.jpg&thumburl=https%3A%2F%2Fss2.bdstatic.' \
               'com%2F70cFvnSh_Q1YnxGkpoWK1HF6hhy%2Fit%2Fu%3D3436138697%2C1578547663%26fm%3D27%26gp%3D0.jpg'
coco_model_path='./5555.jpg'
def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.
    从Releases下载COCO训练的权重。
    coco_model_path: local path of COCO trained weights  存放COCO训练权重的本地路径 如：'./mask_rcnn_coco.h5'
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")

download_trained_weights(coco_model_path)
