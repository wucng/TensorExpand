# -*- coding: UTF-8 -*-
"""
最后对得到的掩膜图像，用于去噪
"""

import cv2
import numpy as np


def main(img, kernel=np.ones(5,5), np.uint8), filter=MORPH_CLOSE):
    filter_fn = img[0:-4] + str(filter)
    
    img = cv2.imread(img, 0)
    kernel = kernel
    
    if filter == "erode":
        erosion = cv2.erode(img, kernel, 1)
    else if filter == "dilate":
        erosion = cv2.dilate(img,kernel,1)
    else if filter == "MORPH_CLOSE":
        erosion = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    else if filter == "MORPH_OPEN":
        erosion = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    # else if filter == "filter2D":
        # dst = cv2.filter2D(img1,-1,kernel)
        #cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
        #输出图像与输入图像大小相同
        #中间的数为-1，输出数值格式的相同plt.figure()
        # erosion = dst
   cv2.imwrite(filter_fn, erosion)

    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("""
        filter type:
        erode:          腐蚀
        dilate:         膨胀
        MORPH_OPEN:     开运算(先腐蚀后膨胀)
        MORPH_CLOSE:    闭运算(先膨胀后腐蚀)
        MORPH_GRADIENT：形态学梯度(提取轮廓)
        MORPH_TOPHAT:   礼帽(原始图像减去开运算后的图像)
        MORPH_BLACKHAT: 黑帽(原始图像减去闭运算后的图像)
        
        """)
    main(sys.argv)
