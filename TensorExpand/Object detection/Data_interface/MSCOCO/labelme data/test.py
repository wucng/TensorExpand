import json
import numpy as np
import skimage.io
import cv2
import matplotlib.pyplot as plt
import PIL.Image
import PIL.ImageDraw


data=json.load(open('1.json'))

img_path=data['imagePath'].split('/')[-1]
img=skimage.io.imread(img_path)

def polygons_to_mask(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return: 
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def polygons_to_mask2(img_shape, polygons):
    '''
    边界点生成mask
    :param img_shape: [h,w]
    :param polygons: labelme JSON中的边界点格式 [[x1,y1],[x2,y2],[x3,y3],...[xn,yn]]
    :return: 
    '''
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray([polygons], np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
    # cv2.fillPoly(mask, polygons, 1) # 非int32 会报错
    cv2.fillConvexPoly(mask, polygons, 1)  # 非int32 会报错
    return mask


points=[]
labels=[]
for shapes in data['shapes']:
    points.append(shapes['points'])
    labels.append(shapes['label'])

mask0=polygons_to_mask(img.shape[:2],points[0])
mask1=polygons_to_mask(img.shape[:2],points[1])

plt.subplot(161)
plt.imshow(img)
plt.axis('off')
plt.title('original')

plt.subplot(162)
plt.imshow(mask0.astype(np.uint8),'gray')
plt.axis('off')
plt.title('SegmentationObject:\n'+labels[0])

plt.subplot(163)
plt.imshow(mask1.astype(np.uint8),'gray')
plt.axis('off')
plt.title('SegmentationObject:\n'+labels[1])

plt.subplot(164)
plt.imshow(mask0*1+mask1*2,'gray')
plt.axis('off')
plt.title('SegmentationClass')

img_mask=np.zeros_like(img,np.uint8)
for i in range(3):
    img_mask[:,:,i]=mask1*img[:,:,i]


plt.subplot(165)
plt.imshow(img_mask)
plt.axis('off')


plt.show()


