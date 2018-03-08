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
            img_mask=cv2.rectangle(image_mask, (x - s, y - s), (x + s, y + s), 1, -1) # 0、1组成的图片
            # images.append(img)
            mask.append(img_mask)
            class_id.append(1) # 对应class ID 1

        elif shape == "circle":
            cv2.circle(image, (x, y), s, color, -1)
            img_mask = cv2.circle(image_mask, (x, y), s, 1, -1)
            # images.append(img)
            mask.append(img_mask)
            class_id.append(2)  # 对应class ID 2

        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            cv2.fillPoly(image, points, color)
            img_mask=cv2.fillPoly(image_mask, points, 1)
            # images.append(img)
            mask.append(img_mask)
            class_id.append(3)  # 对应class ID 3

    # images=np.asarray(images,np.float32) # [h,w,c]
    mask=np.asarray(mask,np.uint8).transpose([1, 2, 0]) # [h,w,instance count]
    class_id=np.asarray(class_id,np.uint8) # [instance count,]
    return image,mask,class_id

image,mask,class_id=random_image(128,128)

plt.subplot(1,len(class_id)+1,1)
plt.imshow(image)
plt.axis('off')
for i in range(len(class_id)):
    plt.subplot(1, len(class_id) + 1, i+2)
    plt.imshow(mask[:,:,i])
    plt.axis('off')
    plt.title(class_id[i])

plt.show()
