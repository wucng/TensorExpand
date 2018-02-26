# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import math

# Image manipulation.
import PIL.Image
from scipy.ndimage.filters import gaussian_filter

# Inception 模型
'''
前面的一些教程都使用了Inception v3模型。本文将会使用Inception模型的另一个变体。
由于Google开发者并没有很好的为其撰写文档（跟通常一样），不太清楚模型是哪个版本。
我们在这里用“Inception 5h”来指代它，因为zip包的文件名就是这样，
尽管看起来这是Inception模型的一个早期的、更简单的版本。
'''

'''
这里使用Inception 5h模型是因为它更容易使用：它接受任何尺寸的输入图像，
然后创建比Inception v3模型（见教程 #13）更漂亮的图像。
'''

import inception5h
# inception.data_dir = 'inception/5h/'
inception5h.maybe_download()

model = inception5h.Inception5h()

# Inception 5h模型有许多层可用来做DeepDreaming。我们列出了12个最常用的层，以供参考。
len(model.layer_tensors)

# 12

# 这个函数载入一张图像，并返回一个浮点型numpy数组。
def load_image(filename):
    image = PIL.Image.open(filename)

    return np.float32(image)

# 将图像保存成jpeg文件。图像是保存着0-255像素的numpy数组。
def save_image(image, filename):
    # Ensure the pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Convert to bytes.
    image = image.astype(np.uint8)

    # Write the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')

# 这是绘制图像的函数。使用matplotlib将得到低分辨率的图像。使用PIL效果比较好

def plot_image(image):
    # Assume the pixel-values are scaled between 0 and 255.

    if False:
        # Convert the pixel-values to the range between 0.0 and 1.0
        image = np.clip(image/255.0, 0.0, 1.0)

        # Plot using matplotlib.
        plt.imshow(image, interpolation='lanczos')
        plt.show()
    else:
        # Ensure the pixel-values are between 0 and 255.
        image = np.clip(image, 0.0, 255.0)

        # Convert pixels to bytes.
        image = image.astype(np.uint8)

        # Convert to a PIL-image and display it.
        display(PIL.Image.fromarray(image))

# 归一化图像，则像素值在0.0到1.0之间。这个在绘制梯度时很有用。
def normalize_image(x):
    # Get the min and max values for all pixels in the input.
    x_min = x.min()
    x_max = x.max()

    # Normalize so all values are between 0.0 and 1.0
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm
# 对梯度做归一化之后，用这个函数绘制。
def plot_gradient(gradient):
    # Normalize the gradient so it is between 0.0 and 1.0
    gradient_normalized = normalize_image(gradient)

    # Plot the normalized gradient.
    plt.imshow(gradient_normalized, interpolation='bilinear')
    plt.show()

# 这个函数调整图像的大小。函数的参数是你指定的具体的图像分辨率，
# 比如(100，200)，它也可以接受一个缩放因子，比如，参数是0.5时,图像每个维度缩小一半。
'''
这个函数用PIL来实现，代码有点长，因为我们用numpy数组来处理图像，其中像素值是浮点值。
PIL不支持这个，因此需要将图像转换成8位字节，来确保像素值在合适的范围内。
然后，图像被调整大小并转换回浮点值。
'''
def resize_image(image, size=None, factor=None):
    # If a rescaling-factor is provided then use it.
    if factor is not None:
        # Scale the numpy array's shape for height and width.
        size = np.array(image.shape[0:2]) * factor

        # The size is floating-point because it was scaled.
        # PIL requires the size to be integers.
        size = size.astype(int)
    else:
        # Ensure the size has length 2.
        size = size[0:2]

    # The height and width is reversed in numpy vs. PIL.
    size = tuple(reversed(size))

    # Ensure the pixel-values are between 0 and 255.
    img = np.clip(image, 0.0, 255.0)

    # Convert the pixels to 8-bit bytes.
    img = img.astype(np.uint8)

    # Create PIL-object from numpy array.
    img = PIL.Image.fromarray(img)

    # Resize the image.
    img_resized = img.resize(size, PIL.Image.LANCZOS)

    # Convert 8-bit pixel values back to floating-point.
    img_resized = np.float32(img_resized)

    return img_resized

# DeepDream 算法
'''
下面的帮助函数计算了在DeepDream中使用的输入图像的梯度。Inception 5h模型可以接受任意尺寸的图像，
但太大的图像可能会占用千兆字节的内存。为了使内存占用最低，我们将输入图像分割成小的图块，然后计算每小块的梯度。

然而，这可能会在DeepDream算法最终生成的图像中产生肉眼可见的线条。因此我们随机地挑选小块，
这样它们的位置就是不同的。这使得在最终的DeepDream图像里，小块之间的缝隙不可见。

这个帮助函数用来确定合适的图块尺寸。比如，期望的图块尺寸为400x400像素，但实际大小取决于图像尺寸。
'''
def get_tile_size(num_pixels, tile_size=400):
    """
    num_pixels is the number of pixels in a dimension of the image.
    tile_size is the desired tile-size.
    """

    # How many times can we repeat a tile of the desired size.
    num_tiles = int(round(num_pixels / tile_size))

    # Ensure that there is at least 1 tile.
    num_tiles = max(1, num_tiles)

    # The actual tile-size.
    actual_tile_size = math.ceil(num_pixels / num_tiles)

    return actual_tile_size


'''
这个帮助函数计算了输入图像的梯度。图像被分割成小块，然后分别计算各个图块的梯度。
图块是随机选择的，避免在最终的DeepDream图像内产生可见的缝隙。
'''
def tiled_gradient(gradient, image, tile_size=400):
    # Allocate an array for the gradient of the entire image.
    grad = np.zeros_like(image)

    # Number of pixels for the x- and y-axes.
    x_max, y_max, _ = image.shape

    # Tile-size for the x-axis.
    x_tile_size = get_tile_size(num_pixels=x_max, tile_size=tile_size)
    # 1/4 of the tile-size.
    x_tile_size4 = x_tile_size // 4

    # Tile-size for the y-axis.
    y_tile_size = get_tile_size(num_pixels=y_max, tile_size=tile_size)
    # 1/4 of the tile-size
    y_tile_size4 = y_tile_size // 4

    # Random start-position for the tiles on the x-axis.
    # The random value is between -3/4 and -1/4 of the tile-size.
    # This is so the border-tiles are at least 1/4 of the tile-size,
    # otherwise the tiles may be too small which creates noisy gradients.
    x_start = random.randint(-3*x_tile_size4, -x_tile_size4)

    while x_start < x_max:
        # End-position for the current tile.
        x_end = x_start + x_tile_size

        # Ensure the tile's start- and end-positions are valid.
        x_start_lim = max(x_start, 0)
        x_end_lim = min(x_end, x_max)

        # Random start-position for the tiles on the y-axis.
        # The random value is between -3/4 and -1/4 of the tile-size.
        y_start = random.randint(-3*y_tile_size4, -y_tile_size4)

        while y_start < y_max:
            # End-position for the current tile.
            y_end = y_start + y_tile_size

            # Ensure the tile's start- and end-positions are valid.
            y_start_lim = max(y_start, 0)
            y_end_lim = min(y_end, y_max)

            # Get the image-tile.
            img_tile = image[x_start_lim:x_end_lim,
                             y_start_lim:y_end_lim, :]

            # Create a feed-dict with the image-tile.
            feed_dict = model.create_feed_dict(image=img_tile)

            # Use TensorFlow to calculate the gradient-value.
            g = session.run(gradient, feed_dict=feed_dict)

            # Normalize the gradient for the tile. This is
            # necessary because the tiles may have very different
            # values. Normalizing gives a more coherent gradient.
            g /= (np.std(g) + 1e-8)

            # Store the tile's gradient at the appropriate location.
            grad[x_start_lim:x_end_lim,
                 y_start_lim:y_end_lim, :] = g

            # Advance the start-position for the y-axis.
            y_start = y_end

        # Advance the start-position for the x-axis.
        x_start = x_end

    return grad

# 优化图像
'''
这个函数是DeepDream算法的主要优化循环。它根据输入图像计算Inception模型中给定层的梯度。
然后将梯度添加到输入图像，从而增加层张量(layer-tensor)的平均值。多次重复这个过程，
并放大Inception模型在输入图像中看到的任何图案。
'''
def optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=3.0, tile_size=400,
                   show_gradient=False):
    """
    Use gradient ascent to optimize an image so it maximizes the
    mean value of the given layer_tensor.

    Parameters:
    layer_tensor: Reference to a tensor that will be maximized.
    image: Input image used as the starting point.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    show_gradient: Plot the gradient in each iteration.
    """

    # Copy the image so we don't overwrite the original image.
    img = image.copy()

    print("Image before:")
    plot_image(img)

    print("Processing image: ", end="")

    # Use TensorFlow to get the mathematical function for the
    # gradient of the given layer-tensor with regard to the
    # input image. This may cause TensorFlow to add the same
    # math-expressions to the graph each time this function is called.
    # It may use a lot of RAM and could be moved outside the function.
    gradient = model.get_gradient(layer_tensor)

    for i in range(num_iterations):
        # Calculate the value of the gradient.
        # This tells us how to change the image so as to
        # maximize the mean of the given layer-tensor.
        grad = tiled_gradient(gradient=gradient, image=img)

        # Blur the gradient with different amounts and add
        # them together. The blur amount is also increased
        # during the optimization. This was found to give
        # nice, smooth images. You can try and change the formulas.
        # The blur-amount is called sigma (0=no blur, 1=low blur, etc.)
        # We could call gaussian_filter(grad, sigma=(sigma, sigma, 0.0))
        # which would not blur the colour-channel. This tends to
        # give psychadelic / pastel colours in the resulting images.
        # When the colour-channel is also blurred the colours of the
        # input image are mostly retained in the output image.
        sigma = (i * 4.0) / num_iterations + 0.5
        grad_smooth1 = gaussian_filter(grad, sigma=sigma)
        grad_smooth2 = gaussian_filter(grad, sigma=sigma*2)
        grad_smooth3 = gaussian_filter(grad, sigma=sigma*0.5)
        grad = (grad_smooth1 + grad_smooth2 + grad_smooth3)

        # Scale the step-size according to the gradient-values.
        # This may not be necessary because the tiled-gradient
        # is already normalized.
        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Update the image by following the gradient.
        img += grad * step_size_scaled

        if show_gradient:
            # Print statistics for the gradient.
            msg = "Gradient min: {0:>9.6f}, max: {1:>9.6f}, stepsize: {2:>9.2f}"
            print(msg.format(grad.min(), grad.max(), step_size_scaled))

            # Plot the gradient.
            plot_gradient(grad)
        else:
            # Otherwise show a little progress-indicator.
            print(". ", end="")

    print()
    print("Image after:")
    plot_image(img)

    return img

# 图像递归优化
'''
Inception模型在相当小的图像上进行训练。不清楚图像的确切大小，但可能每个维度200-300像素。
如果我们使用较大的图像，比如1920x1080像素，那么上面的optimize_image()函数会在图像上添加很多小的图案。

这个帮助函数将输入图像多次缩放，然后用每个缩放图像来执行上面的optimize_image()函数。这在最终的图像中生成较大的图案。它也能加快计算速度。
'''
def recursive_optimize(layer_tensor, image,
                       num_repeats=4, rescale_factor=0.7, blend=0.2,
                       num_iterations=10, step_size=3.0,
                       tile_size=400):
    """
    Recursively blur and downscale the input image.
    Each downscaled image is run through the optimize_image()
    function to amplify the patterns that the Inception model sees.

    Parameters:
    image: Input image used as the starting point.
    rescale_factor: Downscaling factor for the image.
    num_repeats: Number of times to downscale the image.
    blend: Factor for blending the original and processed images.

    Parameters passed to optimize_image():
    layer_tensor: Reference to a tensor that will be maximized.
    num_iterations: Number of optimization iterations to perform.
    step_size: Scale for each step of the gradient ascent.
    tile_size: Size of the tiles when calculating the gradient.
    """

    # Do a recursive step?
    if num_repeats>0:
        # Blur the input image to prevent artifacts when downscaling.
        # The blur amount is controlled by sigma. Note that the
        # colour-channel is not blurred as it would make the image gray.
        sigma = 0.5
        img_blur = gaussian_filter(image, sigma=(sigma, sigma, 0.0))

        # Downscale the image.
        img_downscaled = resize_image(image=img_blur,
                                      factor=rescale_factor)

        # Recursive call to this function.
        # Subtract one from num_repeats and use the downscaled image.
        img_result = recursive_optimize(layer_tensor=layer_tensor,
                                        image=img_downscaled,
                                        num_repeats=num_repeats-1,
                                        rescale_factor=rescale_factor,
                                        blend=blend,
                                        num_iterations=num_iterations,
                                        step_size=step_size,
                                        tile_size=tile_size)

        # Upscale the resulting image back to its original size.
        img_upscaled = resize_image(image=img_result, size=image.shape)

        # Blend the original and processed images.
        image = blend * image + (1.0 - blend) * img_upscaled

    print("Recursive level:", num_repeats)

    # Process the image using the DeepDream algorithm.
    img_result = optimize_image(layer_tensor=layer_tensor,
                                image=image,
                                num_iterations=num_iterations,
                                step_size=step_size,
                                tile_size=tile_size)

    return img_result

session = tf.InteractiveSession(graph=model.graph)

image = load_image(filename='images/hulk.jpg')
plot_image(image)
'''
首先，我们需要Inception模型中的张量的引用，它将在DeepDream优化算法中被最大化。
在这个例子中，我们选择Inception模型的第3层（层索引2）。它有192个通道，我们将尝试最大化这些通道的平均值。
'''
layer_tensor = model.layer_tensors[2]
# layer_tensor
# <tf.Tensor 'conv2d2:0' shape=(?, ?, ?, 192) dtype=float32>

'''
现在运行DeepDream优化算法，总共10次迭代，步长为6.0，这是下面递归优化的两倍。
每次迭代我们都展示它的梯度，你可以看到图像方块之间的痕迹。
'''
img_result = optimize_image(layer_tensor, image,
                   num_iterations=10, step_size=6.0, tile_size=400,
                   show_gradient=True)

# save_image(img_result, filename='deepdream_hulk.jpg')

'''
现在，递归调用DeepDream算法。我们执行5个递归（num_repeats + 1），每个步骤中图像都被模糊并缩小，
然后在缩小图像上运行DeepDream算法。接着，在每个步骤中，将产生的DeepDream图像与原始图像混合，
从原始图像获取一点细节。这个过程重复了多次。

注意，现在DeepDream的图案更大了。这是因为我们先在低分辨率图像上创建图案，然后在较高分辨率图像上进行细化。
'''
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)
'''
现在我们将最大化Inception模型中的较高层。使用7号层（索引6）为例。
该层识别输入图像中更复杂的形状，所以DeepDream算法也将产生更复杂的图像。
这一层似乎识别了狗的脸和毛发，因此DeepDream算法往图像中添加了这些东西。

再次注意，与DeepDream算法其他变体不同的是，这里输入图像的大部分颜色被保留了下来，
创建了更多柔和的颜色。这是因为我们在颜色通道中平滑了梯度，使其变得有点像灰阶，因此不会太多地改变输入图像的颜色。
'''
layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)


layer_tensor = model.layer_tensors[7][:,:,:,0:3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=4, blend=0.2)

layer_tensor = model.layer_tensors[11][:,:,:,0]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=3, blend=0.2)

# Giger
image = load_image(filename='images/giger.jpg')
plot_image(image)

layer_tensor = model.layer_tensors[3]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=3, blend=0.2)

layer_tensor = model.layer_tensors[5]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=3, blend=0.2)

# Escher
image = load_image(filename='images/escher_planefilling2.jpg')
plot_image(image)

layer_tensor = model.layer_tensors[6]
img_result = recursive_optimize(layer_tensor=layer_tensor, image=image,
                 num_iterations=10, step_size=3.0, rescale_factor=0.7,
                 num_repeats=3, blend=0.2)

session.close()