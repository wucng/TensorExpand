参考：

- [TensorFlow 教程 #09 - 视频数据](https://zhuanlan.zhihu.com/p/27160921)
- [Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [thrillerist/TensorFlow-Tutorials](https://github.com/thrillerist/TensorFlow-Tutorials)


----------
我们使用新的数据集[Knifey-Spoony](http://link.zhihu.com/?target=https://github.com/Hvass-Labs/knifey-spoony)，它包含了上千张不同背景下的餐刀、勺子和叉子的图像。训练集有4170张图像，测试集有530张。类别为knifey、sppony和forky
# Images
- All images are 200 x 200 pixels with 3 colour channels.
- There is a total of 4700 jpg-images of which 530 are test-images with different backgrounds than the training-set.
- The knifey class has 1347 images total (137 images in the test-set).
- The spoony class has 2208 images total (242 images in the test-set).
- The forky class has 1145 images total (151 images in the test-set).


knifey-spoony数据集中的图像是用一个简单的[Python脚本](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/convert.py)从视频文件中获取的，脚本在Linux上运行（它需要avconv程序将视频转成图像）。这让你可以很快地从几分钟的录像视频中创建包含上千张图像的数据集。

使用[knifey.py](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/knifey.py)下载和提取数据

![这里写图片描述](https://pic4.zhimg.com/80/v2-b04fb26dcbcb7846aaea7b0fa2acf815_hd.jpg)







