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

- 试着在训练集中删掉一些勺子的图像，这样每种类别的图象数量就差不多（先做个备份）。你还需要删除所有文件名带有*.pkl的缓存文件，然后重新运行Notebook。这样会提高分类准确率吗？比较改变前后的混淆矩阵。
- 用convert.py 脚本建立你自己的数据集。比如，录下汽车和摩托车的视频，然后创建一个分类系统。
- 需要从你创建的训练集中删除一些不明确的图像吗？如何你删掉这些图像之后，分类准确率有什么变化？
- 改变Notebook，这样你可以输入单张图像而不是整个数据集。你不用从Inception模型中保存transfer-values。






