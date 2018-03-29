参考：https://pjreddie.com/darknet/darkgo-go-in-darknet/


----------
[AlphaGo](https://deepmind.com/research/alphago/)让我对游戏中的神经网络感兴趣。

我还没有真正阅读过他们的论文，但是我已经实现了我想象中的与他们的政策网络类似的东西。 这是一个预测Go游戏中最可能的下一步移动的神经网络。 您可以与专业游戏一起玩，看看接下来会发生什么样的动作，让它发挥自身的作用，或者尝试与之对抗！

![这里写图片描述](https://pjreddie.com/media/image/Screen_Shot_2017-03-26_at_10.55.10_PM.png)


目前DarkGo播放约1丹。 这对于没有前瞻性的单个网络来说相当不错，它只评估当前的电路板状态。

来玩我的Online-Go服务器！https://online-go.com/user/view/434218

# Playing With A Trained Model
首先安装Darknet，这可以通过以下方式完成：

```
git clone https://github.com/pjreddie/darknet
cd darknet
make
```
同时下载权重文件：

```
wget pjreddie.com/media/files/go.weights
```
然后以测试模式运行Go引擎：

```
./darknet go test cfg/go.test.cfg go.weights
```
这将带来一个互动的Go板。 您可以：

- 按`enter `即可播放计算机上的第一个建议移动
- 输入一个数字3来播放该号码建议
- 输入一个位置，如A 15来播放该移动
- 输入c A 15以清除A 15上的任何碎片
- 输入b A 15，在A 15放置黑色块
- 输入w A 15，在A 15处放置一个白色块
- 输入p以通过回合

玩的开心！

如果你想让你的网络更加强大，把flag `-multi`添加到测试命令中。 这个评估板在多次旋转和翻转以获得更好的概率估计。 CPU速度可能会很慢，但如果您拥有CUDA，速度会非常快。

# Data
我使用Hugh Perkins' [Github](https://github.com/hughperkins/kgsgo-dataset-preprocessor)的Go数据集。 我送入Darknet的数据是一个1通道图像，用于编码当前的游戏状态。 1代表你的棋子，-1代表你的对手棋子，0代表空闲空间。 该网络预测当前玩家下一步可能玩的位置。

我在这里可以找到后处理后使用的完整数据集（3.0 GB），仅用于训练）：

- [go.train](https://pjreddie.com/media/files/go.train)