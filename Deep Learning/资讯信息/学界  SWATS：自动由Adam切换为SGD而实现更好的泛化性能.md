[网站](https://mbd.baidu.com/newspage/data/landingsuper?context=%7B"nid"%3A"news_10049770018685133327"%7D&n_type=0&p_from=1)

选自arXiv
作者：Nitish Shirish Keskar、Richard Socher
机器之心编译
参与：蒋思源、李泽南
在 ICLR 2018 的高分论文中，有研究者表明因为历史梯度平方的滑动平均值，Adam 等算法并不能收敛到最优解，因此它在泛化误差上可能要比 SGD 等方法差。最近 Salesforce 的研究者提出了一种转换机制，他们试图让算法在训练过程中自动由 Adam 无缝转换到 SGD 而保留两种优化算法的优良属性。
随机梯度下降（SGD）已经成为了深度神经网络最常用的训练算法之一。尽管它非常简单，但在各种应用程序中都表现良好，且也有很强的理论基础。这些理论基础体现在避免陷入鞍点问题（Lee et al., 2016）、提高泛化性能（Hardt et al., 2015; Wilson et al., 2017）和解释为贝叶斯推断（Mandt et al., 2017）等方面。
训练神经网络等价于解决以下非凸优化问题：
其中 f 为损失函数。SGD 的迭代更新公式可以表示为：

其中 w_k 表示第 k 次迭代，α_k 为控制下降步幅大小的参数序列，它同样也可以称为学习率。 f(w_k) hat 表示损失函数对 w_k 所求的随机梯度。SGD 的变体 SGDM 使用迭代的惯性加速训练过程，该方法已在实践中表现出十分优秀的性能（Sutskever et al., 2013）。SGDM 的迭代更新表达式可以表示为：

其中 β ∈ [0, 1) 为动量参数，v_0 初始化为 0。
SGD 的缺点是它在所有方向上一致地缩放梯度而确定下降步长，这对病态问题可能特别有害。因此 SGD 需要依据实际情况频繁地修正学习率 α。
为了纠正这些缺点，一些适应性方法通过估计函数的曲率而提出了解决方案，这些方法包括 Adam（Kingma &amp; Ba, 2015）、Adagrad（Duchi et al., 2011）和 RMSprop（Tieleman &amp; Hinton, 2012）。这些方法可以解释为使用学习率向量的 SGD，即它们会根据训练算法的过程而自适应地修正学习率。此外，对于 SGD 与 SGDM 等方法来说，它们的学习率是一个标量。
然而有趣的是，Adam 虽然在初始部分的训练和泛化度量都优于 SGD，但在收敛部分的性能却停滞不前。这令很多研究者开始寻找结合 Adam 和 SGD 的新方法，他们希望新算法不仅能利用 Adam 的快速初始化过程，同时还利用 SGD 的泛化属性。
此外，Wilson 等人今年发表研究表明适应性方法因为非均匀的梯度缩放而导致泛化性能的损失，因此我们比较自然的策略是利用 Adam 算法初始化训练，然后在适当的时候转换为 SGD 方法。
为了更进一步研究该问题，近日 Nitish Shirish Keskar 和 Richard Socher 提出了 SWATS 算法，该算法使用简单的策略在训练中从 Adam 转换为 SGD 而结合两种算法的优点。SWATS 算法的转换过程是自动完成的，因此它并不会引入更多的超参数。
在 Nitish 等人的策略中，转换点和 SGD 学习率都是作为参数而在训练过程学习的。他们在梯度子空间中监控 Adam 迭代步的投影，并使用它的指数平均作为转换后 SGD 学习率的估计。
论文：Improving Generalization Performance by Switching from Adam to SGD

论文链接：https://arxiv.org/abs/1712.07628
摘要：尽管训练结果十分优秀，Adam、Adagrad 或 RMSprop 等适应性优化方法相比于随机梯度下降（SGD）还是会有较差的泛化性能。这些方法在训练的初始阶段通常表现良好，但在训练的后期的性能会被 SGD 所超越。我们研究了一种混合优化策略，其中初始化部分仍然使用适应性方法，然后在适当的时间点转换为 SGD。具体来说，我们提出了 SWATS 算法，一种在触发条件满足时由 Adam 转化为 SGD 的简单策略。我们提出的转换条件涉及到梯度子空间中的 Adam 迭代步投影。通过设计，该转换条件的监控过程只会添加非常少的计算开销，且并不会增加优化器中超参数的数量。我们试验了多个标准的基准测试，例如在 CIFAR-10 和 CIFAR-100 数据集上的 ResNet、SENet、DenseNet 和 PyramidNet，在 tiny-ImageNet 数据集上的 ResNet，或者是在 PTB、WT2 数据集上使用循环神经网络的语言模型。试验结果表明，我们的策略能令 SGD 与 Adam 算法之间的泛化差距在大多数任务中都能得到缩小。
如图 1 所示，SGD 的期望测试误差能收敛到约为 5% 左右，而 Adam 的泛化误差在 7% 左右就开始震荡，因此精调的学习率策略并没有取得更好的收敛性能。
![image](https://ss2.baidu.com/6ONYsjip0QIZ8tyhnq/it/u=2252143682,1082031211&fm=173&s=25BCEC3249C6CECE56C188D0000050B3&w=640&h=558&img.JPEG)
图 1：在 CIFAR-10 数据集上使用四种优化器 SGD、Adam、Adam-Clip（1,∞）和 Adam-Clip（0,1）训练 DenseNet 架构。SGD 在训练中实现了最佳测试准确率，且与 Adam 的泛化差距大概为 2%。为 Adam 的每个参数设置最小的学习速率可以减小泛化差距。
正如图 2 所示，在 10 个 epoch 之后切换会导致学习曲线非常类似于 SGD，而在 80 个 epoch 之后切换会导致精度下降约 6.5%。
![image](https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=2048163045,1575938519&fm=173&s=68C7E81251C7FEEF62F1DDDD030090A1&w=640&h=615&img.JPEG)
图 2：使用 CIFAR-10 数据集上训练 DenseNet 架构，使用 Adam，在（10、40、80）epoch 后调整 SGD 学习速率至 0.1，动量为 0.9；切换点在图中使用 Sw@ 表示。更早切换可以让模型达到与 SGD 相比的准确率，而如果在训练过程中切换过晚会导致与 Adam 相同的泛化差距。
![image](https://ss2.baidu.com/6ONYsjip0QIZ8tyhnq/it/u=1084124060,2470841666&fm=173&s=E900F41A09AFD0CA40501CDA000080B2&w=591&h=905&img.JPEG)

![image](https://ss0.baidu.com/6ONWsjip0QIZ8tyhnq/it/u=1554739986,3318555428&fm=173&s=011AE832D9CE5CCA5AE571DB0000E0B1&w=640&h=403&img.JPEG)

图 4：在 CIFAR-10 和 CIFAR-100 数据集上训练 ResNet-32、DenseNet、PyramidNet 和 SENet，并比较 SGD（M）、Adam 和 SWATS 的学习速率。
![image](https://ss1.baidu.com/6ONXsjip0QIZ8tyhnq/it/u=1525068649,3589909919&fm=173&s=05B2EC32498FECCA487C01DB0000C0B2&w=640&h=619&img.JPEG)
图 5：在 Tiny-ImageNet 数据集上训练 ResNet-18，并比较 SGD（M）、Adam 和 SWATS 的学习速率。