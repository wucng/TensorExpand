参考：https://pjreddie.com/darknet/hardware-guide/


----------
卷积神经网络现在对计算机视觉来说都非常愤怒。 然而，由于它们相对较新，并且该领域的运行如此之快，许多人都对如何最好地训练它们感到困惑。 我有同学毕业生和行业研究人员问我应该为个人机器或服务器获得什么样的硬件。

NVIDIA希望您购买新的[DIGITS Devbox](https://developer.nvidia.com/devbox)，但价格为15,000美元，延迟时间为8-10周，我不确定为什么有人会需要它。 对于大约6,000美元的税后，您可以建立自己的4 GPU盒，从Newegg几天内发货。

# 完整版本
你可以在[这里](https://secure.newegg.com/Wishlist/PublicWishlists)找到我的完整版本。 这只是为了装箱，你还需要一些GPU。 我从亚马逊获得了4款EVGA Titan X's。

![这里写图片描述](https://pjreddie.com/media/image/build.png)

基本电脑是1400美元，再加上4k美元的GPU，你就可以开始了！ 远低于15,000美元。

# GPUs
很可能构建中最重要也是最昂贵的部分将是GPU，并且有充分的理由。 与CPU相比，GPU对神经网络的训练和测试速度要快100倍以上。 我们的大部分计算会将大矩阵放在一起，所以我们需要一张具有较高单精度性能的卡。

![这里写图片描述](https://pjreddie.com/media/image/evga_titan_x_1.png)

# Titan X

这可能是你想要的。 设计成NVIDIA的最高端游戏GPU，它们的处理能力仅为1,000美元，几乎可以装载7 TFLOPS，并且可以将其中的4个装入一台机器。 有了12 GB的VRAM，他们可以运行所有大型机型，并留出足够的空间。

网上的人似乎认为，EVGA和华硕的质量差不多，但EVGA有更好的客户服务，所以我会找到那个。 我在亚马逊买了他们，因为他们比其他选项便宜一点，他们很快就发货了。

# 比较：Tesla K40，K80，其他GeForce 900s
K40和K80更适合于双精度(double precision)双倍性能，我们并不在意。他们的单精度(single precision)表现与Titan X相当，但表现出显着的涨幅。

其他高端GTX显卡，如980和980 Ti，与Titan X相比，可以享受优惠的单精度性能。但是，如果您想要大量的处理能力，使用4块Titan X的机器制造一台机器要简单得多超过2个机器8块980s。这是比较：

- Titan X：1,000美元，6.9 TFLOPS，6.9 GFLOPS / $
- GTX 980 Ti：670美元，5.6 TFLOPS，8.4 GFLOPS /美元
- GTX 980：500美元，4.1 TFLOPS，8.2 GFLOPS /美元
- 特斯拉K40：3,000美元，4.3 TFLOPS，1.4 GFLOPS /美元
- 特斯拉K80：5,000美元，8.7 TFLOPS，1.7 GFLOPS /美元

从效率的角度来看，980 Ti出炉了。但是，您确实做出了一些牺牲：仅有6 GB的VRAM和较慢的整体性能。如果你现金不足，我会说980 Ti是一个不错的选择。主要的任务是不出于任何原因得到特斯拉。他们疯狂昂贵，因为你得到了什么。如果你希望在一个盒子里的整体处理能力比Titan X是最好的选择。

# 主板：GIGABYTE GA-X99-UD3P
![这里写图片描述](https://pjreddie.com/media/image/Gigabyte-GA-Z97X-Gaming-G1-WiFi-Black-Edition_1.png)

主板最重要的一点是它可以容纳你想要的所有卡。 无论你选择什么，确保它可以支持4张卡。 一般来看看高端游戏主板。 有足够的空间用于4张双倍卡。

# CPU: Intel Core i7-5820K
你不需要一个好的CPU，但你可能会得到一个！ 拥有12个有效内核，这是一个不错的选择。

# 电源：Rosewill Hercules
我们的每张卡都将耗用大约250瓦的电力，而我们的其他计算机可能也需要一些。 这意味着我们需要超过一千瓦的电力！ 为了安全起见，我会使用真正的怪物电源，比如[Rosewill Hercules 1600W](https://www.newegg.com/Product/Product.aspx?Item=N82E16817182251)。 它有一吨的力量; 唯一的问题是它很大！ 我不得不从我的箱子里拿出一个风扇固定器来安装它。

# 内存：G.SKILL Ripjaws 4系列32GB
很多便宜的内存。 32 GB应该足够了，即使这意味着我们的RAM比VRAM更少！

# SSD：Mushkin增强型反应堆2.5“1TB
你可以花费大量的资金获得像NVIDIA的devbox这样的9TB RAID设置，或者你可以购买一个SSD。 无论是在同一台机器上，还是通过远程同步训练好的模型，备份硬盘驱动器仍然是个不错的主意。 但是这款SSD将足以存储和提供数据，甚至可以同时支持4个GPU。 我可以在没有任何延迟培训的情况下同时运行4个我[ fastest Imagenet models](https://pjreddie.com/darknet/imagenet/#reference)。 这意味着它每秒钟载入大约1,000张图像（大约16 MB）！ 由于现在大部分型号都比较大，所以从磁盘加载不会成为你的瓶颈。

唯一的缺点是尺寸。 如果您正在处理非常大的数据集，则可能需要更大的硬盘来存储数据，然后再对其进行预处理。

# Case: Rosewill Thor V2

![这里写图片描述](https://pjreddie.com/media/image/case.png)

这可能至少是重要的，但它确实看起来很酷！ 你真的只需要一个足够大的箱子来舒适地装配你的所有组件。 4个GPU占用大量空间！ 你也想要一些很酷的东西。 Rosewill Thor包装有4个预装的风扇和足够的空间，您不必担心好的电缆管理，这很好，因为我的电缆很糟糕。 它也是非常高评价和相当便宜。
