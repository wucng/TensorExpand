# 如何为新类别重新训练图像分类器
现代图像识别模型具有数百万个参数。 从头开始对它们进行培训需要大量标记的培训数据和大量计算能力（数百GPU时间或更长时间）。 转移学习是一种技巧，通过采用一个已经在相关任务上接受培训的模型并在新模型中重新使用模型来实现这一点。 在本教程中，我们将重用在ImageNet上训练的功能强大的图像分类器中的特征提取功能，并简单地在最上面训练一个新的分类层。 有关该方法的更多信息，请参阅[Decaf上的这篇论文](https://arxiv.org/abs/1310.1531)。

虽然它不如训练整个模型好，但这对于许多应用程序来说是非常有效的，适用于中等数量的训练数据（数千个，而不是数百万个标记的图像），并且可以在笔记本电脑上在短短的三十分钟内运行，而无需 一个GPU。 本教程将向您展示如何在自己的图像上运行示例脚本，并解释一些可帮助控制训练过程的选项。

本教程使用TensorFlow Hub来提取预先训练好的模型或模块。 对于初学者，我们将使用[图像特征提取模块](https://tensorflow.google.cn/modules/google/imagenet/inception_v3/feature_vector/1)以及在ImageNet上训练的Inception V3架构，稍后再回来进一步选择，包括[NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) / PNASNet以及[MobileNet V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)和V2。

开始之前，您需要安装PIP套件`tensorflow-hub`以及最新版本的TensorFlow。 有关详细信息，请参阅TensorFlow Hub的安装说明。

## Training on Flowers
在开始任何训练之前，您需要一组图像来向网络传授您想要识别的新class。 后面的章节解释了如何准备自己的图片，但为了方便起见，我们创建了创作共用许可花卉照片的存档，以供初始使用。 要获取花卉照片集，请运行以下命令：

```python
cd ~
curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz

```
一旦你有了图像，你可以从GitHub下载示例代码（它不是库安装的一部分）：

```python
mkdir ~/example_code
cd ~/example_code
curl -LO https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py
```
在最简单的情况下，可以像这样运行retrainer（大约需要半小时）：

```python
python3 retrain.py --image_dir ~/flower_photos
```
该脚本有许多其他选项。 您可以通过以下方式获得完整列表：

```
python3 retrain.py -h
```
该脚本加载预先训练的模块，并在顶部为所下载的花卉照片训练一个新的分类器。 全部网络都没有经过训练的原始ImageNet类别中的花卉种类。 转移学习的神奇之处在于已经被训练以区分一些对象的较低层可以被重复用于许多识别任务，而没有任何改变。

## Bottlenecks
该脚本可能需要30分钟或更长时间才能完成，具体取决于机器的速度。第一阶段分析磁盘上的所有图片并计算并缓存每个图片的瓶颈值。 “瓶颈”是一个非正式的术语，我们经常在最后一个输出层之前使用该层，实际进行分类。 （`TensorFlow Hub`称之为“图像特征向量”）。这个倒数第二层已经被训练输出一组足够好的值，以便分类器用来区分它被要求识别的所有类。这意味着它必须是对图像有意义和紧凑的总结，因为它必须包含足够的信息才能使分类器在很小的一组值中做出正确的选择。我们最后的层重新训练可以在新类上工作的原因是，它证明了区分ImageNet中所有1,000个类所需的信息通常对区分新类型的对象也很有用。

由于每个图像在训练过程中都会重复使用多次，并计算每个瓶颈需要大量时间，因此它可以加快速度，将这些瓶颈值缓存在磁盘上，因此无需重复计算。默认情况下，它们存储在`/tmp/bottleneck`目录中，如果您重新运行脚本，它们将被重用，因此您不必再等待此部分。

## Training
一旦瓶颈完成，网络顶层的实际培训就开始了。您会看到一系列步骤输出，每个输出都显示训练准确性，验证准确性和交叉熵。训练准确性显示当前训练批次中使用的图像的百分比是否标有正确的分类。验证的准确性是从不同集合中随机选择的一组图像的精度。关键的区别在于，训练的准确性基于网络能够学习的图像，因此网络可以适应训练数据中的噪声。衡量网络性能的一个真正衡量标准是衡量其在训练数据中未包含的数据集上的表现 - 这是通过验证准确度来衡量的。如果列车准确度高但验证准确度仍然较低，那意味着网络过度拟合并记忆训练图像中的特定功能，这些功能通常不会有帮助。交叉熵是一种损失函数，可以让我们看到学习过程的进展情况。培训的目标是让损失尽可能小，因此您可以通过关注损失是否持续下降趋势来判断学习是否奏效，而忽略短期噪音。

默认情况下，此脚本将运行4,000个训练步骤。每个步骤从训练集中随机选择10幅图像，从高速缓存中找出它们的瓶颈，并将它们送入最终图层进行预测。然后将这些预测与实际标签进行比较，以通过反向传播过程更新最终图层的权重。随着过程的继续，您应该看到所报告的准确性提高，并且在所有步骤完成后，将对一组图像进行最终测试准确性评估，并将其与训练和验证图片分开保存。此测试评估是对训练模型如何在分类任务上执行的最佳估计。您应该看到90％到95％之间的准确度值，但由于训练过程中存在随机性，所以准确值会因运行而异。这个数字是基于模型完全训练后给出正确标签的测试集中图像的百分比。

## Visualizing the Retraining with TensorBoard
该脚本包含TensorBoard摘要，使其更易于理解，调试和优化再训练。 例如，您可以将图表和统计数据可视化，例如训练期间权重或准确度如何变化。

要启动TensorBoard，请在再训练期间或之后运行此命令：

```
tensorboard --logdir /tmp/retrain_logs
```
一旦TensorBoard正在运行，浏览您的Web浏览器到`localhost:6006`以查看TensorBoard。

`retrain.py`脚本默认将TensorBoard摘要记录到`/tmp/retrain_logs`。 您可以使用`--summaries_dir`标志更改目录。

[TensorBoard的GitHub](https://github.com/tensorflow/tensorboard)存储库有关于TensorBoard使用情况的更多信息，包括提示和技巧以及调试信息。

## Using the Retrained Model
该脚本会将训练有关您类别的新模型写入`/tmp/output_graph.pb`，并将包含标签的文本文件写入`/tmp/output_labels.txt`。 新模型包含嵌入到其中的TF-Hub模块和新的分类层。 这两个文件都采用[C ++和Python图像分类示例](https://tensorflow.google.cn/tutorials/image_recognition)可以读入的格式，因此您可以立即开始使用新模型。 由于您已替换顶层，因此您需要在脚本中指定新名称，例如，如果使用的是label_image，则使用`--output_layer = final_result`标志。

以下是如何在训练图中运行label_image示例的示例。 按照惯例，所有TensorFlow Hub模块都会接受图像输入，其颜色值在固定范围`[0,1]`内，因此您不需要设置`--input_mean`或`--input_std`标志。

```python
curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg

```
你应该看到一个花标签列表，在大多数情况下菊花在上面（尽管每个重新训练的模型可能略有不同）。 你可以用你自己的图像替换--image参数来尝试这些。

如果你想在自己的Python程序中使用再训练模型，那么上面的label_image脚本是一个合理的起点。 label_image目录还包含C ++代码，您可以将它用作模板以将tensorflow与您自己的应用程序集成在一起。

如果您发现默认的Inception V3模块对于您的应用程序太大或太慢，请查看下面的“[其他模型架构](https://tensorflow.google.cn/tutorials/image_retraining#other_architectures)”部分，以了解如何加速和缩小您的网络。

## Training on Your Own Categories
如果你已经设法让这个脚本处理花图像，你可以开始寻找教它识别你关心的类别。 从理论上讲，你需要做的就是将它指向一组子文件夹，每个子文件夹以你的一个类别命名并仅包含该类别的图像。 如果你这样做，并将子目录的根文件夹作为参数传递给`--image_dir`，那么脚本应该像对花一样进行训练。

以下是鲜花档案的文件夹结构，以便为您提供脚本所寻找布局的示例：
![这里写图片描述](https://tensorflow.google.cn/images/folder_structure.png)

在实践中，可能需要一些工作来获得所需的准确性。 我会尽力引导您解决下面可能遇到的一些常见问题。

## Creating a Set of Training Images
首先要看看你收集的图像，因为我们通过训练看到的最常见的问题来自所馈入的数据。

要使训练运作良好，您应该至少收集一百张您想要识别的各种对象的照片。你可以收集的越多，训练好的模型的准确性就越好。您还需要确保**照片能够很好地表现您的应用程序实际会遇到的情况**。例如，如果您将所有照片放在室内的空白墙壁上，并且用户尝试识别室外的物体，则在部署时可能看不到良好的效果。

要避免的**另一个缺陷**是学习过程会在标签图像彼此共有的任何东西上出现，如果您不小心，那可能是没用的。例如，如果您在蓝色房间中拍摄一种物体，而另一种物体拍摄为绿色，则模型最终将基于其背景颜色的预测，而不是您实际关心的物体的特征。为避免这种情况，请尝试在各种情况下尽可能在不同时间和不同设备上拍摄照片。

您可能还想考虑您使用的类别。将大量不同的物理形式分成较小的物体可能是值得的，这些小物体在视觉上更加独特。例如，可以使用“汽车”，“摩托车”和“卡车”来代替“车辆”。同样值得考虑一下你是否拥有“封闭世界”或“开放世界”的问题。在封闭的世界里，你唯一要求分类的东西就是你所了解的对象类。这可能适用于植物识别应用程序，你知道用户可能正在拍摄一朵花，所以你只需要决定哪些物种。相比之下，漫游机器人可能会通过其摄像头在世界各地漫游时看到各种不同的东西。在这种情况下，您希望分类器报告它是否不确定它看到的是什么。这可能很难做到，但是如果您经常收集大量没有相关对象的典型“背景”照片，则可以将它们添加到图像文件夹中额外的“unknown”类。

这也值得检查，以确保所有的图像都标有正确的标签。用户生成的标签通常不适合我们的用途。例如：标有#daisy的图片也可能包含名为Daisy的人物和角色。如果你仔细检查你的图片并清除任何错误，它可以为你的整体准确性做出奇迹。

## Training Steps
如果您对图像满意，可以通过改变学习过程的细节来改善您的结果。 最简单的尝试是`--how_many_training_steps`。 默认值为4,000，但如果将其增加到8,000，则训练时间会延长两倍。 准确度提高的速度减慢了你训练的时间，并且在某些时候会完全停止（甚至由于过度拟合而下降），但是你可以尝试看看什么模型最适合你。

## Distortions
改善图像训练结果的一种常见方式是以**随机方式**对训练输入进行变形，裁剪或增亮。这有利于扩大训练数据的有效大小，这要归功于相同图像的所有可能的变化，并且倾向于帮助网络学会应对在分类器的实际使用中将发生的所有失真。在我们的脚本中实现这些扭曲的最大缺点是瓶颈缓存不再有用，因为输入图像永远不会重复使用。这意味着训练过程需要更长的时间（很多小时），所以建议您尝试这种方法，只有在您有合适的满意后才能打磨模型。

您可以通过将`--random_crop`，`--random_scale`和`--random_brightness`传递给脚本来启用这些扭曲。这些都是控制每个图像应用了多少扭曲的百分比值。对每个人开始5或10的值是合理的，然后试验看看他们哪些人可以帮助你的应用程序。 `--flip_left_right`将水平随机镜像一半图像，只要这些反转很可能在您的应用程序中发生，这是有意义的。例如，如果你试图识别字母，这不是一个好主意，因为翻转它们会破坏它们的意义。

## Hyper-parameters
还有其他几个参数可以尝试调整，以查看它们是否有助于您的结果。 `--learning_rate`控制训练过程中更新到最后一层的大小。 直观地说，如果这个比较小，那么学习需要更长的时间，但最终可能会帮助整体的精确度。 但情况并非总是如此，所以您需要仔细试验以了解适合您的情况。 `--train_batch_size`控制在每个训练步骤中检查多少图像以估计最终图层的更新。

## Training, Validation, and Testing Sets
当您将脚本指向一个图像文件夹时，脚本在引擎盖下进行的操作之一是将它们分成三组。最大的通常是训练集，它是训练期间输入网络的所有图像，结果用于更新模型的权重。您可能想知道为什么我们不使用所有图像进行培训？当我们进行机器学习时，一个很大的潜在问题是我们的模型可能只是记住训练图像的不相关细节，以提出正确的答案。例如，您可以想象一个网络在每张照片的背景中记住一个模式，并使用它来将标签与对象进行匹配。它可以在训练过程中看到的所有图像上产生良好的效果，但是由于没有学习到物体的一般特性，只记住了训练图像的不重要细节，因此在新图像上失败。

这个问题被称为过度拟合，为了避免它，我们将一些数据保留在训练过程之外，以便模型不能记住它们。然后，我们使用这些图像作为检查来确保过度拟合不会发生，因为如果我们看到它们有很好的准确性，这是一个很好的迹象表明网络不是过度配合。通常的做法是将80％的图像放入主要的训练集中，保留10％作为训练期间的频繁验证运行，然后最终使用10％作为测试集来预测实时数据，分类器的世界表现。这些比率可以使用`--testing_percentage`和`--validation_percentage`标志进行控制。一般来说，您应该能够将这些值保留为默认值，因为您通常无法找到培训来调整它们的优势。

请注意，该脚本使用图像文件名（而不是完全随机的函数）在训练，验证和测试集之间划分图像。这样做是为了确保图像不会在不同运行的训练集和测试集之间移动，因为如果用于训练模型的图像随后用于验证集，那么这可能会成为问题。

您可能会注意到验证准确性在迭代中波动。这种波动的很大一部分源自这样的事实，即为每个验证准确性测量选择验证集合的随机子集。通过选择`--validation_batch_size = -1`，可以大大降低波动，但需要增加一些培训时间，其中每个精度计算使用整个验证集。

一旦训练完成，您可能会发现在测试集中检查错误分类的图像是很有见地的。这可以通过添加标志`--print_misclassified_test_images`来完成。这可能有助于您了解模型中哪些类型的图像最容易混淆，哪些类别最难区分。例如，您可能会发现某个特定类别的某个子类型或某种不寻常的照片角度特别难以识别，这可能会鼓励您添加更多该子类型的训练图像。通常，检查错误分类的图像也可能会指出输入数据集中的错误，如错误标记，低质量或模糊的图像。但是，通常应避免在测试集中修正个别错误，因为它们可能仅仅反映（更大）训练集中的更一般问题。

## Other Model Architectures
默认情况下，脚本使用图像特征提取模块，其中包含Inception V3体系结构的预训练实例。 这是一个很好的开始，因为它为再训练脚本提供了准确的结果和适度的运行时间。 但现在让我们来看看[TensorFlow Hub模块的更多选项](https://tensorflow.google.cn/modules/image)。

一方面，该列表显示了更新的强大体系结构，如[NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)（特别是[nasnet_large](https://tensorflow.google.cn/modules/google/imagenet/nasnet_large/feature_vector/1)和[pnasnet_large](https://tensorflow.google.cn/modules/google/imagenet/pnasnet_large/feature_vector/1)），它们可以为您提供一些额外的精度。

另一方面，如果您打算在移动设备或其他资源受限的环境中部署您的模型，您可能希望以较小的文件大小或更快的速度（也可以在训练中）交易一点精度。 为此，请尝试实现[MobileNet V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)或V2体系结构的[不同模块](https://tensorflow.google.cn/modules/image#mobilenet)，或尝试使用[nasnet_mobile](https://tensorflow.google.cn/modules/google/imagenet/nasnet_mobile/feature_vector/1)。

使用不同的模块进行训练很简单：只需将模块URL传递给`--tfhub_module`标志，例如：

```python
python retrain.py \
    --image_dir ~/flower_photos \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1

```
这将在`/tmp/output_graph.pb`中创建一个使用MobileNet V2基准版本的9 MB模型文件。 在浏览器中打开模块URL将会转到模块文档。

如果您只是想让它更快一点，可以将输入图像的大小（第二个数字）从'224'减小到'192'，'160'或'128'像素的平方，甚至'96' （仅适用于V2）。 要获得更积极的节省，您可以选择百分比（第一个数字）“100”，“075”，“050”或“035”（V1为'025'）来控制每个神经元的“特征深度” 位置。 权重的数量（以及文件大小和速度）随着该分数的平方收缩。 GnetHub上的[MobileNet V1 blogpost ](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)和[MobileNet V2页报告](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)了Imagenet分类的相应权衡。

Mobilenet V2不会将特征深度百分比应用于瓶颈层。 Mobilenet V1的确做到了，这使得分类层的工作更加困难。 它会帮助欺骗并使用原来的1001 ImageNet类的分数而不是严格的瓶颈？ 您可以简单地尝试在模块名称中将`mobilenet_v1.../feature_vector`替换为`mobilenet_v1.../classification`。

像以前一样，您可以将所有重新培训的模型与label_image.py一起使用。 您将需要指定您的模型预期的图像大小，例如：

像以前一样，您可以将所有重新训练的模型与`label_image.py`一起使用。 您将需要指定您的模型预期的图像大小，例如：

```shell
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--input_height=224 --input_width=224 \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg

```

有关将再训练模型部署到移动设备的更多信息，请参阅本教程的[codelab版本](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)，特别是[第2部分](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#0)，其中介绍了[TensorFlow Lite](https://tensorflow.google.cn/mobile/tflite/)及其提供的其他优化（包括模型权重的量化）。
