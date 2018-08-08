- [Image Retraining](https://tensorflow.google.cn/hub/tutorials/image_retraining)

---
# 如何为新类别重新训练图像分类器

现代图像识别模型具有数百万个参数。从头开始训练需要大量标记的训练数据和大量计算能力（数百小时GPU或更多）。转移学习是一种技术，通过采用一个已经在相关任务上训练并在新模型中重复使用的模型来快速完成大部分工作。在本教程中，我们将重用ImageNet上训练的强大图像分类器的特征提取功能，并简单地在顶部训练新的分类层。有关该方法的更多信息，您可以在Decaf上看到这篇论文。

虽然它不如训练整个模型那么好，但对于许多应用来说这是非常有效的，适用于适量的训练数据（数千，而不是数百万标记的图像），并且可以在笔记本电脑上运行30分钟而不需要一个GPU。本教程将向您展示如何在您自己的图像上运行示例脚本，并将解释您有助于控制训练过程的一些选项。

本教程使用[TensorFlow Hub](https://tensorflow.google.cn/hub/)来调用预先训练过的模型或模块。 对于初学者，我们将使用具有在ImageNet上训练的Inception V3架构的[图像特征提取模块](https://tensorflow.google.cn/hub/modules/google/imagenet/inception_v3/feature_vector/1)，并[稍后返回其他选项](https://tensorflow.google.cn/hub/tutorials/image_retraining#other_architectures)，包括[NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html) / PNASNet，以及[MobileNet V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)和V2。

在开始之前，您需要安装`pip3 install tensorflow-hub`以及最新版本的`TensorFlow`。 有关详细信息，请参阅[TensorFlow Hub的安装说明](https://tensorflow.google.cn/hub/installation)。

## Training on Flowers
在开始任何训练之前，您需要一组图像来向网络传授您想要识别的新类。 后面的部分介绍了如何准备自己的图像，但为了方便起见，我们创建了一个创建公共许可花卉照片的存档，以便最初使用。 要获取花卉照片集，请运行以下命令：

```
cd ~
curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
```
获得图像后，可以从GitHub下载示例代码（它不是库安装的一部分）：

```
mkdir ~/example_code
cd ~/example_code
curl -LO https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py
```
在最简单的情况下，训练可以这样运行（大约需要半小时）：

```
python retrain.py --image_dir ~/flower_photos
```
该脚本还有许多其他选项。 您可以通过以下方式获得完整列表：

```
python retrain.py -h
```
此脚本加载预先训练的模块，并在您下载的花卉照片的顶部训练新的分类器。 在完整的网络训练过程中，没有一种花卉种类属于原始的ImageNet类。 转移学习的神奇之处在于，经过训练以区分某些对象的较低层可以重复用于许多识别任务而无需任何改动。

## Bottlenecks
根据机器的速度，脚本可能需要30分钟或更长时间才能完成。第一阶段分析磁盘上的所有映像，并计算和缓存每个映像的瓶颈值。 '瓶颈'是一个非正式术语，我们经常在实际进行分类的最终输出层之前使用该层。 （TensorFlow Hub将其称为“图像特征向量”。）此倒数第二层已经过训练，可以输出一组足够好的值，分类器可以用它来区分要求识别的所有类。这意味着它必须是一个有意义且紧凑的图像摘要，因为它必须包含足够的信息，以便分类器在一组非常小的值中做出正确的选择。我们的最后一层再训练可以在新类上工作的原因是，结果表明，区分ImageNet中所有1,000个类所需的信息通常也可用于区分新类型的对象。

因为每个图像在训练期间多次重复使用并且计算每个瓶颈需要花费大量时间，所以它会加速将这些瓶颈值缓存到磁盘上，因此不必反复重新计算它们。默认情况下，它们存储在`/tmp/bottleneck`目录中，如果重新运行脚本，它们将被重用，因此您不必再次等待此部分。

## Training
一旦瓶颈完成，就开始对网络顶层进行实际培训。您将看到一系列步骤输出，每个步骤输出显示训练准确性，验证准确性和交叉熵。训练准确性显示当前训练批次中使用的图像百分比标记为正确的类别。验证准确度是来自不同组的随机选择的图像组的精度。关键的区别在于训练精度基于网络能够学习的图像，因此网络可以过度拟合训练数据中的噪声。衡量网络性能的真正标准是测量其在训练数据中未包含的数据集上的性能 - 这是通过验证准确度来衡量的。如果列车精度高但验证精度仍然很低，则意味着网络过度拟合并记住训练图像中的特定特征，这些特征对于更一般无用。交叉熵是一种损失函数，可以让我们一瞥学习过程的进展情况。培训的目标是尽可能减少损失，因此您可以通过关注损失是否保持向下趋势，忽略短期噪音来判断学习是否有效。

默认情况下，此脚本将运行4,000个训练步骤。每个步骤从训练集中随机选择十个图像，从缓存中找到它们的瓶颈，并将它们输入到最后一层以获得预测。然后将这些预测与实际标签进行比较，以通过反向传播过程更新最终层的权重。随着过程的继续，您应该看到报告的准确度得到改善，并且在完成所有步骤之后，对与训练和验证图片分开的一组图像运行最终测试准确度评估。该测试评估是训练模型将如何在分类任务上执行的最佳估计。您应该看到准确度值介于90％和95％之间，但是由于训练过程中的随机性，准确值会因批次不同而不同。此数字基于完全训练模型后给定正确标签的测试集中图像的百分比。

## 使用TensorBoard可视化再训练
该脚本包含TensorBoard摘要，可以更容易理解，调试和优化再培训。 例如，您可以可视化图形和统计数据，例如在训练期间权重或准确度如何变化。

要启动TensorBoard，请在重新训练期间或之后运行此命令：

```
tensorboard --logdir /tmp/retrain_logs
```
TensorBoard运行后，将Web浏览器导航到localhost：6006以查看TensorBoard。

rewin.py脚本默认情况下会将TensorBoard摘要记录到`/tmp/retrain_logs`。 您可以使用`--summaries_dir`标志更改目录。

TensorBoard的GitHub存储库提供了有关TensorBoard使用的更多信息，包括提示和技巧以及调试信息。

## Using the Retrained Model
该脚本会将在您的类别上训练的新模型写入`/tmp/output_graph.pb`，并将包含标签的文本文件写入`/tmp/output_labels.txt`。 新模型包含内嵌的TF-Hub模块和新的分类层。 这两个文件都采用C ++和Python图像分类示例可以读入的格式，因此您可以立即开始使用新模型。 由于您已经替换了顶层，因此您需要在脚本中指定新名称，例如，如果您使用`label_image`，则使用标志`--output_layer = final_result`。

以下是如何使用重新训练的图形运行`label_image`示例的示例。 按照惯例，所有TensorFlow Hub模块都接受具有固定范围[0,1]中颜色值的图像输入，因此您无需设置`--input_mean`或`--input_std`标志。

```
curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```
你应该看到一个花卉标签列表，在大多数情况下顶部有菊花（虽然每个重新训练的模型可能会略有不同）。 您可以使用自己的图像替换`--image`参数来尝试这些。

如果您想在自己的Python程序中使用重新训练的模型，那么上面的`label_image`脚本是一个合理的起点。 label_image目录还包含C ++代码，您可以将其用作模板，将tensorflow与您自己的应用程序集成。

如果您发现默认的Inception V3模块对于您的应用程序来说太大或太慢，请查看下面的“其他模型架构”部分，了解加快和缩小网络的选项。

## Training on Your Own Categories
如果您已经设法让脚本处理花卉示例图像，您可以开始考虑教它来识别您关心的类别。 理论上，您需要做的就是将其指向一组子文件夹，每个子文件夹以您的一个类别命名，并且仅包含该类别的图像。 如果您这样做并将子目录的根文件夹作为参数传递给--image_dir，则脚本应该像对花一样训练。

以下是鲜花存档的文件夹结构，为您提供脚本所需布局类型的示例：
![这里写图片描述](https://tensorflow.google.cn/images/folder_structure.png)

在实践中，可能需要一些工作来获得您想要的准确性。 我将尝试引导您解决下面可能遇到的一些常见问题。

## Creating a Set of Training Images
首先要看的是你收集的图像，因为我们通过培训看到的最常见的问题来自于被输入的数据。

为了使训练更好地运作，您应该收集至少一百张您想要识别的每种物体的照片。您收集的越多，您训练的模型的准确性就越高。您还需要确保照片很好地代表了您的应用程序实际遇到的内容。例如，如果您将所有照片都放在室内空白墙上​​并且用户试图在户外识别物体，则部署时可能看不到好的结果。

要避免的另一个缺陷是，学习过程会对标记图像彼此相同的任何内容产生影响，如果你不小心，那可能是没用的东西。例如，如果您在蓝色房间中拍摄一种物体，而另一种物体在绿色物体中拍摄，则模型最终会根据背景颜色进行预测，而不是您实际关注的物体的特征。为避免这种情况，请尝试在不同时间和不同设备上尽可能多地拍摄照片。

您可能还想考虑您使用的类别。将大量不同物理形式的大类别划分为更具视觉冲突力的小类别可能是值得的。例如，您可以使用“汽车”，“摩托车”和“卡车”代替“车辆”。同样值得思考的是你是否有“封闭世界”或“开放世界”问题。在一个封闭的世界中，你唯一要求分类的东西就是你所知道的对象类别。这可能适用于您知道用户可能正在拍摄花卉照片的植物识别应用程序，因此您所要做的就是决定使用哪种物种。相比之下，漫游机器人可以通过其相机在世界各地漫游时看到各种不同的东西。在这种情况下，您希望分类器报告它是否不确定它看到了什么。这可能很难做到，但通常如果您收集大量典型的“背景”照片而其中没有相关对象，您可以将它们添加到图像文件夹中的额外“未知”类。

还需要检查以确保所有图像都标记正确。用户生成的标签通常不能用于我们的目的。例如：标记为#daisy的图片也可能包含名为Daisy的人物和角色。如果您浏览图像并清除任何错误，它可以为您的整体准确性创造奇迹。

## Training Steps
如果您对图像感到满意，可以通过更改学习过程的详细信息来了解改善结果的方法。 最简单的尝试是--how_many_training_steps。 默认为4,000，但如果将其增加到8,000，它将训练两倍。 准确度的提高速度会减慢你训练的时间，并且在某些时候会完全停止（甚至因过度拟合而下降），但你可以尝试看看什么最适合你的模型。

## Distortions
改善图像训练结果的常用方法是以随机方式使训练输入变形，裁剪或增亮。由于相同图像的所有可能变化，这具有扩展训练数据的有效大小的优点，并且倾向于帮助网络学习应对将在分类器的实际使用中发生的所有失真。在我们的脚本中启用这些失真的最大缺点是瓶颈缓存不再有用，因为输入图像永远不会被完全重用。这意味着训练过程需要更长的时间（许多小时），因此建议您尝试将此作为一种抛光模型的方法，只有在您拥有一个相当满意的模型之后。

通过将`--random_crop`， `- random_scale`和`--random_brightness`传递给脚本来启用这些失真。这些都是控制每个图像应用了多少失真的百分比值。从每个值的5或10开始是合理的，然后试验看哪些值对您的应用有帮助。` --flip_left_right`将水平地随机镜像一半图像，只要这些反转可能在您的应用程序中发生，这是有意义的。例如，如果你试图识别字母，这不是一个好主意，因为翻转它们会破坏它们的含义。

## Hyper-parameters
您可以尝试调整其他几个参数，看看它们是否有助于您的结果。` --learning_rate`控制训练期间最后一层更新的大小。 直观地，如果这个小的学习率将花费更长时间，但它最终可以帮助整体精确度。 但情况并非总是这样，所以你需要仔细研究，看看哪种方法适合你的情况。 `--train_batch_size`控制在每个训练步骤中检查的图像数量，以估计最终图层的更新。

## Training, Validation, and Testing Sets
当您将脚本指向图像文件夹时，脚本所做的一件事就是将它们分成三组。最大的通常是训练集，它们是训练期间馈入网络的所有图像，其结果用于更新模型的权重。您可能想知道为什么我们不使用所有图像进行培训？当我们进行机器学习时，一个很大的潜在问题是我们的模型可能只是记住训练图像的不相关细节，以得出正确的答案。例如，您可以想象一个网络在其显示的每张照片的背景中记住一个图案，并使用它来匹配标签与对象。它可以在训练之前看到的所有图像上产生良好的效果，但是然后在新图像上失败，因为它没有学习对象的一般特征，只记住训练图像的不重要细节。

这个问题被称为过度拟合，为了避免它，我们将一些数据保留在训练过程之外，这样模型就无法记住它们。然后我们使用这些图像作为检查以确保不会发生过度拟合，因为如果我们看到它们具有良好的准确性，则表明网络没有过度拟合是一个好兆头。通常的分割是将80％的图像放入主训练集中，保留10％以备在训练期间经常作为验证运行，然后将最终的10％作为测试集用于预测实际情况世界表现的分类器。可以使用--testing_percentage和--validation_percentage标志来控制这些比率。一般情况下，您应该能够将这些值保留为默认值，因为通常不会发现培训调整它们的任何优势。

请注意，该脚本使用图像文件名（而不是完全随机的函数）在训练，验证和测试集之间划分图像。这样做是为了确保图像不会在不同运行的训练集和测试集之间移动，因为如果用于训练模型的图像随后在验证集中使用，那么这可能是一个问题。

您可能会注意到验证准确度在迭代之间波动。大部分这种波动源于这样的事实：为每个验证精度测量选择验证集的随机子集。通过选择--validation_batch_size = -1，使用整个验证集进行每次精度计算，可以大大减少波动，代价是培训时间有所增加。

培训完成后，您可能会发现检查测试集中错误分类的图像非常有见地。这可以通过添加标志`--print_misclassified_test_images`来完成。这可以帮助您了解哪种类型的图像最容易混淆模型，以及哪些类别最难以区分。例如，您可能会发现特定类别的某些子类型或某些不寻常的照片角度特别难以识别，这可能会鼓励您添加该子类型的更多训练图像。通常，检查错误分类的图像也可能指向输入数据集中的错误，例如错误标记，低质量或模糊图像。然而，通常应该避免在测试集中修正个别错误，因为它们可能仅仅反映（更大）训练集中的更一般的问题。

## Other Model Architectures
默认情况下，脚本使用带有Inception V3体系结构预训练实例的图像特征提取模块。 这是一个很好的起点，因为它为再训练脚本提供了高精度的结果和适中的运行时间。 但现在让我们来看看[TensorFlow Hub模块的其他选项](https://tensorflow.google.cn/hub/modules/image)。

一方面，该列表显示了更新的，功能强大的体系结构，例如[NASNet](https://research.googleblog.com/2017/11/automl-for-large-scale-image.html)（特别是[nasnet_large](https://tensorflow.google.cn/hub/modules/google/imagenet/nasnet_large/feature_vector/1)和[pnasnet_large](https://tensorflow.google.cn/hub/modules/google/imagenet/pnasnet_large/feature_vector/1)），它可以为您提供额外的精确度。

另一方面，如果您打算在移动设备或其他资源受限的环境中部署模型，您可能希望以更小的文件大小或更快的速度（也在培训中）交换一点精度。 为此，尝试实现[MobileNet V1](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)或V2架构的不同[模块](https://tensorflow.google.cn/hub/modules/image#mobilenet)，或者[nasnet_mobile](https://tensorflow.google.cn/hub/modules/google/imagenet/nasnet_mobile/feature_vector/1)。

使用不同模块进行培训很简单：只需将--tfhub_module标志与模块URL一起传递，例如：

```
python retrain.py \
    --image_dir ~/flower_photos \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1
```
这将在`/tmp/output_graph.pb`中创建一个9 MB的模型文件，其中的模型使用MobileNet V2的基线版本。在浏览器中打开模块URL将转到模块文档。

如果你只是想让它快一点，你可以将输入图像（第二个数字）的大小从'224'减小到'192'，'160'或'128'像素的平方，甚至'96' （仅适用于V2）。为了更积极的节省，您可以选择百分比（第一个数字）'100'，'075'，'050'或'035'（V1的'025'）来控制“特征深度”或每个神经元的数量位置。权重的数量（以及文件大小和速度）随着该分数的平方而缩小。 GitHub上的MobileNet V1博文和MobileNet V2页面报告了Imagenet分类的各自权衡。

Mobilenet V2不会将功能深度百分比应用于瓶颈层。 Mobilenet V1做到了，这使得分类层的工作对于小深度更难。是否有助于欺骗和使用原始1001 ImageNet类的分数而不是严格的瓶颈？您可以尝试将removenet_v1 ... / feature_vector替换为模块名称中的mobilenet_v1 ... / classification。

和以前一样，您可以将所有重新训练的模型与label_image.py一起使用。您需要指定模型所需的图像大小，例如：

```
python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--input_height=224 --input_width=224 \
--image=$HOME/flower_photos/daisy/21652746_cc379e0eea_m.jpg
```
有关将重新训练的模型部署到移动设备的更多信息，请参阅本教程的codelab版本，尤其是[第2部分](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0)，其中介绍了[TensorFlow Lite](https://tensorflow.google.cn/mobile/tflite/)及其提供的其他优化（包括模型权重的量化）。