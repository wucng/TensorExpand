# [TensorFlow Official Models](https://github.com/tensorflow/models/tree/master/official)

TensorFlow官方模型是使用TensorFlow的高级API的示例模型的集合。 它们旨在通过最新的TensorFlow API进行良好维护，测试和更新。 它们还应进行合理优化，以便在保持易读性的同时实现快速性能。

这些模型用作端到端测试，确保模型在每个新的TensorFlow构建中以相同的速度和性能运行。

# Tensorflow releases
模型的主分支正在开发中，它们针对从TensorFlow的主分支构建的夜间二进制文件。 我们的目标是尽可能使它们与最新版本保持向后兼容（目前是TensorFlow 1.5），但我们不能总是保证兼容性。

针对TensorFlow版本的官方模型的稳定版本可用作标记分支或可下载版本。 模型存储库版本号与目标TensorFlow版本匹配，因此分支r1.4.0和版本v1.4.0与TensorFlow v1.4.0兼容。

如果您使用的是早于1.4的TensorFlow版本，请更新您的安装。

# Requirements
在此仓库中运行模型之前，请按照以下步骤操作：

1、使用以下命令将top-level `/models`文件夹添加到Python路径：

```
import sys
sys.path # 找到python加载路径

# ------------------
vim /etc/profile

export PYTHONPATH=/usr/lib/python3/dist-packages/
export PYTHONPATH=$PYTHONPATH:xxxx/models/
export PYTHONPATH=$PYTHONPATH:xxxx/models/official/
export PYTHONPATH=$PYTHONPATH:xxxx/models/research/
export PYTHONPATH=$PYTHONPATH:xxxx/models/samples/
export PYTHONPATH=$PYTHONPATH:xxxx/models/tutorials/

保存完成后：source /etc/profile

# or
在程序中添加以下语句

import sys
sys.path.append('xxxx/models/')
sys.path.append('xxxx/models/official/')
sys.path.append('xxxx/models/research/')
sys.path.append('xxxx/models/samples/')
sys.path.append('xxxx/models/tutorials/')

```

2、安装依赖项：

```
pip3 install --user -r official/requirements.txt
# or
pip install --user -r official/requirements.txt
```
为了使官方模型更易于使用，我们计划创建一个可安装pip的官方模型包。 这正在[＃917](https://github.com/tensorflow/models/issues/917)中进行跟踪。
# Available models
注意：请确保按照“ `Requirements`”部分中的步骤操作。

- boosted_trees：一个Gradient Boosted Trees模型，用于对来自HIGGS数据集的higgs玻色子过程进行分类。
- mnist：从MNIST数据集中对数字进行分类的基本模型。
- resnet：一个深度残留网络，可用于对CIFAR-10和ImageNet的1000个类的数据集进行分类。
- transformer：用于将WMT英语转换为德语数据集的变换器模型。
- wide_deep：一种结合了广泛模型和深度网络的模型，用于对人口普查收入数据进行分类。
- 更多型号来！

如果您想对模型进行任何修复或改进，请提交拉取请求。

# New Models
该团队正积极致力于将新模型添加到存储库。 每个模型都应遵循以下准则，以维护可读，可用和可维护代码的目标。

## General guidelines
- 代码应有充分的文档和测试。
- 可以相对轻松地从空白环境中运行。
- 可训练：单GPU / CPU（基线），多GPU，TPU
- 兼容Python 2和3（必要时使用六个）
- 符合[Google Python样式指南](https://google.github.io/styleguide/pyguide.html)

## Implementation guidelines
存在这些指导原则，因此模型实现是一致的，以提高可读性和可维护性。

- 使用常用的实用功能
- 在培训结束时导出SavedModel。
- 一致的标志和标志解析库（在这里阅读更多）
- 制作基准和日志（在此处阅读更多）
