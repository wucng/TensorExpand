[MNIST in TensorFlow](https://github.com/tensorflow/models/tree/master/official/mnist)

---
该目录构建卷积神经网络，使用[tf.data](https://www.tensorflow.org/api_docs/python/tf/data)，[tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)和[tf.layers](https://www.tensorflow.org/api_docs/python/tf/layers) API对[MNIST数据集](http://yann.lecun.com/exdb/mnist/)进行分类。

# Setup
首先，您只需要安装最新版本的TensorFlow。 首先确保已将models文件夹添加到Python路径中; 否则您可能会遇到类似ImportError的错误：没有名为official.mnist的模块。

然后训练模型，运行以下命令：

```
python mnist.py
```
该模型将开始训练，并将自动评估验证数据。

可以使用以下命令运行说明性单元测试和基准测试：

```
python mnist_test.py
python mnist_test.py --benchmarks=.
```
# Exporting the model
您可以使用参数`--export_dir`将模型导出为Tensorflow [SavedModel](https://www.tensorflow.org/guide/saved_model)格式：

```
python mnist.py --export_dir /tmp/mnist_saved_model
```
SavedModel将保存在`/tmp/mnist_saved_model/`下的带时间戳的目录中（例如`/tmp/mnist_saved_model/1513630966/`）。

使用SavedModel获取预测使用[saved_model_cli](https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel)检查并执行SavedModel。

```
saved_model_cli run --dir /tmp/mnist_saved_model/TIMESTAMP --tag_set serve --signature_def classify --inputs image=examples.npy
```
examples.npy按顺序包含numpy数组中example5.png和example3.png的数据。 数组值标准化为0到1之间的值。

输出应类似于以下内容：

```
Result for output key classes:
[5 3]
Result for output key probabilities:
[[  1.53558474e-07   1.95694142e-13   1.31193523e-09   5.47467265e-03
    5.85711526e-22   9.94520664e-01   3.48423509e-06   2.65365645e-17
    9.78631419e-07   3.15522470e-08]
 [  1.22413359e-04   5.87615965e-08   1.72251271e-06   9.39960718e-01
    3.30306928e-11   2.87386645e-02   2.82353517e-02   8.21146413e-18
    2.52568233e-03   4.15460236e-04]]
```
# Experimental: Eager Execution
可以使用以下方法训练mnist.py中定义的完全相同的模型，而无需创建TensorFlow图：

```
python mnist_eager.py
```
# Experimental: TPU Acceleration
`mnist.py`（和`mnist_eager.py`）演示了如何训练神经网络来对CPU和GPU上的数字进行分类。 `mnist_tpu.py`可用于使用TPU训练相同的模型以进行硬件加速。 [tensorflow/tpu](https://github.com/tensorflow/tpu)存储库中的更多信息。
