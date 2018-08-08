[Classifying Higgs boson processes in the HIGGS Data Set](https://github.com/tensorflow/models/tree/master/official/boosted_trees)

---
[HIGGS数据集](https://archive.ics.uci.edu/ml/datasets/HIGGS)包含1100万个样本，具有28个特征，用于分类问题，以区分产生希格斯玻色子的信号过程和不产生希格斯玻色子的后台过程。

我们使用Gradient Boosted Trees算法来区分这两个类。

---
代码示例使用高级别`tf.estimator.Estimator`和`tf.data.Dataset`。这些API非常适合快速迭代并快速使模型适应您自己的数据集而无需进行重大代码检修。它允许您从单工作者培训转移到分布式培训，并且可以轻松导出模型二进制文件以进行预测。这里，为了进一步简化和更快的执行，我们使用实用函数`tf.contrib.estimator.boosted_trees_classifier_train_in_memory`。当输入作为内存数据集（如numpy数组）提供时，此实用程序功能尤其有效。

`Estimator`的输入函数通常使用`tf.data.Dataset` API，它可以处理各种数据控制，如流，批处理，转换和混洗。但是，`boosted_trees_classifier_train_in_memory（）`实用程序函数要求将整个数据作为单个批处理提供（即不使用batch（）API）。因此，在本实践中，只使用`Dataset.from_tensors（）`将numpy数组转换为结构化tensor，`Dataset.zip（）`用于将要素和标签放在一起。有关数据集的更多参考，请在此[处阅读更多内容](https://www.tensorflow.org/guide/datasets)。

# Running the code
首先确保已将models文件夹添加到Python路径中; 否则您可能会遇到类似`ImportError: No module named official.boosted_trees`。

# Setup
此示例用于培训的HIGGS数据集由UC Irvine机器学习库托管。 我们提供了一个下载和清理必要文件的脚本。

```
python data_download.py
```
这将下载一个文件并将处理过的文件存储在`--data_dir`指定的目录下（默认为`/tmp/higgs_data/`）。 要更改目标目录，请设置`--data_dir`标志。 该目录可以是Tensorflow支持的网络存储（如Google Cloud Storage，`gs：// <bucket> / <path> /`）。 下载到本地临时文件夹的文件大约是2.8 GB，处理过的文件大约是0.8 GB，因此应该有足够的存储空间来处理它们。

# Training
此示例在训练期间使用大约3 GB的RAM。 您可以在本地运行代码，如下所示：

```
python train_higgs.py
```
该模型默认保存为`/tmp/higgs_model`，可以使用`--model_dir`标志更改。 请注意，每次训练开始之前都会清理`model_dir`。

模型参数可以通过标志来调整，例如`--n_trees`， `- max_depth`，`--learning_rate`等。 查看代码了解详细信息。

当使用默认参数训练时，最终精度将在74％左右，并且在eval集上损失将大约为0.516。

默认情况下，1100万个中的前100万个示例用于训练，最后100万个示例用于评估。 可以通过标志`--train_start`，` - train_count`， `- eval_start`， `- eval_count`等选择训练/评估数据作为索引范围。

# TensorBoard
运行TensorBoard以检查有关图表和培训进度的详细信息。

```
tensorboard --logdir=/tmp/higgs_model  # set logdir as --model_dir set during training.
```
# Inference with SavedModel
您可以使用参数`--export_dir`将模型导出为Tensorflow SavedModel格式：

```
python train_higgs.py --export_dir /tmp/higgs_boosted_trees_saved_model
```
模型完成训练后，使用`saved_model_cli`检查并执行SavedModel。

请尝试以下命令来检查SavedModel：

将$ {TIMESTAMP}替换为生成的文件夹（例如1524249124）

```python
# List possible tag_sets. Only one metagraph is saved, so there will be one option.
saved_model_cli show --dir /tmp/higgs_boosted_trees_saved_model/${TIMESTAMP}/

# Show SignatureDefs for tag_set=serve. SignatureDefs define the outputs to show.
saved_model_cli show --dir /tmp/higgs_boosted_trees_saved_model/${TIMESTAMP}/ \
    --tag_set serve --all
```
# Inference
让我们用这个模型来预测两个例子的输入组。 请注意，此模型使用自定义解析模块导出SavedModel，该模块接受csv行作为特征。 （每行是一个包含28列的示例;与训练数据不同，请注意不要添加标签列。）

```
aved_model_cli run --dir /tmp/boosted_trees_higgs_saved_model/${TIMESTAMP}/ \
    --tag_set serve --signature_def="predict" \
    --input_exprs='inputs=["0.869293,-0.635082,0.225690,0.327470,-0.689993,0.754202,-0.248573,-1.092064,0.0,1.374992,-0.653674,0.930349,1.107436,1.138904,-1.578198,-1.046985,0.0,0.657930,-0.010455,-0.045767,3.101961,1.353760,0.979563,0.978076,0.920005,0.721657,0.988751,0.876678", "1.595839,-0.607811,0.007075,1.818450,-0.111906,0.847550,-0.566437,1.581239,2.173076,0.755421,0.643110,1.426367,0.0,0.921661,-1.190432,-1.615589,0.0,0.651114,-0.654227,-1.274345,3.101961,0.823761,0.938191,0.971758,0.789176,0.430553,0.961357,0.957818"]'
```
这将打印出预测的类和类概率。 就像是：

```
Result for output key class_ids:
[[1]
 [0]]
Result for output key classes:
[['1']
 ['0']]
Result for output key logistic:
[[0.6440273 ]
 [0.10902369]]
Result for output key logits:
[[ 0.59288704]
 [-2.1007526 ]]
Result for output key probabilities:
[[0.3559727 0.6440273]
 [0.8909763 0.1090237]]
```
请注意，“predict”signature_def给出的结果与“classification”或“serving_default”不同（要更详细）。

# Additional Links

如果您对分布式培训感兴趣，请查看[Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed)。

您还可以在[Cloud ML Engine上训练模型](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction)，该[模型提供超参数调整以最大化模型](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#hyperparameter_tuning)的结果，并允许[data](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction#deploy_a_model_to_support_prediction)。
