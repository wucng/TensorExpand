[PennTreebank上的LSTM语言模型](https://github.com/tensorpack/tensorpack/tree/master/examples/PennTreebank)

这个例子主要是为了证明：

- 如何在迭代之间训练具有持久状态的RNN。 在这里它只是管理图中的状态。 `state_saving_rnn`可用于更复杂的用例。
- 如何使用TF读取器管道而不是DataFlow，用于训练和推理。

它训练PTB数据集上的语言模型，基本上相当于[tensorlfow/models](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)中的PTB示例及其“中”配置。 它具有与原始示例相同的性能和速度。 请注意，数据管道完全从tensorflow示例中复制。

# Train

```
./PTB-LSTM.py
```