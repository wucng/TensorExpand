将导入模块的变量与其周围模型的变量一起训练称为微调。 微调可以提高质量，但会增加新的并发症。 我们建议消费者只有在探索简单的质量调整之后才能进行微调。

## For Consumers
要启用微调，请使用`hub.Module（...，trainable = True）`实例化模块，使其变量可训练并导入TensorFlow的`REGULARIZATION_LOSSES`。 如果模块有多个图形变体，请确保选择一个适合培训。 通常，这是标签`{“train”}`。

选择不会破坏预先训练的权重的训练制度，例如，低于从头开始训练的低学习率。

## For Publishers
为了让消费者更容易调整，请注意以下事项：

 - 微调需要正规化。你的模块是通过`REGULARIZATION_LOSSES`集合导出的，这个集合将`tf.layers.dense（...，kernel_regularizer = ...）`等的选择放入消费者从`tf.losses.get_regularization_losses（）`获得的东西中。喜欢这种定义L1 / L2正则化损失的方式。
 - 在发布者模型中，避免通过`tf.train.FtrlOptimizer`，`tf.train.ProximalGradientDescentOptimizer`和其他近端优化器的l1_和l2_regularization_strength参数定义L1 / L2正则化。这些不是与模块一起出口的，并且全球的正规化优势可能不适合消费者。除了广泛（即稀疏线性）或宽和深模型中的L1正则化之外，应该可以使用单独的正则化损失。
 - 如果您使用dropout，batch normalization或类似的训练技术，则将丢失率和其他超参数设置为在许多预期用途中有意义的值。
