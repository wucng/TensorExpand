# 一般的做法
要定义一个新模块，发布者使用函数`module_fn`调用`hub.create_module_spec（）`。 该函数构造一个表示模块内部结构的图形，使用`tf.placeholder（）`作为调用者提供的输入。 然后它通过一次或多次调用`hub.add_signature（name, inputs, outputs）`来定义签名。

For example:

```
def module_fn():
  inputs = tf.placeholder(dtype=tf.float32, shape=[None, 50])
  layer1 = tf.layers.fully_connected(inputs, 200)
  layer2 = tf.layers.fully_connected(layer1, 100)
  outputs = dict(default=layer2, hidden_activations=layer1)
  # Add default signature.
  hub.add_signature(inputs=inputs, outputs=outputs)

...
spec = hub.create_module_spec(module_fn)

```

可以使用`hub.create_module_spec（）`的结果（而不是路径）在特定的TensorFlow图中实例化模块对象。 在这种情况下，没有检查点，模块实例将使用变量初始化器。

任何模块实例都可以通过其`export(path, session)`方法序列化到磁盘。 导出模块将其定义与`session`中其变量的当前状态一起序列化到传递的路径中。 这可以在第一次导出模块时使用，也可以在导出精调模块时使用。

为了与TensorFlow Estimators兼容，`hub.LatestModuleExporter`从最新检查点导出模块，就像`tf.estimator.LatestExporter`从最新检查点导出整个模型一样。

模块发布者应该尽可能实现[通用签名](https://tensorflow.google.cn/hub/common_signatures/)，以便消费者可以轻松地交换模块并找到最适合他们问题的模块。

## Real example
查看我们的[文本嵌入模块导出器](https://github.com/tensorflow/hub/blob/r0.1/examples/text_embeddings/export.py)，了解如何使用通用文本嵌入格式创建模块的真实示例。
