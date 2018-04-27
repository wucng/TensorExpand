# Using a Module 

## 实例化模块
将TensorFlow Hub模块导入到TensorFlow程序中，方法是使用`URL或文件系统路径`从字符串中创建Module对象，例如：

```
m = hub.Module("path/to/a/module_dir")
```

这将模块的变量添加到当前的TensorFlow图中。 运行其初始化程序将从磁盘读取预先训练的值。 同样，表和其他状态被添加到图中。

## 缓存模块
从URL创建模块时，模块内容将被下载并缓存在本地系统临时目录中。 使用`TFHUB_CACHE_DIR`环境变量可以覆盖模块缓存的位置。

例如，将`TFHUB_CACHE_DIR`设置为`/my_module_cache`：

```
$ export TFHUB_CACHE_DIR=/my_module_cache
```
然后从URL创建一个模块：

```
m = hub.Module("https://tfhub.dev/google/progan-128/1")
```
导致将模块下载并解压缩到`/my_module_cache`中。

## Applying a Module
一旦实例化，模块m可以被调用零或更多次，如从张量输入到张量输出的Python函数：

```
y = m(x)
```
每次这样的调用都会将操作添加到当前的TensorFlow图表中以从x计算y。 如果这涉及具有训练权重的变量，则这些变量在所有应用程序之间共享

模块可以定义多个命名的签名，以允许以多种方式应用（类似于Python对象具有方法的方式）。 模块的文档应该描述可用的签名。 上面的调用应用名为“`default`”的签名。 可以通过将其名称传递给可选的`signature =`参数来选择任何签名。

如果签名具有多个输入，则必须将其作为字典传递，并使用由签名定义的键。 同样，如果签名具有多个输出，则可以通过在签名定义的密钥下传递`as_dict = True`来将这些输出作为字典进行检索。 （如果`as_dict = False`，则关键“`default`”用于返回单个输出。）因此，应用模块的最一般形式如下所示：

```python
outputs = m(dict(apples=x1, oranges=x2), signature="fruit_to_pet", as_dict=True)
y1 = outputs["cats"]
y2 = outputs["dogs"]
```

调用者必须提供由签名定义的所有输入，但不要求使用模块的所有输出。 TensorFlow将只运行模块中那些最终作为`tf.Session.run（）`中目标的依赖关系的部分。 实际上，模块出版商可以选择提供各种输出用于高级用途（如中间层的激活）以及主输出。 模块消费者应该优雅地处理额外的输出。
