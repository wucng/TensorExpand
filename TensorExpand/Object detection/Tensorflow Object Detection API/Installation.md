<font size=4>

[toc]

# Installation
## Dependencies
Tensorflow对象检测API依赖于以下库：

```
Protobuf 3.0.0
Python-tk
Pillow 1.0
lxml
tf Slim (which is included in the "tensorflow/models/research/" checkout)
Jupyter notebook
Matplotlib
Tensorflow (>=1.9.0)
Cython
contextlib2
cocoapi
```
有关安装Tensorflow的详细步骤，请按照[Tensorflow安装说明](https://www.tensorflow.org/install/)进行操作。 典型用户可以使用以下命令之一安装Tensorflow：

```
# For CPU
pip3 install tensorflow
# For GPU
pip3 install tensorflow-gpu
```
其余库可以通过`apt-get`安装在Ubuntu 16.04上：

```
sudo apt-get install protobuf-compiler python3-pil python3-lxml python3-tk
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user jupyter
pip3 install --user matplotlib
```
或者，用户可以使用pip安装依赖项：

```
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user pillow
pip3 install --user lxml
pip3 install --user jupyter
pip3 install --user matplotlib
```

注意：有时“`sudo apt-get install protobuf-compiler`”将为您安装Protobuf 3+版本，并且一些用户在使用3.5时会遇到问题。 如果是这种情况，请尝试手动安装。
# COCO API installation

如果您对使用COCO评估指标感兴趣，请下载[cocoapi](https://github.com/cocodataset/cocoapi)并将pycocotools子文件夹复制到`tensorflow/models/research`目录。 默认指标基于Pascal VOC评估中使用的指标。 要使用COCO对象检测指标，请将`metrics_set：“coco_detection_metrics”`添加到配置文件中的`eval_config`消息中。 要使用COCO实例分段度量标准，请将metrics_set：“coco_mask_metrics”添加到配置文件中的`eval_config`消息中。

```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
# or pip3 install pycocotools
cp -r pycocotools <path_to_tensorflow>/models/research/
```
# Protobuf Compilation
Tensorflow对象检测API使用Protobufs配置模型和训练参数。 在使用框架之前，必须编译Protobuf库。 这应该通过从`tensorflow/models/research/`目录运行以下命令来完成：

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```
注意：如果在编译时遇到错误，则可能使用了不兼容的protobuf编译器。 如果是这种情况，请使用以下手动安装
# Manual protobuf-compiler installation and usage
下载并安装protoc的3.0版本，然后解压缩该文件。

```
# From tensorflow/models/research/
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
```
再次运行编译过程，但使用下载的protoc版本

```
# From tensorflow/models/research/
./bin/protoc object_detection/protos/*.proto --python_out=.
```
# Add Libraries to PYTHONPATH
在本地运行时，`tensorflow / models / research /`和`slim`目录应该附加到`PYTHONPATH`。 这可以通过从`tensorflow / models / research /`运行以下命令来完成：

```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
注意：此命令需要从您启动的每个新终端运行。 如果您希望避免手动运行，可以将其作为新行添加到`〜/ .bashrc`文件的末尾，将`“pwd”`替换为系统上`tensorflow / models / research`的绝对路径。

# Testing the Installation
您可以测试是否已正确安装Tensorflow对象检测
通过运行以下命令API：

```
python3 object_detection/builders/model_builder_test.py
```
