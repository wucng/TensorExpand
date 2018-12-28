<font size=4>

[toc]

# [Installing Detectron](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)

本文档介绍了如何安装Detectron，其依赖项（包括Caffe2）和COCO数据集。

# Requirements:
- NVIDIA GPU, Linux, Python2
- Caffe2，各种标准Python包和COCO API; 有关安装这些依赖项的说明，请参见下文

Notes:

- Detectron运营商目前没有CPU实施; 需要GPU系统。
- Detectron已经过CUDA 8.0和cuDNN 6.0.21的广泛测试。

# Caffe2
要安装具有CUDA支持的Caffe2，请按照[Caffe2](https://caffe2.ai/)网站上的[安装说明](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=prebuilt)进行操作。 如果您已安装Caffe2，请确保将Caffe2更新为包含[Detectron模块](https://github.com/caffe2/caffe2/tree/master/modules/detectron)的版本。

请确保您的Caffe2安装成功，然后再运行以下命令并按照注释中的指示检查其输出。

```python
# To check if Caffe2 build was successful
python2 -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

# To check if Caffe2 GPU build was successful
# This must print a number > 0 in order to use Detectron
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```
如果找不到caffe2 Python包，您可能需要调整`PYTHONPATH`环境变量以包含其位置（`/path/to/caffe2/build`，其中build是Caffe2 CMake构建目录）。

# Other Dependencies
Install the [COCO API](https://github.com/cocodataset/cocoapi):

```python
# or pip install pycocotools
	
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```
请注意，`# COCOAPI=/path/to/install/cocoapi`等指令表明您应该选择要克隆软件的路径，然后相应地设置环境变量（在本例中为`COCOAPI`）。
# Detectron
克隆Detectron存储库：

```
# DETECTRON=/path/to/clone/detectron
git clone https://github.com/facebookresearch/detectron $DETECTRON
```
安装Python依赖项：

```
pip install -r $DETECTRON/requirements.txt
```
设置Python模块：

```
cd $DETECTRON && make
```
检查Detectron测试是否通过（例如，对于[SpatialNarrowAsOp test](https://github.com/facebookresearch/Detectron/blob/master/detectron/tests/test_spatial_narrow_as_op.py)）：

```
python2 $DETECTRON/detectron/tests/test_spatial_narrow_as_op.py
```
# That's All You Need for Inference
此时，您可以使用预训练的Detectron模型进行推理。 看一下我们的[推理教程](https://github.com/facebookresearch/Detectron/blob/master/GETTING_STARTED.md)中的一个例子。 如果要在COCO数据集上训练模型，请继续安装说明。

# Datasets
Detectron通过`detectron/datasets/data`的符号链接找到数据集，并将数据集存储到数据集图像和注释的实际位置。 有关如何为COCO和其他数据集创建符号链接的说明，请参阅[detectron / datasets / data / README.md](https://github.com/facebookresearch/Detectron/blob/master/detectron/datasets/data/README.md)。

创建符号链接后，这就是开始训练模型所需的全部内容。

# Advanced Topic: Custom Operators for New Research Projects
请先阅读常见问题[解答](https://github.com/facebookresearch/Detectron/blob/master/FAQ.md)的自定义运算符部分。

为方便起见，我们为构建自定义运营商提供CMake支持。 所有自定义运算符都内置在一个可以从Python动态加载的库中。 将您的自定义运算符实现放在[detectron/ops/](https://github.com/facebookresearch/Detectron/tree/master/detectron/ops)下，并参阅[detectron / tests / test_zero_even_op.py](https://github.com/facebookresearch/Detectron/blob/master/detectron/tests/test_zero_even_op.py)，以获取如何从Python加载自定义运算符的示例。

构建自定义运算符库：

```
cd $DETECTRON && make ops
```
检查自定义运算符测试是否通过：

```
python2 $DETECTRON/detectron/tests/test_zero_even_op.py
```
# Docker Image
我们提供了一个[Dockerfile](https://github.com/facebookresearch/Detectron/blob/master/docker/Dockerfile)，您可以使用它在`Caffe2 image`上构建一个`Detectron image`，满足顶部概述的要求。 如果您想使用与我们默认使用的`Caffe2 image`不同的`Caffe2 image`，请确保它包含[Detectron模块](https://github.com/caffe2/caffe2/tree/master/modules/detectron)。

构建镜像：

```
cd $DETECTRON/docker
docker build -t detectron:c2-cuda9-cudnn7 .
```
运行镜像：

```
nvidia-docker run --rm -it detectron:c2-cuda9-cudnn7 python2 detectron/tests/test_batch_permutation_op.py
```
# Troubleshooting
如果出现Caffe2安装问题，请首先阅读相关Caffe2安装[说明的故障排除](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=prebuilt)部分。 在下文中，我们为Caffe2和Detectron提供了额外的故障排除提示。

# Caffe2 Operator Profiling
Caffe2提供了性能[分析](https://github.com/caffe2/caffe2/tree/master/caffe2/contrib/prof)支持，您可能会发现它对基准测试或调试操作员很有用（有关示例用法，请参阅[BatchPermutationOp测试](https://github.com/facebookresearch/Detectron/blob/master/detectron/tests/test_batch_permutation_op.py)）。 默认情况下不会构建性能分析支持，您可以通过在运行Caffe2 CMake时设置`-DUSE_PROF = ON`标志来启用它。

# CMake Cannot Find CUDA and cuDNN
有时CMake在您的机器上找到CUDA和cuDNN目录时遇到问题。

在构建Caffe2时，您可以通过运行以下命令将CMake指向CUDA和cuDNN目录：

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/dir \
  -DCUDNN_ROOT_DIR=/path/to/cudnn/root/dir
```
同样，在构建自定义Detectron运算符时，您可以使用：

```
cd $DETECTRON
mkdir -p build && cd build
cmake .. \
  -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit/dir \
  -DCUDNN_ROOT_DIR=/path/to/cudnn/root/dir
make
```
请注意，您可以使用相同的命令让`CMake`使用特定版本的CUDA和cuDNN，这可能是您计算机上安装的多个版本。
# Protobuf Errors
Caffe2使用protobuf作为其序列化格式，需要3.2.0或更高版本。 如果您的protobuf版本较旧，您可以从Caffe2 protobuf子模块构建protobuf并使用该版本。

构建Caffe2 protobuf子模块：

```
# CAFFE2=/path/to/caffe2
cd $CAFFE2/third_party/protobuf/cmake
mkdir -p build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$HOME/c2_tp_protobuf \
  -Dprotobuf_BUILD_TESTS=OFF \
  -DCMAKE_CXX_FLAGS="-fPIC"
make install
```
将Caffe2 CMake指向新建的protobuf：

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DPROTOBUF_PROTOC_EXECUTABLE=$HOME/c2_tp_protobuf/bin/protoc \
  -DPROTOBUF_INCLUDE_DIR=$HOME/c2_tp_protobuf/include \
  -DPROTOBUF_LIBRARY=$HOME/c2_tp_protobuf/lib64/libprotobuf.a
```
如果您同时安装了`system`和`anaconda`软件包，也可能会遇到protobuf问题。 这可能会导致问题，因为版本可能在编译时或运行时混合。 遵循上面的命令也可以克服此问题。
# Caffe2 Python Binaries
如果您在构建Caffe2 Python二进制文件时遇到CMake无法找到所需Python路径的问题（例如在virtualenv中），您可以尝试将Caffe2 CMake指向python库并使用以下命令包含dir：

```
cmake .. \
  # insert your Caffe2 CMake flags here
  -DPYTHON_LIBRARY=$(python2 -c "from distutils import sysconfig; print(sysconfig.get_python_lib())") \
  -DPYTHON_INCLUDE_DIR=$(python2 -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")
```
# Caffe2 with NNPACK Build
Detectron不需要使用NNPACK支持构建的Caffe2。 如果在安装Caffe2期间遇到NNPACK相关问题，可以通过设置`-DUSE_NNPACK = OFF` CMake标志来安全地禁用NNPACK。

# Caffe2 with OpenCV Build
类似于上面的NNPACK情况，您可以通过设置`-DUSE_OPENCV = OFF` CMake标志来禁用OpenCV。
# COCO API Undefined Symbol Error
如果由于未定义的符号而遇到COCO API导入错误（如[此处](https://github.com/cocodataset/cocoapi/issues/35)所述），请确保您的python版本没有混合。 例如，如果您同时[安装了system和conda numpy](https://stackoverflow.com/questions/36190757/numpy-undefined-symbol-pyfpe-jbuf)，则可能会出现此问题。
# CMake Cannot Find Caffe2
如果您在构建自定义运算符时遇到CMake无法找到Caffe2包的问题，请确保在Caffe2安装过程中运行`make install`。