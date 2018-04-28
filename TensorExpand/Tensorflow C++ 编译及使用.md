参考：

- [Tensorflow C++ 编译和调用图模型](https://blog.csdn.net/rockingdingo/article/details/75452711)
- [Tensorflow C++库的编译和使用方法](https://blog.csdn.net/xinchen1234/article/details/78750079)

------
# 配置环境
要依赖安装：`tensorflow`, `bazel`, `protobuf` , `eigen`(一种矩阵运算的库)

更新系统环境 `apt-get update && apt-get upgrade`


## 1、[bazel安装](https://docs.bazel.build/versions/master/install.html)
详细安装参考[这里](https://docs.bazel.build/versions/master/install.html)

- Step 1: Install required packages
```python
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python
```
- Step 2: Download Bazel

在[这里](https://github.com/bazelbuild/bazel/releases)下载`bazel-<version>-installer-linux-x86_64.sh`

- Step 3: Run the installer

```
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-<version>-installer-linux-x86_64.sh --user
```
`--user`标志将Bazel安装到系统上的`$HOME/bin`目录，并将`.bazelrc`路径设置为`$HOME/.bazelrc`。 使用`--help`命令查看其他安装选项。

- Step 4: Set up your environment

如果您使用上面的`--user`标志运行Bazel安装程序，则Bazel可执行文件将安装在您的`$HOME/bin`目录中。 将此目录添加到默认路径是一个好主意，如下所示：

```
export PATH="$PATH:$HOME/bin"
```
您也可以将此命令添加到`~/.bashrc`文件中。

注意：Bazel包含一个嵌入式JDK，即使已经安装了JDK，也可以使用它。 `bazel- <version> -without-jdk-installer-linux-x86_64.sh`是没有嵌入式JDK的安装程序版本。 如果您已经安装了JDK 8，请仅使用此安装程序。 后来的JDK版本不受支持。

## 2、安装Protobuf
参考：[linux 安装protobuf,以及python版](https://blog.csdn.net/lmb1612977696/article/details/78611918)

```
wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz

tar -xf  protobuf-all-3.5.1.tar.gz  
cd protobuf-3.5.1  
./configure   
make   
make check   
make install  
```
## 3、安装 Eigen, 用于矩阵运算 
 参考：[ubuntu16.04+eigen3安装](https://www.cnblogs.com/newneul/p/8256803.html)

[eigen官网](http://eigen.tuxfamily.org/index.php?title=Main_Page)
```python
# 默认安装到/usr/local/include里(可在终端中输入locate eigen3查看位置)
sudo apt-get install libeigen3-dev
```
## 4、升级binutils到22以上版本 

```
    wget http://ftp.gnu.org/gnu/binutils/binutils-2.25.1.tar.bz2;
    tar -xjf binutils-2.25.1.tar.bz2;
    cd binutils-2.25.1;
    make; make install
```

## 5、下载tensorflow源码编译

```python
# 从github下载tensorflow源代码  
git clone --recursive https://github.com/tensorflow/tensorflow  
  
## 进入根目录后编译
cd tensorflow
# 配置
./configure

# 用bazel进行build 
# 编译生成.so文件, 编译C++ API的库 (建议)  
bazel build //tensorflow:libtensorflow_cc.so  
# 或
bazel build -c opt –verbose_failures //tensorflow:libtensorflow_cc.so  
# –verbose_failures是为了当编译有错的时候，可以看到出错时的编译命令，方便查找原因或向google反馈

# 也可以选择,编译C API的库  
bazel build //tensorflow:libtensorflow.so 
# 或
bazel build -c opt –verbose_failures //tensorflow:libtensorflow.so

# 用makefile进行编译 
cd tensorflow/contrib/makefile
./build_all_linux.sh，
# 其他平台比如ios，android版本的编译脚本也在这个目录下，运行对应脚本即可
```
在等待30多分钟后, 如果编译成功，在tensorflow根目录下出现 bazel-bin, bazel-genfiles 等文件夹, 按顺序执行以下命令将对应的libtensorflow_cc.so文件和其他文件拷贝进入 /usr/local/lib/ 目录

```python
mkdir /usr/local/include/tf  
cp -r bazel-genfiles/ /usr/local/include/tf/  
cp -r tensorflow /usr/local/include/tf/  
cp -r third_party /usr/local/include/tf/  
cp -r bazel-bin/tensorflow/libtensorflow_cc.so /usr/local/lib/ 
```
这一步完成后，我们就准备好了libtensorflow_cc.so文件等，后面在自己的C++编译环境和代码目录下编译时链接这些库即可。

# Python线下定义模型和训练
- 1.1 定义Graph中输入和输出tensor名称

为了方便我们在调用C++ API时，能够准确根据Tensor的名称取出对应的结果，在python脚本训练时就要先定义好每个tensor的tensor_name。 如果tensor包含命名空间namespace的如"namespace_A/tensor_A" 需要用完整的名称。(Tips: 对于不清楚tensorname具体是什么的，可以在输出的 .pbtxt文件中找对应的定义)； 这个例子中，我们定义以下3个tensor的tensorname

```python
    class TensorNameConfig(object):  
        input_tensor = "inputs"  
        target_tensor = "target"  
        output_tensor = "output_node"  
        # To Do  
```

- 1.2 输出graph的定义文件*.pb和参数文件 *.ckpt

我们要在训练的脚本nn_model.py中加入两处代码：第一处是将tensorflow的graph_def保存成./models/目录下一个文件nn_model.pbtxt, 里面包含有图中各个tensor的定义名称等信息。 第二处是在训练代码中加入保存参数文件的代码，将训练好的ANN模型的权重Weight和Bias同时保存到./ckpt目录下的*.ckpt, *.meta等文件。最后执行 python nn_model.py 就可以完成训练过程

```python
    # 保存图模型  
    tf.train.write_graph(session.graph_def, FLAGS.model_dir, "nn_model.pbtxt", as_text=True)  
      
    # 保存 Checkpoint  
    checkpoint_path = os.path.join(FLAGS.train_dir, "nn_model.ckpt")  
    model.saver.save(session, checkpoint_path)  
      
    # 执行命令完成训练过程  
    python nn_model.py  
```

- 1.3 使用freeze_graph.py小工具整合模型freeze_graph
最后利用tensorflow自带的 freeze_graph.py小工具把.ckpt文件中的参数固定在graph内，输出nn_model_frozen.pb

```python
    # 运行freeze_graph.py 小工具  
    # freeze the graph and the weights  
    python freeze_graph.py --input_graph=../model/nn_model.pbtxt --input_checkpoint=../ckpt/nn_model.ckpt --output_graph=../model/nn_model_frozen.pb --output_node_names=output_node  
      
    # 或者执行  
    sh build.sh  
      
    # 成功标志:   
    # Converted 2 variables to const ops.  
    # 9 ops in the final graph.  
```
 脚本中的参数解释：

 
```
 --input_graph: 模型的图的定义文件nn_model.pb （不包含权重）；
--input_checkpoint: 模型的参数文件nn_model.ckpt；
--output_graph: 绑定后包含参数的图模型文件 nn_model_frozen.pb；
-- output_node_names: 输出待计算的tensor名字【重要】；
```

发现tensorflow不同版本下运行freeze_graph.py 脚本时可能遇到的Bug挺多的，列举一下：

[略](https://blog.csdn.net/rockingdingo/article/details/75452711)

最后如果输出如下: Converted variables to const ops. * ops in the final graph 就代表绑定成功了！发现绑定了参数的的.pb文件大小有10多MB。


#  C++API调用模型和编译
在C++预测阶段，我们在工程目录下引用两个tensorflow的头文件:

- 2.1 C++加载模型

```c
    #include "tensorflow/core/public/session.h"  
    #include "tensorflow/core/platform/env.h"  
```
在这个例子中我们把C++的API方法都封装在基类里面了。 FeatureAdapterBase 用来处理输入的特征，以及ModelLoaderBase提供统一的模型接口load()和predict()方法。然后可以根据自己的模型可以继承基类实现这两个方法，如本demo中的ann_model_loader.cpp。可以参考下，就不具体介绍了。

新建Session, 从model_path 加载*.pb模型文件，并在Session中创建图。预测的核心代码如下：

```cpp
    // @brief: 从model_path 加载模型，在Session中创建图  
    // ReadBinaryProto() 函数将model_path的protobuf文件读入一个tensorflow::GraphDef的对象  
    // session->Create(graphdef) 函数在一个Session下创建了对应的图;  
      
    int ANNModelLoader::load(tensorflow::Session* session, const std::string model_path) {  
        //Read the pb file into the grapgdef member  
        tensorflow::Status status_load = ReadBinaryProto(Env::Default(), model_path, &graphdef);  
        if (!status_load.ok()) {  
            std::cout << "ERROR: Loading model failed..." << model_path << std::endl;  
            std::cout << status_load.ToString() << "\n";  
            return -1;  
        }  
      
        // Add the graph to the session  
        tensorflow::Status status_create = session->Create(graphdef);  
        if (!status_create.ok()) {  
            std::cout << "ERROR: Creating graph in session failed..." << status_create.ToString() << std::endl;  
            return -1;  
        }  
        return 0;  
    }  
```

预测阶段的函数调用 session->Run(input_feature.input, {output_node}, {}, &outputs);

 参数 const FeatureAdapterBase& input_feature, 内部的成员input_feature.input是一个Map型, std::vector<std::pair >, 类似于python里的feed_dict={"x":x, "y": y}，这里的C++代码中的输入tensor_name也一定要和python训练脚本中的一致, 如上文中设定的"inputs", "targets" 等。调用基类 FeatureAdapterBase中的方法assign(std::string, std::string tname, std::vector* vec) 函数来定义。

参数 const std::string output_node, 对应的就是在python脚本中定义的输出节点的名称，如"name_scope/output_node"

```c
    int ANNModelLoader::predict(tensorflow::Session* session, const FeatureAdapterBase& input_feature,  
            const std::string output_node, double* prediction) {  
        // The session will initialize the outputs  
        std::vector<tensorflow::Tensor> outputs;         //shape  [batch_size]  
      
        // @input: vector<pair<string, tensor> >, feed_dict  
        // @output_node: std::string, name of the output node op, defined in the protobuf file  
        tensorflow::Status status = session->Run(input_feature.input, {output_node}, {}, &outputs);  
        if (!status.ok()) {  
            std::cout << "ERROR: prediction failed..." << status.ToString() << std::endl;  
            return -1;  
        }  
      
        // ...  
    }  
```

- 2.2 C++编译的方法

记得我们之前预先编译好的libtensorflow_cc.so文件，要成功编译需要链接那个库。 运行下列命令：

```cpp
    # 使用g++  
    g++ -std=c++11 -o tfcpp_demo \  
    -I/usr/local/include/tf \  
    -I/usr/local/include/eigen3 \  
    -g -Wall -D_DEBUG -Wshadow -Wno-sign-compare -w  \  
    `pkg-config --cflags --libs protobuf` \  
    -L/usr/local/lib/libtensorflow_cc \  
    -ltensorflow_cc main.cpp ann_model_loader.cpp  
```

 参数含义:

  

```
  a) -I/usr/local/include/tf # 依赖的include文件
  b) -L/usr/local/lib/libtensorflow_cc # 编译好的libtensorflow_cc.so文件所在的目录
  c) -ltensorflow_cc # .so文件的文件名
```

为了方便调用，尝试着写了一个`Makefile`文件，将里面的路径换成自己的，每次直接用make命令执行就好

```
make 
```

此外，在直接用g++来编译的过程中可能会遇到一些Bug, 现在记录下来


