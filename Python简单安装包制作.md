参考：

- https://my.oschina.net/leopardsaga/blog/174021
- http://blog.csdn.net/bin381/article/details/52822849

---

1、项目目录下创建 `setup.py` 文件， 目录与内容如下举例


```python
~/vomm$ tree
.
├── LICENSE
├── MANIFEST
├── MANIFEST.in
├── README.md
├── setup.py
├── vomm
│   ├── classes.py
│   ├── __init__.py
│   └── tests
│       ├── __init__.py
│       └── test_vomm.py


vim setup.py 
"""
两种方式引入setup. 
一种从setuptools包，一种从distutils.core包，前者可以方便上传至PyPI发布.

从setuptools包引入setup，要同时引入find_packages包用来搜索项目内的各packages
"""
from setuptools import setup, find_packages

setup(
    name = 'vomm',
    version = 0.1,
    packages = find_packages(),
    author = 'Honghe Wu',
    author_email = 'leopardsaga@gmail.com',
    url = '',
    license = 'http://www.apache.org/licenses/LICENSE-2.0.html',
    description = 'Variable Order Markov Model'
    ) 
```


2、添加 `MANIFEST.in`, 内容至少包含README说明文件

```
 $cat MANIFEST.in 
  include README.md
```

3、setup.py编译命令

```python
python setup.py bdist_egg # 生成类似 bee-0.0.1-py2.7.egg，支持 easy_install 
python setup.py sdist     # 生成类似 bee-0.0.1.tar.gz，支持 pip 
python setup.py build     #编译
python setup.py bdist_wininst # Windows exe
python setup.py bdist_rpm     # rpm
```

4、Python gz压缩包制作

前2步同上
最后一步为 `python setup.py sdist`, 生成 tar.gz 文件
tar.gz 在Linux与Windows都可方便pip安装 `pip install <package>.tar.gz`，也方便发布上PyPI
上传到 PyPI

暂时不弄，参考 怎样制作一个 [Python Egg](http://liluo.org/blog/2012/08/how-to-create-python-egg/)

