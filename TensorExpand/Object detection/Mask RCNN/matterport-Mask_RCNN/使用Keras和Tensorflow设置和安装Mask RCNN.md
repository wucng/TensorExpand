参考：

- Github slide: https://github.com/markjay4k/Mask-RCN...
- Mask RCNN Repo: https://github.com/matterport/Mask_RCNN
- requirements.txt: https://github.com/markjay4k/Mask-RCN...
- Mask RCNN paper: https://arxiv.org/pdf/1703.06870.pdf
- [video](https://www.youtube.com/watch?v=2TikTv6PWDw)


----------
# 概述如何安装
Step 1: create a conda virtual environment with python 3.6
Step 2: install the dependencies
Step 3: Clone the Mask_RCNN repo
Step 4: install pycocotools
Step 5: download the pre-trained weights
Step 6: Test it

# Step 1 - Create a conda virtual environment
we will be using Anaconda with python 3.6.
If you don't have Anaconda, follow this tutorial
https://www.youtube.com/watch?v=T8wK5loXkXg

- run this command in a CMD window

```
conda create -n MaskRCNN python=3.6 pip
```

# Step 2 - Install the Dependencies
- place the requirements.txt in your cwdir
https://github.com/markjay4k/Mask-RCNN-series/blob/master/requirements.txt
- run these commands

```
actvitate MaskRCNN
pip install -r requirements.txt
```

- NOTE: we're installing these (tf-gpu requires some pre-reqs)
numpy, scipy, cython, h5py, Pillow, scikit-image, 
tensorflow-gpu==1.5, keras, jupyter

# Step 3 - Clone the Mask RCNN Repo
- Run this command

```
git clone https://github.com/matterport/Mask_RCNN.git
```

# Step 4 - Install pycocotools
- NOTE: pycocotools requires Visual C++ 2015 Build Tools
- download here if needed http://landinghub.visualstudio.com/visual-cpp-build-tools

- clone this repo

```
git clone https://github.com/philferriere/cocoapi.git
```

- use pip to install pycocotools

```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

# Step 5 - Download the Pre-trained Weights
- Go here https://github.com/matterport/Mask_RCNN/releases
- download the [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) file
- place the file in the Mask_RCNN directory


# Step 6 - Let's Test it!
open up the [demo.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/demo.ipynb) and run it