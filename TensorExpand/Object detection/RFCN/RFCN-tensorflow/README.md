TensorFlow implementation of RFCN
=================================

Paper is available on https://arxiv.org/abs/1605.06409.

Building
--------

The ROI pooling and the MS COCO loader needs to be compiled first. To do so, run make in the root directory of the project. You may need to edit *BoxEngine/ROIPooling/Makefile* if you need special linker/compiler options.

*NOTE:* If you have multiple python versions on your system, and you want to use a different one than "python", provide an environment variable called PYTHON before calling make. For example: PYTHON=python3 make

You may get undefined symbol problems while trying to load the .so file. This will be the case if you built your TensorFlow version yourself and the Makefile fails to auto-detect your ABI version. You may encounter errors like "tensorflow.python.framework.errors_impl.NotFoundError: BoxEngine/ROIPooling/roi_pooling.so: undefined symbol: \_ZN10tensorflow7strings6StrCatB5cxx11ERKNS0_8AlphaNumE" in the log. In this case clean the project (make clean) and rebuild it with USE_OLD_EABI=0 flag (USE_OLD_EABI=0 make).

You may want to build ROI pooling without GPU support. Use the USE_GPU=0 flag to turn off the CUDA part of the code.

You may want to install python dependencies by running:

pip install --user -r packages.txt

Testing
-------

You can run trained models with test.py. Model path should be given without file extension (without .data* and .index). An example:

![preview](https://cloud.githubusercontent.com/assets/2706617/25061919/2003e832-21c1-11e7-9397-14224d39dbe9.jpg)

Pretrained model
----------------

You can download a pretrained model from here:

http://xdever.engineerjs.com/rfcn-tensorflow-export.tar.bz2

Extract it to your project directory. Then you can run the network with the following command:

./test.py -n export/model -i \<input image\> -o \<output image\>

*NOTE:* this pretrained model was not hyperparameter-optimized in any way. The model can (and will) have much better performance when optimized. Try out different learning rates and classification to regression loss balances. Optimal values are highly test dependent.

Training the network
--------------------

For training the network you will first need to download the MS COCO dataset. Download the needed files and extract them to a directory with the following structure:
```
<COCO>
├─  annotations
│    ├─  instances_train2014.json
│    └─  ...
|
├─  train2014
└─  ...

```
Run the following command:
./main.py -dataset \<COCO\> -name \<savedir\>
* \<COCO\> - full path to the coco root directory
* \<savedir\> - path where files will be saved. This directory and its subdirectories will be automatically created.

The \<savedir\> will have the following structure:
```
<savedir>
├─  preview
│    └─  preview.jpg - preview snapshots from training process.
|
├─  save - TensorFlow checkpoint directory
│    ├─  checkpoint
│    ├─  model_*.*
│    └─  ...
└─  args.json - saved command line arguments.

```

You can always kill the training process and resume it later just by running
./main.py -name \<savedir\>
without any other parameters. All command line parameters will be saved and reloaded automatically.

License
-------

The software is under Apache 2.0 license. See http://www.apache.org/licenses/LICENSE-2.0 for further details.

Notes
-----

This code requires TensorFlow >=1.0 (last known working version is 1.4.1). Tested with python3.6, build it *should* work with python 2.
