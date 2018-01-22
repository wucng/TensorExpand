参考：http://blog.topspeedsnail.com/archives/10729


----------
本帖训练一个简单的神经网络模型，用来判断声音是男是女。

本帖数据集取自[voice-gender](https://github.com/primaryobjects/voice-gender)项目，这个项目使用了n种分类模型，并比较了准确率，但是它没有使用神经网络模型，本帖算是一个补充。

# 数据集
这个[数据集](http://blog.topspeedsnail.com/wp-content/uploads/2016/12/voice.csv)是经过R语言处理过的，它提取出了.WAV文件的一些声音属性。如果你想自己从wav文件中提取声音属性，参看voice-gender项目中一个叫sound.R源码文件。

数据集字段：”meanfreq”,”sd”,”median”,”Q25″,”Q75″,”IQR”,”skew”,”kurt”,”sp.ent”,”sfm”,”mode”,”centroid”,”meanfun”,”minfun”,”maxfun”,”meandom”,”mindom”,”maxdom”,”dfrange”,”modindx”,”label”。最后一个字段标记了是男声还是女声，前面字段是声音属性。

# 代码

