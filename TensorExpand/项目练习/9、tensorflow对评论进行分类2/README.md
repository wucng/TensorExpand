参考：http://blog.topspeedsnail.com/archives/10420


----------
机器学习模型的基本开发流程图

![这里写图片描述](http://blog.topspeedsnail.com/wp-content/uploads/2016/11/%E5%B1%8F%E5%B9%95%E5%BF%AB%E7%85%A7-2016-11-15-%E4%B8%8B%E5%8D%885.04.33.png)

# 使用的数据集
使用的数据集：http://help.sentiment140.com/for-students/ (情绪分析)

下载链接[Stanford link](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)

数据集包含1百60万条推特，包含消极、中性和积极tweet。不知道有没有现成的微博数据集。

数据格式：移除表情符号的CSV文件，字段如下：

- 0 – the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- 1 – the id of the tweet (2087)
- 2 – the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- 3 – the query (lyx). If there is no query, then this value is NO_QUERY.
- 4 – the user that tweeted (robotickilldozr)
- 5 – the text of the tweet (Lyx is cool)

training.1600000.processed.noemoticon.csv（238M）
testdata.manual.2009.06.14.csv（74K）

# 数据预处理


如果数据文件太大，不能一次加载到内存，可以把数据导入数据库
[Dask](https://github.com/dask/dask)可处理大csv文件

1维卷积使用[参考这里](https://github.com/fengzhongyouxia/TensorExpand/blob/master/TensorExpand/%E9%A1%B9%E7%9B%AE%E7%BB%83%E4%B9%A0/8%E3%80%81tensorflow%E5%AF%B9%E8%AF%84%E8%AE%BA%E8%BF%9B%E8%A1%8C%E5%88%86%E7%B1%BB/main.py)

# 数据处理流程图
![这里写图片描述](http://img.blog.csdn.net/20180117170756883?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2M3ODE3MDgyNDk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

