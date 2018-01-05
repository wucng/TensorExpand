import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=datasets.load_iris()
iris_X=iris.data
iris_y=iris.target
# print(iris_X[:2])
# print(iris_y[:])

X_train,X_test,y_train,y_test=train_test_split(iris_X,iris_y,test_size=0.3) # 30%做测试

knn=KNeighborsClassifier() # knn model
knn.fit(X_train,y_train) # train
print('分类精度：',knn.score(X_test,y_test))
print(knn.predict(X_test)) # 预测值
print(y_test) # 真实值
