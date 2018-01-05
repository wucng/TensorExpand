from sklearn import tree
from sklearn import datasets
import pickle

iris=datasets.load_iris()
X,y=iris.data,iris.target
# '''
res=tree.DecisionTreeClassifier()
# res=tree.DecisionTreeRegressor()
res.fit(X,y)

print(res.score(X,y))
print('pred:',res.predict(X[:10]),'real:',y[:10])


