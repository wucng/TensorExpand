from sklearn import svm
from sklearn import datasets
import pickle

boston=datasets.load_boston()
X,y=boston.data,boston.target
# '''
res=svm.SVR()
res.fit(X,y)

print(res.score(X,y))
print('pred:',res.predict(X[:10]),'real:',y[:10])

