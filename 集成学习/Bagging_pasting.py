#hard voting

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
x,y=datasets.make_moons(n_samples=500,noise=0.3,random_state=666)
plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()

#voting_classifier

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)

#使用bagging

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
bagging_clf=BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=500,max_samples=100,bootstrap=True)
bagging_clf.fit(x_train,y_train)
print(bagging_clf.score(x_test,y_test))

#oob——csore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
bagging_clf=BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=500,max_samples=100,bootstrap=True,oob_score=True)
bagging_clf.fit(x,y)
print(bagging_clf.score(x_test,y_test))
print(bagging_clf.oob_score_)