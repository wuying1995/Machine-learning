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

#使用adaBooting

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
ADA_clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=500)

ADA_clf.fit(x_train,y_train)
print(ADA_clf.score(x_test,y_test))

#GRADIENT BOOTING
from sklearn.ensemble import GradientBoostingClassifier
GD_=GradientBoostingClassifier(max_depth=2,n_estimators=30)
GD_.fit(x_train,y_train)
print(GD_.score(x_test,y_test))