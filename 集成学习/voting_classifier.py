#hard voting

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
x,y=datasets.make_moons(n_samples=500,noise=0.3,random_state=42)
plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()

#voting_classifier

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)

#lr
from sklearn.linear_model  import LogisticRegression
log_clf=LogisticRegression()
log_clf.fit(x_train,y_train)
print(log_clf.score(x_test,y_test))
#svc
from sklearn.svm import SVC
svm_clf=SVC()
svm_clf.fit(x_train,y_train)
print(svm_clf.score(x_test,y_test))
#
from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier()
dt_clf.fit(x_train,y_train)
print(dt_clf.score(x_test,y_test))

#voting
from sklearn.ensemble import VotingClassifier
voting_clf=VotingClassifier(estimators=[
    ("log_clf",LogisticRegression()),
    ("svm_clf",SVC()),
    ("DT_CLF",DecisionTreeClassifier())
],voting="hard")
voting_clf.fit(x_train,y_train)
print(voting_clf.score(x_test,y_test))