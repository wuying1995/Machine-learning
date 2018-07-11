import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
boston =datasets.load_boston()
x=boston.data[:,5]
y=boston.target
x=x[y<50.0]
y=y[y<50.0]
plt.scatter(x,y)
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)
from SimpleLineareRegression import simpleLinearRegression
clf = simpleLinearRegression()
clf.fit(x_train,y_train)
print(clf.a_)
print(clf.b_)
plt.scatter(x_train,y_train)
plt.plot(x_train,clf.predict(x_train),color="r")
plt.show()
y_predict=clf.predict(x_test)
#Mse /mae
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
a=mean_squared_error(y_test,y_predict)
print(a)
b=mean_absolute_error(y_test,y_predict)
print(b)
from sklearn.metrics import r2_score
c=r2_score(y_test,y_predict)
print(c)
