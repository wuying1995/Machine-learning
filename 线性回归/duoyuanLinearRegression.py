import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
boston=datasets.load_boston()
x=boston.data
y=boston.target
x=x[y<50.0]
y=y[y<50.0]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=66)
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x_train,y_train)
print(lin.coef_)
print(lin.intercept_)
print(lin.score(x_test,y_test))