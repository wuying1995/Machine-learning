import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x=np.random.uniform(-3.0,3.0,size=100)
x=x.reshape(-1,1)
y=0.5*x**2+ x + 2 + np.random.normal(0,1,size=100).reshape(-1,1)
plt.scatter(x,y)
plt.show()
from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x,y)
y_=lin.predict(x)
plt.scatter(x,y)
plt.plot(x,y_,color="r")
plt.show

##scikit_learn 中多项式回归和prinpline
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)#最多几次幂
poly.fit(x)
x1=poly.transform(x)

#linearRegression

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(x1,y)
y_=lin.predict(x1)
#绘制

plt.scatter(x,y)
plt.plot(np.sort(x1),y_[np.argsort(x1)],color='r')
plt.show()


