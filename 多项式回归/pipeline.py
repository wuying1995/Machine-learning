import numpy as np
import matplotlib.pyplot as plt

np.random.seed(666)
x=np.random.uniform(-3.0,3.0,size=100)
x=x.reshape(-1,1)
y=0.5*x**2+ x + 2 + np.random.normal(0,1,size=100).reshape(-1,1)
#plt.scatter(x,y)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

poly_= Pipeline({
    ("poly",PolynomialFeatures(degree=2)),#特征
    ("std_scaler",StandardScaler()),   #数值均一化
    ("lin",LinearRegression())
})
poly_.fit(x,y)
y_= poly_.predict(x)
plt.scatter(x,y)
plt.plot(np.sort(x),y_[np.argsort(x)],color="r")
plt.show()







