import numpy as np
import matplotlib.pyplot as plt


np.random.seed(666)
x=np.random.normal(0,1,size=(200,2))
y=np.array(x[:,0]**2+x[:,1]<1.5,dtype="int")

#随机选20个点 强制变为结果为1

for _ in range(20):
    y[np.random.randint(200)] = 1
plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()
# np.random.seed(666)
# x=np.random.uniform(-3.0,3.0,size=100)
# x=x.reshape(-1,1)
# y=0.5*x**2+ x + 2 + np.random.normal(0,1,size=100).reshape(-1,1)
# plt.scatter(x,y)
#sklearn lr

#1,train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)

#2.sklearn lr
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x_train,y_train)
print(LR.score(x_train,y_train))

#多项式逻辑回归
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
def PolyLogisticRegression(degree):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('LR',LinearRegression())
    ])
poly=PolyLogisticRegression(degree=2)
poly.fit(x_train,y_train)
print(poly.score(x_train,y_train))
poly2=PolyLogisticRegression(degree=20)
poly2.fit(x_train,y_train)
print(poly2.score(x_train,y_train))
print(poly2.score(x_test,y_test))
#cc
def PolyLogisticRegression(degree,c):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('LR',LinearRegression(c=c))
    ])
poly3=PolyLogisticRegression(degree=20,c=0.1)
poly3.fit(x_train,y_train)
print(poly3.score(x_train,y_train))
#l1
def PolyLogisticRegression(degree,c,penalty='l2'):
    return Pipeline([
        ('poly',PolynomialFeatures(degree=degree)),
        ('std_scaler',StandardScaler()),
        ('LR',LinearRegression(c=c,penalty=penalty))
    ])
poly4=PolyLogisticRegression(degree=20,c=0.1,penalty="l1")
poly4.fit(x_train,y_train)
print(poly4.score(x_train,y_train))