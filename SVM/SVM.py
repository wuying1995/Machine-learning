import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris=datasets.load_iris()
x=iris.data
y=iris.target
#二分类
x=x[y<2,:2]
y=y[y<2]
plt.scatter(x[y==0,0],x[y==0,1],color="red")
plt.scatter(x[y==1,0],x[y==1,1],color="blue")
plt.show()
#标准化工作
from sklearn.preprocessing import StandardScaler
sta=StandardScaler()
sta.fit(x)
x_sta=sta.transform(x)
#
from sklearn.svm import  LinearSVC
svc=LinearSVC(C=1e9)


