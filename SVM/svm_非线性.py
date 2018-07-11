import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
x,y=datasets.make_moons()
print(x.shape)
plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()
x,y=datasets.make_moons(noise=0.15,random_state=666)
plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()

#多项是特征
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
def Polynomialsvm(degree,C):
    return Pipeline([
        ("poly",PolynomialFeatures(degree=degree)),
        ("std",StandardScaler()),
        ("LinearSVC",LinearSVC(C=C))
    ])
poly_svc=PolynomialFeatures(degree=3)
poly_svc.fit(x,y)
#决策边界
def plot_decision_boundary(model,axis):
  x0,x1=np.meshgrid(
      np.linspace(axis[0],axis[1],int((axis[1]-axis[0]))*100).reshape(-1,1),
      np.linspace(axis[2],axis[3], int((axis[3] - axis[2])) * 100).reshape(-1,1)
  )
  x_new=np.c_[x0.ravel(),x1.ravel()]
  y_predict=model.predict(x)
  zz=y_predict.reshape(x0.shape)
  from matplotlib.colors import ListedColormap
  custom_cmap=ListedColormap(["#EF9A9A,#FFF59D,#90CAF9"])
  plt.contourf(x0,x1,zz,linewidth=5,camp=custom_cmap)


def plot_decision_boundary(model, axis):  # 绘图模块
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, linewidth=5, cmap=custom_cmap)


from sklearn.preprocessing import StandardScaler  # 数据标准化

standarscaler = StandardScaler()
standarscaler.fit(x)
x_standar = standarscaler.transform(x)
from sklearn.svm import LinearSVC  # SVC指的是support vector classifier

svc = LinearSVC(C=1e9)  # 这里先传入比较大的C，也就是约束ξ值
svc.fit(x_standar, y)
plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(x_standar[y == 0, 0], x_standar[y == 0, 1])
plt.scatter(x_standar[y == 1, 0], x_standar[y == 1, 1])
plt.show()  # 如下左图，将C设为0.01，这时允许ξ比较大，如下右图所示，将会产生错分样本

#使用多项式核svm
from sklearn.svm import SVC
def poly_(degree,C=1.0):
    return Pipeline([
        ("std",StandardScaler()),
        ("kernelSVC",SVC(kernel="poly",degree=degree,C=C))
    ])
poly_svc=poly_(degree=3)
poly_svc.fit(x,y)
#决策边界
