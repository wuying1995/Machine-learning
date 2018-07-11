#scikit-learn 中的PCA

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
digits=datasets.load_digits()
x=digits.data
y=digits.target
#随机数据
##x=np.empty((100,2))
##x[:,0] = np.random.uniform(0.,100.,size=100)
##x[:,1] = 0.75 * x[:,0] +3.+np.random.normal(0,10.,size=100)
##plt.scatter(x[:,0],x[:,1])
##plt.show()

#train_test_split 分为训练数据集，测试数据集
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=666)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train,y_train)
print(knn_clf.score(x_test,y_test))

#pca 降维
from sklearn.decomposition import PCA
# pca=PCA(n_components=2)
# pca.fit(x_train)
# x_train_reduction=pca.transform(x_train)
# x_test_reduction=pca.transform(x_test)
#
# knn_clf=KNeighborsClassifier()
# knn_clf.fit(x_train_reduction,y_train)
# print(knn_clf.score(x_test_reduction,y_test))

pca=PCA(0.95)
pca.fit(x_train)
print(pca.n_components_)
x_train_reduction=pca.transform(x_train)
x_test_reduction=pca.transform(x_test)

knn_clf=KNeighborsClassifier()
knn_clf.fit(x_train_reduction,y_train)
print(knn_clf.score(x_test_reduction,y_test))