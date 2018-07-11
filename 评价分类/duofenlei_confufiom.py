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
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.8,random_state=666)

#十分类问题  LR 采用ovr解决多分类问题
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))

y_predict=clf.predict(x_test)


#sklearn 混淆矩阵

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_predict))
cfm=confusion_matrix(y_test,y_predict)
plt.matshow(cfm,cmap=plt.cm.gray)
plt.show()
#
row_sum=np.sum(cfm,axis=1)
err_matrix=cfm/row_sum
np.fill_diagonal(err_matrix,0)
print(err_matrix)
plt.matshow(err_matrix,cmap=plt.cm.gray)
plt.show()
#准确率 默认解决二分类
from sklearn.metrics import precision_score
print(precision_score(y_test,y_predict,average="micro"))

#召回率
from sklearn.metrics import recall_score
print(recall_score(y_test,y_predict))

#f1 score

from sklearn.metrics import f1_score
print(f1_score(y_test,y_predict))

#decision _score  对预值调整
decision_scores=clf.decision_function(x_test)
y_predict2=np.array(decision_scores>=5,dtype="int")

print(confusion_matrix(y_test,y_predict))
print(precision_score(y_test,y_predict))
print(recall_score(y_test,y_predict))

#sklearn 中的Precision_Recall 曲线
from sklearn.metrics import precision_recall_curve
precisions,recalls,threeholds=precision_recall_curve(y_test,decision_scores)
plt.plot(threeholds,precisions[:-1])
plt.plot(threeholds,recalls[:-1])
plt.show()
plt.plot(precisions,recalls)
plt.show()

#sklearn 中的ROC
from sklearn.metrics import roc_curve
fprs,tprs,threeholds=roc_curve(y_test,decision_scores)
plt.plot(fprs,tprs)
plt.show()

#ROC曲线的而面积大的
from  sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test,decision_scores))






# #pca 降维
# from sklearn.decomposition import PCA
# # pca=PCA(n_components=2)
# # pca.fit(x_train)
# # x_train_reduction=pca.transform(x_train)
# # x_test_reduction=pca.transform(x_test)
# #
# # knn_clf=KNeighborsClassifier()
# # knn_clf.fit(x_train_reduction,y_train)
# # print(knn_clf.score(x_test_reduction,y_test))
#
# pca=PCA(0.95)
# pca.fit(x_train)
# print(pca.n_components_)
# x_train_reduction=pca.transform(x_train)
# x_test_reduction=pca.transform(x_test)
#
# knn_clf=KNeighborsClassifier()
# knn_clf.fit(x_train_reduction,y_train)
# print(knn_clf.score(x_test_reduction,y_test))