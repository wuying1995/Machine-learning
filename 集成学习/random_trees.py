
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
x,y=datasets.make_moons(n_samples=500,noise=0.3,random_state=42)
plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()

#

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(n_estimators=500,random_state=666,oob_score=True,n_jobs=-1)
RF.fit(x,y)
print(RF.oob_score_)

RF2=RandomForestClassifier(n_estimators=500,max_leaf_nodes=16,random_state=666,oob_score=True,n_jobs=-1)
RF2.fit(x,y)
print(RF2.oob_score_)

#extra_trees
from sklearn.ensemble import ExtraTreesClassifier
EF=ExtraTreesClassifier(n_estimators=500,bootstrap=True,random_state=666,oob_score=True)
EF.fit(x,y)
print(EF.oob_score_)
#以上 解决都是分类，都有相关回归问题
from sklearn.ensemble import RandomForestRegressor
