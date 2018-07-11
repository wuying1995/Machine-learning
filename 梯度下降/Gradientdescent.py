from sklearn import datasets

boston=datasets.load_boston()
x=boston.data
y=boston.target
x=x[y<50.0]
y=y[y<50.0]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)

#x需要进行归一化处理
from sklearn.preprocessing import StandardScaler
standardScaler=StandardScaler()
standardScaler.fit(x_train)
x_train_standard=standardScaler.transform(x_train)
x_test_standard=standardScaler.transform(x_test)

from sklearn.linear_model import SGDRegressor
Slin=SGDRegressor(n_iter=100)
Slin.fit(x_train_standard,y_train)
print(Slin.score(x_test_standard,y_test))