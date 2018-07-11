import numpy as np
class simpleLinearRegression(object):
    def _init_(self):
        self.a=None
        self.b=None
    def fit(self,x_train,y_train):
        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)
        num=(x_train-x_mean).dot(y_train-y_mean)
        d=(x_train-x_mean).dot(x_train-x_mean)
        self.a_=num/d
        self.b_=y_mean-self.a_*x_mean
        return self
    def predict(self,x_predict):
        return np.array([self._predict(x) for x in x_predict])
    def _predict(self,x_single):
        return self.a_*x_single+self.b_
    def __repr__(self):
        return "SimpleLinearRegression()"