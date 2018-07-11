import numpy as np
from sklearn import datasets
digits=datasets.load_digits()
x=digits.data
y=digits.target


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=666)
#knn=KNeighborsClassifier()
#print(cross_val_score(knn,x_train ,y_train))

#train_test_split调参

# best_score,best_p,best_k=0,0,0
# for k in range(2,11):
#     for p in range(1,6):
#         knn=KNeighborsClassifier(weights='distance',n_neighbors=k,p=p)
#         knn.fit(x_train,y_train)
#         score=knn.clf.score(x_test,y_test)
#         if score>best_score
#             best_score,best_p,best_k=score,p,k
#             print( best_score,best_p,best_k)


#交叉验证

best_score,best_p,best_k=0,0,0
for k in range(2,11):
    for p in range(1,6):
        knn=KNeighborsClassifier(weights='distance',n_neighbors=k,p=p)
        scores=cross_val_score(knn,x_train ,y_train)
        score=np.mean(scores)
        if score>best_score:
            best_score,best_p,best_k=score,p,k
            print("best_k=",best_k)
            print("best_p=",best_p)
            print("best_score=", best_score)


    best_knn=KNeighborsClassifier(weights="distance",n_neighbors=2,p=2)
    best_knn.fit(x_train,y_train)
    print(best_knn.score(x_test,y_test))