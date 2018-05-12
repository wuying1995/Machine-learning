from sklearn.linear_model import LogisticRegression

def colicSklearn():
    frTrain = open('horseColicTraining.txt')  # 打开训练集
    frTest = open('horseColicTest.txt')  # 打开测试集
    trainingSet = [];
    trainingLabels = []
    testSet = [];
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[-1]))
    #clf = LogisticRegression(solver='liblinear',max_iter=10).fit(trainingSet,trainingLabels)
    clf = LogisticRegression(solver='sag', max_iter=5000).fit(trainingSet, trainingLabels)
    test_accuracy = clf.score(testSet,testLabels)*100
    print('正确率:%f%%' % test_accuracy)

colicSklearn()