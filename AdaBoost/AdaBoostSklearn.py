from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def loadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()

        for i in range(len(lineArr)):
            lineArr[i] = float(lineArr[i])
        labelMat.append(int(lineArr[-1]))
        del lineArr[-1]
        dataMat.append(lineArr)
    return dataMat, labelMat


if __name__ == '__main__':
    trainDataArr, trainLabelArr = loadData('horseColicTraining2.txt')
    testDataArr, testLabelArr = loadData('horseColicTest2.txt')

    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=10)
    bdt.fit(trainDataArr, trainLabelArr)
    predictions = bdt.predict(trainDataArr)

    errArr = np.mat(np.ones((len(trainLabelArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != trainLabelArr].sum() / len(trainDataArr) * 100))
    predictions = bdt.predict(testDataArr)
    errArr = np.mat(np.ones((len(testDataArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testDataArr) * 100))