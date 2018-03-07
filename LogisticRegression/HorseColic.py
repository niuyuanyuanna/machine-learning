import numpy as np
import random
from sklearn.linear_model import LogisticRegression


def loadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineList = []
        lineList.append(1.0)
        for i, data in enumerate(lineArr):
            if i == len(lineArr) - 1:
                break
            else:
                lineList.append(float(data))
        dataMat.append(lineList)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def gradAscent(dataMat, labelMat, maxCycles=150):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).transpose()  # 转置
    m, n = np.shape(dataMatrix)  # 此处m为数据的个数，n为θ的个数
    alpha = 0.001
    weights = np.ones((n, 1))
    # weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # h(X)=g(θ'X)   θ:nx1  X:mxn
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.transpose() * error
        # weights_array = np.append(weights_array, weights)
    # weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA()  # 将矩阵转换为数组


def randGradAscent(dataMat, labelMat, maxCycles=150):
    dataMatrix = np.array(dataMat)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for k in range(maxCycles):
        dataIndex = list(range(m))
        for i in range(m):
            randIndex = int(random.uniform(0, len(dataIndex)))
            alpha = 4 / (1.0 + k + i) + 0.01
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = labelMat[randIndex] - h
            weights = weights + alpha * dataMatrix[randIndex] * error
            del dataIndex[randIndex]
    return np.mat(weights).transpose()


def sklearnColic(trainingSet, trainingLabels, testSet, testLabels):
    classifier = LogisticRegression(solver='liblinear', max_iter=10).\
        fit(trainingSet, trainingLabels)
    testAccurcy = classifier.score(testSet, testLabels) * 100
    return testAccurcy

def classifyVector(weights, testDataMat, testLabelMat):
    testDataMat = np.mat(testDataMat)
    probList = sigmoid(testDataMat * weights).getA()
    erroCount = 0
    for i, prob in enumerate(probList):
        if prob > 0.5:
            probList[i] = 1
        else:
            probList[i] = 0

    for j in range(len(testLabelMat)):
        if probList[j] != testLabelMat[j]:
            erroCount += 1
    return float(erroCount) / len(testLabelMat) * 100


def sigmoid(input_X):
    return 1.0 / (1 + np.exp(-input_X))


if __name__ == '__main__':
    trainDataMat, trainLabelMat = loadData('horseColicTraining.txt')
    testDataMat, testLabelMat = loadData('horseColicTest.txt')
    weights1 = gradAscent(trainDataMat, trainLabelMat)
    weights2 = randGradAscent(trainDataMat, trainLabelMat)
    # erroRate1 = classifyVector(weights1, testDataMat, testLabelMat)
    erroRate2 = classifyVector(weights2, testDataMat, testLabelMat)
    accuracy = sklearnColic(trainDataMat, trainLabelMat, testDataMat, testLabelMat)
    erroRate3 = 100.0 - accuracy
    # print(erroRate1)
    # print(erroRate2)
    print(erroRate3)
