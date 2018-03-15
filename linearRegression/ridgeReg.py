import numpy as np
import matplotlib.pyplot as plt


def loadData(fileName):
    dataArr = []
    labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineList = []
        for i in range(len(lineArr) - 1):
            lineList.append(float(lineArr[i]))
        labelArr.append(float(lineArr[-1]))
        dataArr.append(lineList)
    return dataArr, labelArr


def preprocessing(dataArr, labelArr):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)

    xMeans = np.mean(dataMat, axis=0)
    xVars = np.var(dataMat, axis=0)
    xNew = (dataMat - xMeans) / xVars

    yMean = np.mean(labelMat)
    yVar = np.var(labelMat)
    yNew = (labelMat - yMean) / yVar

    return xNew, yNew


def ridgeRegression(X, y):
    allWeights = []
    lamCount = 30
    for j in range(lamCount):
        i = np.eye(np.shape(X)[1])
        lam = np.exp(j - 10)
        denom = X.T * X + lam * i
        if np.linalg.det(denom) == 0.0:
            print('非奇异矩阵')
            continue
        w = denom.I * X.T * y.T
        weights = w.T.flatten().A[0]
        allWeights.append(weights)
    return allWeights


def showData(dataMat, labelMat):
    allWeights = ridgeRegression(dataMat, labelMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(allWeights)

    ax_title_text = ax.set_title(u'log(lambada) relationship with w')
    ax_xlabel_text = ax.set_xlabel(u'log(lambada)')
    ax_ylabel_text = ax.set_ylabel(u'w')
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    dataArr, labelArr = loadData('abalone.txt')
    X, y = preprocessing(dataArr, labelArr)
    showData(X, y)
