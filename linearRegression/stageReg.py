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


def culError(dataMat, labelMat, weights):
    wMat = np.mat(weights)
    yPre = dataMat * wMat
    diffMat = yPre - labelMat.T
    return float(diffMat.T * diffMat)


def stageWiseRegression(dataMat, labelMat, maxCycle, eps):
    m, n = np.shape(dataMat)
    returnMat = np.zeros((maxCycle, n))  # 初始化numIt次迭代的回归系数矩阵
    ws = np.zeros((n, 1))  # 初始化回归系数矩阵
    # wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(maxCycle):  # 迭代numIt次                                                               #打印当前回归系数矩阵
        lowestError = float('inf')
        for j in range(n):  # 遍历每个特征的回归系数
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                rssE = culError(dataMat, labelMat, wsTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T  # 记录numIt次迭代的回归系数矩阵
    return returnMat


def showData(allWeights):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(allWeights)

    ax_title_text = ax.set_title(u'iteration relationship with w')
    ax_xlabel_text = ax.set_xlabel(u'iteration count')
    ax_ylabel_text = ax.set_ylabel(u'w')
    plt.setp(ax_title_text, size=20, weight='bold', color='red')
    plt.setp(ax_xlabel_text, size=10, weight='bold', color='black')
    plt.setp(ax_ylabel_text, size=10, weight='bold', color='black')
    plt.show()


if __name__ == '__main__':
    dataArr, labelArr = loadData('abalone.txt')
    dataMat, labelMat = preprocessing(dataArr, labelArr)
    allWeights = stageWiseRegression(dataMat, labelMat, 1000, 0.001)
    showData(allWeights)


