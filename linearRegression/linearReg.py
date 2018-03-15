import numpy as np
import matplotlib.pyplot as plt


def loadData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineData = [float(lineArr[0]), float(lineArr[1])]
        dataMat.append(lineData)
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def loadBigData(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split()
        lineList = []
        for i in range(len(lineArr) -1):
            lineList.append(float(lineArr[i]))
        labelMat.append(float(lineArr[-1]))
        dataMat.append(lineList)
    return dataMat, labelMat



def showData(dataArr, labelArr):
    X = np.mat(dataArr)
    Y = np.mat(labelArr)

    Ypre1 = lwlr(dataArr, labelArr, 1.0)
    Ypre2 = lwlr(dataArr, labelArr, 0.1)
    Ypre3 = lwlr(dataArr, labelArr, 0.003)
    srtInd = X[:, 1].argsort(0)
    xSort = X[srtInd][:, 0]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
    axs[0].plot(xSort[:, 1], Ypre1[srtInd], c='red')
    axs[1].plot(xSort[:, 1], Ypre2[srtInd], c='red')
    axs[2].plot(xSort[:, 1], Ypre3[srtInd], c='red')
    axs[0].scatter(X[:, 1].flatten().A[0], Y.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[1].scatter(X[:, 1].flatten().A[0], Y.flatten().A[0], s=20, c='blue', alpha=.5)
    axs[2].scatter(X[:, 1].flatten().A[0], Y.flatten().A[0], s=20, c='blue', alpha=.5)

    axs0_title_text = axs[0].set_title(u'lwlr curve,k=1.0')
    axs1_title_text = axs[1].set_title(u'lwlr curve,k=0.01')
    axs2_title_text = axs[2].set_title(u'lwlr curve,k=0.003')
    plt.setp(axs0_title_text, size=8, weight='bold', color='red')
    plt.setp(axs1_title_text, size=8, weight='bold', color='red')
    plt.setp(axs2_title_text, size=8, weight='bold', color='red')
    plt.xlabel('X')
    plt.show()


def linearRegression(dataArr, labelArr):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    squarX = dataMat.T * dataMat
    squarXDet = np.linalg.det(squarX)
    if squarXDet == 0:
        print('非奇异矩阵，不可求逆')
        return
    weights = squarX.I * dataMat.T * labelMat
    return weights


def lwlr(dataArr, labelArr, k):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m = np.shape(dataMat)[0]
    # 权重对角阵
    weights = np.mat(np.eye(m))
    preY = np.zeros(m)

    for i in range(m):
        for j in range(m):
            diffMat = dataMat[i, :] - dataMat[j, :]
            weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
        squarX = dataMat.T * weights * dataMat
        squarXDet = np.linalg.det(squarX)
        if squarXDet == 0:
            print('非奇异矩阵，不可求逆')
            return
        omiga = squarX.I * dataMat.T * weights * labelMat
        preY[i] = (float(dataMat[i, :] * omiga))
    # 返回预测值矩阵mx1
    return preY


def testErroRate(testLabelMat, predictLabelMat):
    return ((testLabelMat - predictLabelMat)**2).sum()



if __name__ == '__main__':
    dataArr, labelArr = loadData('ex0.txt')
    # weights = linearRegression(dataArr, labelArr)
    showData(dataArr, labelArr)
