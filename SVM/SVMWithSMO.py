import numpy as np
import matplotlib.pyplot as plt
import random


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def showDataSet(dataMat, labelMat, w, b):
    positiveData = []
    negativeData = []
    for i in range(len(labelMat)):
        if labelMat[i] > 0:
            positiveData.append(dataMat[i])
        else:
            negativeData.append(dataMat[i])
    positiveDataNp = np.array(positiveData)
    negativeDataNp = np.array(negativeData)
    plt.scatter(np.transpose(positiveDataNp)[0], np.transpose(positiveDataNp)[1])
    plt.scatter(np.transpose(negativeDataNp)[0], np.transpose(negativeDataNp)[1])

    x1 = max(dataMat)[0]     # x1为第一列特征的最大值
    x2 = min(dataMat)[0]     # x2为第一列特征的最小值
    a1, a2 = w
    b = float(b)
    a1 = float(a1[0])
    a2 = float(a2[0])
    y1 = (-b - a1 * x1) / a2
    y2 = (-b - a1 * x2) / a2
    plt.plot([x1, x2], [y1, y2])

    # 找出支持向量点
    for i, alpha in enumerate(alphas):
        if alpha > 0:
            x, y = dataMat[i]
            plt.scatter([x], [y], s=150, c='none', alpha=0.7, linewidth=1.5, edgecolor='red')

    plt.show()


def smoAlgorithm(dataMat, labelMat, maxCycles, C=0.6, toler=0.01):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))
    for k in range(maxCycles):
        alphaPaireChange = 0
        for i in range(m):
            fxi = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[i, :].T)) + b
            ei = fxi - labelMatrix[i]
            if ((labelMatrix[i] * ei < -toler) and (alphas[i] < C)) or ((labelMatrix[i] * ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个与alpha_i成对优化的alpha_j
                j = i
                while j == i:
                    j = int(random.uniform(0, m))

                # 步骤1：计算误差Ej
                fxj = float(np.multiply(alphas, labelMatrix).T * (dataMatrix * dataMatrix[j, :].T)) + b
                ej = fxj - float(labelMatrix[j])
                # 保存更新前的aplpha值，使用深拷贝
                alphaIOld = alphas[i].copy()
                alphaJOld = alphas[j].copy()

                # 步骤2：计算上下界L和H
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue

                # 步骤3：计算eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                      - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 步骤4：更新alpha_j
                alphas[j] -= labelMatrix[j] * (ei - ej) / eta

                # 步骤5：修剪alpha_j
                if alphas[j] > H:
                    alphas[j] = H
                elif alphas[j] < L:
                    alphas[j] = L

                if (abs(alphas[j] - alphaJOld) < 0.00001):
                    print("alpha_j变化太小")

                # 步骤6：更新alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJOld - alphas[j])

                # 步骤7：更新b1和b2
                b1 = b - ei \
                     - labelMatrix[i] * (alphas[i] - alphaIOld) * dataMatrix[i, :] * dataMatrix[i, :].T \
                     - labelMatrix[j] * (alphas[j] - alphaJOld) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - ej \
                     - labelMatrix[i] * (alphas[i] - alphaIOld) * dataMatrix[j, :] * dataMatrix[i, :].T \
                     - labelMatrix[j] * (alphas[j] - alphaJOld) * dataMatrix[j, :] * dataMatrix[j, :].T

                # 步骤8：根据b1和b2更新b
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[i] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphaPaireChange += 1
                print("第%d次迭代 样本:%d, alpha优化次数:%d" % (k, i, alphaPaireChange))

    return b, alphas


def culcuW(alphas, dataMat, labelMat):
    alphas = np.array(alphas)
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    labelMatRe = labelMat.reshape(1, -1)          # 变为1xm的矩阵
    labelMatRe2 = np.tile(labelMatRe.T, (1, 2))   # 复制labelMatRe.T一行两列 变为mx2的矩阵
    labelMatRe3 = (labelMatRe2 * dataMat).T       # 使用array直接做乘法是矩阵中对应元素相乘等同于numpy.multiply
    w = np.dot(labelMatRe3, alphas)               # 得到w为2x1的矩阵
    return w.tolist()


if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # showDataSet(dataMat, labelMat)
    b, alphas = smoAlgorithm(dataMat, labelMat, 150)
    print(b)
    print(alphas)
    # w = culcuW(alphas, dataMat, labelMat)
    # showDataSet(dataMat, labelMat, w, b)
