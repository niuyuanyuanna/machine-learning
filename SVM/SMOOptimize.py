import numpy as np
import matplotlib.pyplot as plt
import random


class optStruct:
    def __init__(self, dataMat, labelMat, toler, C):
        self.X = dataMat
        self.labels = labelMat
        self.tol = toler
        self.C = C
        self.m = np.shape(dataMat)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))  # 误差缓存


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def calcEk(oS, k):
    fxk = float(np.multiply(oS.alphas, oS.labels).T * (oS.X * oS.X[k, :].T) + oS.b)
    Ek = fxk - float(oS.labels[k])
    return Ek


# 启发式方法寻找j，选择具有最大步长的j
def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheIndex = np.nonzero(oS.eCache[:, 0].A)[0]  # 返回误差不为0的数据索引
    if len(validEcacheIndex) > 1:
        for k in validEcacheIndex:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ek - Ei)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = i
        while j == i:
            j = int(random.uniform(0, oS.m))
        Ej = calcEk(oS, j)
        return j, Ej


# 内循环，完成alpha，b的更新
def innerLoop(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labels[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) \
            or ((oS.labels[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 启发式选择与alpha_i成对优化的alpha_j并计算误差Ej
        j, Ej = selectJ(i, oS, Ei)

        # 保存更新前的aplpha值，使用深拷贝
        alphaIOld = oS.alphas[i].copy()
        alphaJOld = oS.alphas[j].copy()

        # 步骤2：计算上下界L和H
        if oS.labels[i] != oS.labels[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L==H")
            return 0

        # 步骤3：计算eta
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T \
              - oS.X[i, :] * oS.X[i, :].T \
              - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0

        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labels[j] * (Ei - Ej) / eta

        # 步骤5：修剪alpha_j
        if oS.alphas[j] > H:
            oS.alphas[j] = H
        elif oS.alphas[j] < L:
            oS.alphas[j] = L

        # 更新alpha_j的错误缓存
        EjNew = calcEk(oS, j)
        oS.eCache[j] = [1, EjNew]

        if abs(oS.alphas[j] - alphaJOld) < 0.00001:
            print("alpha_j变化太小")
            return 0

        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labels[j] * oS.labels[i] * (alphaJOld - oS.alphas[j])
        # 更新alpha_i错误缓存
        EiNew = calcEk(oS, i)
        oS.eCache[i] = [1, EiNew]

        # 步骤7：更新b1和b2
        b1 = oS.b - Ei \
             - oS.labels[i] * (oS.alphas[i] - alphaIOld) * oS.X[i, :] * oS.X[i, :].T \
             - oS.labels[j] * (oS.alphas[j] - alphaJOld) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej \
             - oS.labels[i] * (oS.alphas[i] - alphaIOld) * oS.X[j, :] * oS.X[i, :].T \
             - oS.labels[j] * (oS.alphas[j] - alphaJOld) * oS.X[j, :] * oS.X[j, :].T

        # 步骤8：根据b1和b2更新b
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[i] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0

        return 1
    else:
        return 0


def smoOptAlgorithm(dataMat, labelMat, maxCycles, C=0.6, toler=0.01):
    oS = optStruct(np.mat(dataMat), np.mat(labelMat).transpose(), toler, C)
    cycle = 0
    entireSet = True
    alphaPaireChange = 0

    while (cycle < maxCycles) and ((alphaPaireChange > 0) or (entireSet)):
        alphaPaireChange = 0
        if entireSet:
            for i in range(oS.m):
                alphaPaireChange += innerLoop(i, oS)
                print('fullset, iter: %d i: %d, pairs changed %d ' % (cycle, i, alphaPaireChange))
            cycle += 1
        else:
            # 遍历非边界值
            nonBoundLs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundLs:
                alphaPaireChange += innerLoop(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (cycle, i, alphaPaireChange))
            cycle += 1
        if entireSet:
            entireSet = False
        elif alphaPaireChange == 0:
            entireSet = True
        print('iteration number: %d' % cycle)
    return oS.b, oS.alphas


def calcW(alphas, dataMat, labelMat):
    alphas = np.array(alphas)
    dataMat = np.array(dataMat)
    labelMat = np.array(labelMat)
    labelMatRe = labelMat.reshape(1, -1)          # 变为1xm的矩阵
    labelMatRe2 = np.tile(labelMatRe.T, (1, 2))   # 复制labelMatRe.T一行两列 变为mx2的矩阵
    labelMatRe3 = (labelMatRe2 * dataMat).T       # 使用array直接做乘法是矩阵中对应元素相乘等同于numpy.multiply
    w = np.dot(labelMatRe3, alphas)               # 得到w为2x1的矩阵
    return w.tolist()


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



if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    b, alphas = smoOptAlgorithm(dataMat, labelMat, 150)
    # print(b)
    # print(alphas)
    w = calcW(alphas, dataMat, labelMat)
    # print(w)
    # showDataSet(dataMat, labelMat, w, b)

