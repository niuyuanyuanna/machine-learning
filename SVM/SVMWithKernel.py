import numpy as np
import matplotlib.pyplot as plt
import random
from os import listdir


class optStruct:
    def __init__(self, dataMat, labelMat, C, toler, kTup):
        self.X = dataMat
        self.labels = labelMat
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMat)[0]
        self.b = 0
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrick(self.X, self.X[i, :], kTup)


def kernelTrick(dataMat, vectorXi, kTup):
    m, n = np.shape(dataMat)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':  # 线性核函数
        K = dataMat * vectorXi.T
    elif kTup[0] == 'rbf':  # 高斯核函数
        for j in range(m):
            deltaRow = dataMat[j, :] - vectorXi
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1] ** 2))
    else:
        raise NameError('invalid kernel')
    return K


def calcEk(oS, k):
    fxk = float(np.multiply(oS.alphas, oS.labels).T * oS.K[:, k] + oS.b)
    Ek = fxk - float(oS.labels[k])
    return Ek


def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheIndex = np.nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheIndex > 1):
        for k in validEcacheIndex:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaEk = abs(Ei - Ek)
            if deltaEk > maxDeltaE:
                maxDeltaE = deltaEk
                maxK = k
                Ej = Ek
        return maxK, Ej
    else:
        j = i
        while j == i:
            j = int(random.uniform(0, oS.m))
        Ej = calcEk(j)
        return j, Ej


def loadData(fileName):
    fr = open(fileName)
    dataMat = []
    labelMat = []
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[-1]))
    return dataMat, labelMat


def img2vector(fileName):
    returnVect = np.zeros((1, 1024))
    fr = open(fileName)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImage(dirName):
    # from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def showData(dataMat, labelMat):
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

    plt.show()


def clipAlpha(ai, H, L):
    if ai > H:
        ai = H
    if L > ai:
        ai = L
    return ai


def updateEk(oS, k):
    Ek = calcEk(oS, k)  # 计算Ek
    oS.eCache[k] = [1, Ek]


def innerLoop(i, oS):
    Ei = calcEk(oS, i)
    # 优化alpha,设定一定的容错率。
    if ((oS.labels[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or (
        (oS.labels[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        # 使用内循环启发方式2选择alpha_j,并计算Ej
        j, Ej = selectJ(i, oS, Ei)
        # 保存更新前的aplpha值，使用深拷贝
        alphaIold = oS.alphas[i].copy();
        alphaJold = oS.alphas[j].copy();
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
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print("eta>=0")
            return 0
        # 步骤4：更新alpha_j
        oS.alphas[j] -= oS.labels[j] * (Ei - Ej) / eta
        # 步骤5：修剪alpha_j
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        # 更新Ej至误差缓存
        updateEk(oS, j)
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print("alpha_j变化太小")
            return 0
        # 步骤6：更新alpha_i
        oS.alphas[i] += oS.labels[j] * oS.labels[i] * (alphaJold - oS.alphas[j])
        # 更新Ei至误差缓存
        updateEk(oS, i)
        # 步骤7：更新b_1和b_2
        b1 = oS.b - Ei - oS.labels[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
             oS.labels[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labels[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
             oS.labels[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        # 步骤8：根据b_1和b_2更新b
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def SMOWithKernel(dataMat, labelMat, maxCycles, toler, C, kTup):
    oS = optStruct(np.mat(dataMat), np.mat(labelMat).transpose(), C, toler, kTup)
    cycle = 0
    entireSet = True
    alphaPairChange = 0
    while (cycle < maxCycles) and ((alphaPairChange > 0) or entireSet):
        alphaPairChange = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairChange += innerLoop(i, oS)
                print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (cycle + 1, i, alphaPairChange))
            cycle += 1
        else:
            nonBoundLs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundLs:
                alphaPairChange += innerLoop(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (cycle + 1, i, alphaPairChange))
            cycle += 1
        if entireSet:
            entireSet = False
        elif alphaPairChange == 0:
            entireSet = True
        print('iteration number: %d' % (cycle + 1))
    return oS.b, oS.alphas


def classify(dataMat, labelMat, testMat, testLabel, alphas, kTup, b):
    trainDataMat = np.mat(dataMat)
    trainLabelMat = np.mat(labelMat).transpose()
    erroCount = 0
    m, n = np.shape(testMat)

    svIndex = np.nonzero(alphas.A > 0)[0]
    svDataMat = trainDataMat[svIndex]
    svLabelMat = trainLabelMat[svIndex]

    for i in range(m):
        # 支撑向量点计算支撑超平面
        kernelEval = kernelTrick(svDataMat, trainDataMat[i, :], kTup)
        predict = kernelEval.T * np.multiply(svLabelMat, alphas[svIndex]) + b
        if np.sign(predict) != np.sign(trainLabelMat[i]):
            erroCount += 1
    print('training set erro rate is %d' % (float(erroCount / m) * 100))

    testDataMat = np.mat(testMat)
    testLabelMat = np.mat(testLabel).transpose()
    erroCount = 0
    m, n = np.shape(testDataMat)

    for j in range(m):
        kernelEval = kernelTrick(svDataMat, testDataMat[j, :], kTup)
        predict = kernelEval.T * np.multiply(svLabelMat, alphas[svIndex]) + b
        if np.sign(predict) != np.sign(testLabelMat[j]):
            erroCount += 1
    print('testing set erro rate is %d' % (float(erroCount / m) * 100))


if __name__ == '__main__':
    # trainDataArr, trainLabelArr = loadData('trainingSetRBF.txt')
    # testDataArr, testLabelArr = loadData('testSetRBF.txt')
    # showData(trainDataMat, trainLabelMat)
    trainDataArr, trainLabelArr = loadImage('trainingDigits')
    testDataArr, testLabelArr = loadImage('testDigits')
    kTup = ('rbf', 1.3)
    b, alphas = SMOWithKernel(trainDataArr, trainLabelArr, 400, 0.0001, 100, kTup)
    classify(trainDataArr, trainLabelArr, testDataArr, testLabelArr, alphas, kTup, b)
