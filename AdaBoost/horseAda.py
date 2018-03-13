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


def stumpClassify(dataMat, i, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMat)[0], 1))  # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMat[:, i] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMat[:, i] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray


def buildStump(dataArr, labelArr, D):
    dataMat = np.mat(dataArr)
    # labelMat = np.mat(labelArr)
    m, n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()  # 找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMat, i, threshVal, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                for k in range(len(errArr)):
                    if predictedVals[k] == labelArr[k]:
                        errArr[k] = 0
                # errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoost(dataArr, labelArr, numIt):
    labelMat = np.mat(labelArr).transpose()
    m = labelMat.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    weakClassArr = []
    # 初始化的权重
    D = np.mat(np.ones((m, 1)) / m)
    for i in range(numIt):
        # 计算错误率
        bestStump, minError, bestClassEst = buildStump(dataArr, labelArr, D)

        # 计算弱学习算法权重
        alpha = float(0.5 * np.log((1 - minError) / max(minError, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)

        # 更新D值
        numerator = np.exp(-1 * alpha * np.multiply(labelMat, bestClassEst))
        D = np.multiply(numerator, D)
        D = D / D.sum()
        # print('iter = ', i+1, 'D =', D.T)

        aggClassEst += alpha * bestClassEst
        # print('aggClassEst', aggClassEst.T)
        erroEst = (np.sign(aggClassEst) != labelMat)
        # print('erroEst:', erroEst.T)
        erroRate = erroEst.sum() / float(m)
        # print('errorRate:', erroRate)
        if erroRate == 0.0:
            break
    print('training erroRate:', erroRate)
    return erroEst, weakClassArr


def classify(testData, weakClassArr):
    testDataMat = np.mat(testData)
    m = np.shape(testDataMat)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for weakClass in weakClassArr:
        classEst = stumpClassify(testDataMat, weakClass['dim'],
                                 weakClass['thresh'], weakClass['ineq'])
        aggClassEst += weakClass['alpha'] * classEst
        # print('aggClassEst:', aggClassEst.T)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    trainDataArr, trainLabelArr = loadData('horseColicTraining2.txt')
    testDataArr, testLabelArr = loadData('horseColicTest2.txt')
    erroEst, weakClassArr = adaBoost(trainDataArr, trainLabelArr, 40)


    aggClassEst = classify(testDataArr, weakClassArr)
    testErroEst = (aggClassEst != np.mat(testLabelArr).T)
    erroCount = testErroEst.sum()
    erroRate = float(erroCount) / len(testDataArr)
    print('testing erroRate:', erroRate)

