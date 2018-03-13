import numpy as np
import matplotlib.pyplot as plt


def loadData():
    dataMat = np.mat(
        [[1., 2.1],
         [1.5, 1.6],
         [1.3, 1.],
         [1., 1.],
         [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def showData(dataMat, classLabels):
    positiveData = []
    negativeData = []
    for i in range(len(dataMat)):
        if classLabels[i] > 0:
            positiveData.append(dataMat[i])
        else:
            negativeData.append(dataMat[i])
    posDataArr = np.array(positiveData)
    negativeData = np.array(negativeData)
    plt.scatter(np.transpose(posDataArr)[0], np.transpose(posDataArr)[1])  # 正样本散点图
    plt.scatter(np.transpose(negativeData)[0], np.transpose(negativeData)[1])  # 负样本散点图
    plt.show()


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
    D = np.mat(np.ones((5, 1)) / 5)
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
        print('iter = ', i+1, 'D =', D.T)

        aggClassEst += alpha * bestClassEst
        print('aggClassEst', aggClassEst.T)
        erroEst = (np.sign(aggClassEst) != labelMat)
        print('erroEst:', erroEst.T)
        erroRate = erroEst.sum() / float(m)
        print('errorRate:', erroRate)
        if erroRate == 0.0:
            break
    return erroEst, weakClassArr


def classify(testData, weakClassArr):
    testDataMat = np.mat(testData)
    m = np.shape(testDataMat)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for weakClass in weakClassArr:
        classEst = stumpClassify(testDataMat, weakClass['dim'],
                                 weakClass['thresh'], weakClass['ineq'])
        aggClassEst += weakClass['alpha'] * classEst
        print('aggClassEst:', aggClassEst.T)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataArr, labelArr = loadData()
    # 初始化的权重
    erroEst, weakClassArr = adaBoost(dataArr, labelArr, 40)
    result = classify([[0, 0], [5, 5]], weakClassArr)