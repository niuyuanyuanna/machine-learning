import matplotlib.pyplot as plt
import numpy as np
import random


def load_dataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[-1]))
    fr.close()
    return dataMat, labelMat


def plot_dataSet(weights, dataMat, labelMat):
    dataArr = np.array(dataMat)
    n = np.shape(dataMat)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)
    ax.plot(x, y)
    plt.title('DataSet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def sigmoid(input_X):
    return 1.0 / (1 + np.exp(-input_X))


def grad_ascent(dataMat, labelMat, maxCycles):
    dataMatrix = np.mat(dataMat)
    labelMatrix = np.mat(labelMat).transpose()  # 转置
    m, n = np.shape(dataMatrix)  # 此处m为数据的个数，n为θ的个数
    alpha = 0.001
    weights = np.ones((n, 1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)  # h(X)=g(θ'X)   θ:nx1  X:mxn
        error = labelMatrix - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles, n)
    return weights.getA(), weights_array  # 将矩阵转换为数组


def rand_grad_ascent(dataMatrix, labelMat, maxCycles):
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    weights_array = np.ones([])
    for j in range(maxCycles):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))  # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))  # 选择随机选取的一个样本，计算h
            error = labelMat[randIndex] - h  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)  # 添加回归系数到数组中
            del (dataIndex[randIndex])  # 删除已经使用的样本
    weights_array = weights_array.reshape(maxCycles * m, n)  # 改变维度
    return weights, weights_array  # 返回


def stocGradAscent1(dataMatrix, classLabels, numIter):
    m,n = np.shape(dataMatrix)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = np.ones(n)                                                       #参数初始化
    weights_array = np.array([])                                            #存储每次更新的回归系数
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0, len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]       #更新回归系数
            weights_array = np.append(weights_array, weights, axis=0)         #添加回归系数到数组中
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    weights_array = weights_array.reshape(numIter*m, n)                         #改变维度
    return weights, weights_array


def plotWeights(weights_array1, weights_array2):
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=3,nclos=2时,代表fig画布被分为六个区域,axs[0][0]表示第一行第一列
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(20, 10))
    x1 = np.arange(0, len(weights_array1), 1)
    # 绘制w0与迭代次数的关系
    axs[0][0].plot(x1, weights_array1[:, 0])
    axs0_title_text = axs[0][0].set_title(u'梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs1_ylabel_text = axs[1][0].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs2_xlabel_text = axs[2][0].set_xlabel(u'迭代次数')
    axs2_ylabel_text = axs[2][0].set_ylabel(u'W2')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0与迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs0_title_text = axs[0][1].set_title(u'改进的随机梯度上升算法：回归系数与迭代次数关系')
    axs0_ylabel_text = axs[0][1].set_ylabel(u'W0')
    plt.setp(axs0_title_text, size=20, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w1与迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs1_ylabel_text = axs[1][1].set_ylabel(u'W1')
    plt.setp(axs1_ylabel_text, size=20, weight='bold', color='black')
    # 绘制w2与迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs2_xlabel_text = axs[2][1].set_xlabel(u'迭代次数')
    axs2_ylabel_text = axs[2][1].set_ylabel(u'W1')
    plt.setp(axs2_xlabel_text, size=20, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=20, weight='bold', color='black')

    plt.show()


if __name__ == '__main__':
    dataMat, labelMat = load_dataSet()
    # weights = grad_ascent(dataMat, labelMat)
    weights_1, weights_array_1 = rand_grad_ascent(dataMat, labelMat, 150)
    weights_2, weights_array_2 = grad_ascent(dataMat, labelMat, 150)
    # weights_1, weights_array_1 = stocGradAscent1(np.array(dataMat), labelMat, 150)


    # plot_dataSet(weights, dataMat, labelMat)
    # plot_dataSet()
    plotWeights(weights_array_1, weights_array_2)
