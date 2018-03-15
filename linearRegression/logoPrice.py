from bs4 import BeautifulSoup
import numpy as np
import random


def loadData(inFile, yr, numPce, origPrc):
    retX = []
    retY = []
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while len(currentRow) != 0:
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                # print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)
    return retX, retY


def linearRegression(dataMat, labelMat):
    squarX = dataMat.T * dataMat
    squarXDet = np.linalg.det(squarX)
    if squarXDet == 0:
        print('非奇异矩阵，不可求逆')
        return
    weights = squarX.I * dataMat.T * labelMat.T
    return weights


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
    return np.mat(allWeights)


def crossValidation(dataArr, labelArr, numVal):
    m = len(labelArr)
    indexList = list(range(m))
    erroMat = np.mat((numVal, 30))
    for i in range(numVal):
        random.shuffle(indexList)
        trainX = []
        trainY = []
        testX = []
        testY = []
        for j in range(m):
            if j < m * 0.9:
                trainX.append(dataArr[indexList[j]])
                trainY.append(labelArr[indexList[j]])
            else:
                testX.append(dataArr[indexList[j]])
                testY.append(dataArr[indexList[j]])

        dataMat = np.mat(trainX)
        labelMat = np.mat(trainY)
        xMeans = np.mean(dataMat, axis=0)
        yMean = np.mean(labelMat)
        xVars = np.var(dataMat, axis=0)
        xNew = (dataMat - xMeans) / xVars
        yNew = labelMat - yMean
        ws = ridgeRegression(xNew, yNew)
        testXMat = np.mat(testX)
        # testYMat = np.mat(testY)
        testXNew = (testXMat - xMeans) / xVars
        preY = testXMat * ws.T + float(yMean)



if __name__ == '__main__':
    lgX1, lgY1 = loadData('./lego/lego8288.html', 2006, 800, 49.99)
    lgX2, lgY2 = loadData('./lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    lgX3, lgY3 = loadData('./lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    lgX4, lgY4 = loadData('./lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    lgX5, lgY5 = loadData('./lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    lgX6, lgY6 = loadData('./lego/lego10196.html', 2009, 3263, 249.99)
    lgX = np.concatenate((lgX1, lgX2, lgX3, lgX4, lgX5, lgX6), axis=0)
    lgY = np.concatenate((lgY1, lgY2, lgY3, lgY4, lgY5, lgY6), axis=0)
    # dataMat, labelMat = preprocessing(lgX, lgY)
    # dataMat = np.mat(lgX)
    # labelMat = np.mat(lgY)
    # ws = linearRegression(dataMat, labelMat)
    # print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % \
    #       (ws[0], ws[1], ws[2], ws[3], ws[4]))
    crossValidation(lgX, lgY, 10)
