import numpy as np


def load_data():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vector


def creat_vocab_list(posting_list):
    vocab_set = set([])
    for document in posting_list:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def creat_words_vector(vocab_list, posting_list):
    train_mat = []
    for input_list in posting_list:
        word_vector = [0] * len(vocab_list)
        for word in input_list:
            if word in vocab_list:
                word_vector[vocab_list.index(word)] = 1
            else:
                print('the word :%s is not in my vocabulary' % word)
        train_mat.append(word_vector)
    return train_mat


def naive_bayes_train(train_mat, class_vector):
    [num_docs, num_words] = np.shape(train_mat)
    pAbusive = sum(class_vector) / float(num_docs)

    p0Num = np.ones(num_words)   # 拉普拉斯平滑
    p1Num = np.ones(num_words)   # 分子初始化为1
    p0Denom = 2.0
    p1Denom = 2.0  # 分母初始化为2
    for i in range(num_docs):
        if class_vector[i] == 1:  # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += train_mat[i]
            p1Denom += sum(train_mat[i])
        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += train_mat[i]
            p0Denom += sum(train_mat[i])
    p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifierNB(test_mat, p0Vect, p1Vect, pAbusive):
    result_list = []
    for test_list in test_mat:
        # p1 = reduce(lambda x, y: x * y, test_list * p1Vect) * pAbusive  # 对应元素相乘
        # p0 = reduce(lambda x, y: x * y, test_list * p0Vect) * (1.0 - pAbusive)
        p1 = sum(test_list * p1Vect) + np.log(pAbusive)        # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
        p0 = sum(test_list * p0Vect) + np.log(1.0 - pAbusive)

        print('p0:', p0)
        print('p1:', p1)
        if p1 > p0:
            result_list.append(1)
        else:
            result_list.append(0)
    return result_list


if __name__ == '__main__':
    posting_list, class_vector = load_data()
    print('原始数据集:\n', posting_list)
    # vocab_list将词条向量化，一个单词在词汇表中出现过一次，相应位置记作1，如果没有出现，在相应位置记作0
    vocab_list = creat_vocab_list(posting_list)
    print('词汇表，单词只出现一次:\n', vocab_list)

    train_mat = creat_words_vector(vocab_list, posting_list)
    print('所有词条向量组成的列表:\n', train_mat)

    p0Vect, p1Vect, pAbusive = naive_bayes_train(train_mat, class_vector)
    print('朴素贝叶斯计算每个词不属于侮辱性质的概率p0Vect:\n', p0Vect)
    print('朴素贝叶斯计算每个词属于侮辱性质的概率p0Vect:\n', p1Vect)
    print('侮辱性文档占比pAbusive:', pAbusive)

    test_entry = [['love', 'my', 'dalmation'], ['stupid', 'garbage']]
    test_mat = creat_words_vector(vocab_list, test_entry)
    print('test_mat:\n', test_mat)

    result_list = classifierNB(test_mat, p0Vect, p1Vect, pAbusive)
    print('result_list:\n', result_list)
