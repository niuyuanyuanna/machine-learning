import os.path
import glob
import re
import numpy as np
import random

INPUT_DATA = 'email'


def creat_vocab_list(data_set):
    vocab_set = set([])
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def text_parse(long_str):
    list_of_words = re.split(r'\W*', long_str)
    return [tok.lower() for tok in list_of_words if len(tok) > 2]


def load_data_set():
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    doc_list = []
    class_list = []
    for sub_dir in sub_dirs:
        # 得到的第一个为当前目录，跳过
        if is_root_dir:
            is_root_dir = False
            continue

        extension = 'txt'
        file_list = []
        dir_name = os.path.basename(sub_dir)  # 文件名，如“spam”

        print(dir_name)
        file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        for file_name in file_list:
            with open(file_name, 'r') as f:
                word_list = text_parse(f.read())
                doc_list.append(word_list)
                if dir_name == 'ham':
                    class_list.append(0)
                else:
                    class_list.append(1)
    return doc_list, class_list


def creat_data_matrix(doc_list, vocab_list):
    data_mat = []
    data_bag = []
    for input_list in doc_list:
        word_vector = np.zeros(len(vocab_list))
        word_count = np.zeros(len(vocab_list))
        for word in input_list:
            if word in vocab_list:
                word_vector[vocab_list.index(word)] = 1
                word_count[vocab_list.index(word)] += 1
            else:
                print('the word :%s is not in my vocabulary' % word)
        data_mat.append(word_vector)
        data_bag.append(word_count)
    return data_mat, data_bag


def sep_train_test_data_set(data_mat, class_list):
    train_mat = []
    test_mat = []
    train_class_list=[]
    test_class_list = []
    rand_index_list = []

    for i in range(6):   # 随机选择6封邮件作为测试集
        rand_index = int(random.uniform(0, len(data_mat)))
        test_mat.append(data_mat[rand_index])
        test_class_list.append(class_list[rand_index])
        rand_index_list.append(rand_index)
    for i, doc in enumerate(data_mat):
        if i not in rand_index_list:
            train_mat.append(doc)
            train_class_list.append(class_list[i])
    return train_mat, test_mat, train_class_list, test_class_list


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


def classifierNB(test_mat, test_class_list, p0Vect, p1Vect, pAbusive):
    result_list = []

    erro_count = 0
    for index, test_list in enumerate(test_mat):
        # p1 = reduce(lambda x, y: x * y, test_list * p1Vect) * pAbusive  # 对应元素相乘
        # p0 = reduce(lambda x, y: x * y, test_list * p0Vect) * (1.0 - pAbusive)
        p1 = sum(test_list * p1Vect) + np.log(pAbusive)        # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
        p0 = sum(test_list * p0Vect) + np.log(1.0 - pAbusive)

        print('p0:', p0)
        print('p1:', p1)
        if p1 > p0:
            result = 1
        else:
            result = 0
        result_list.append(result)
        if result== test_class_list[index]:
            erro_count += 1
    print('erro rate:', erro_count/float(len(test_class_list)))
    return result_list


if __name__ == '__main__':
     doc_list, class_list = load_data_set()

     vocab_list = creat_vocab_list(doc_list)
     data_mat, _ = creat_data_matrix(doc_list, vocab_list)
     train_mat, test_mat, train_class_list, test_class_list = sep_train_test_data_set(data_mat, class_list)
     print('train_mat:\n', train_mat)
     print('test_mat:\n', test_mat)

     p0Vect, p1Vect, pAbusive = naive_bayes_train(train_mat, train_class_list)
     print('p0Vector:\n', p0Vect)
     print('p1Vector:\n', p1Vect)
     print('pAbusive:\n', pAbusive)

     result_list = classifierNB(test_mat, test_class_list, p0Vect, p1Vect, pAbusive)
     print('result_list:\n', result_list)
