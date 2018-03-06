import jieba
import os
import glob
import random
from sklearn.naive_bayes import MultinomialNB

INPUT_DATA = 'sogoSample'


def text_parse(long_str):
    word_cut = jieba.cut(long_str, cut_all=False)  # 精简模式
    list_of_words = list(word_cut)
    return list_of_words


def load_data_set():
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    train_list = []
    test_list = []
    train_class = []
    test_class = []
    doc_list = []
    for sub_dir in sub_dirs:
        # 得到的第一个为当前目录，跳过
        if is_root_dir:
            is_root_dir = False
            continue

        extension = 'txt'
        file_list = []
        dir_name = os.path.basename(sub_dir)  # 文件名，如“C000008”

        print(dir_name)
        file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        rand_index = int(random.uniform(0, len(file_list)))

        for i, file_name in enumerate(file_list):
            with open(file_name, 'r', encoding='utf-8') as f:
                word_list = text_parse(f.read())
                doc_list.append(word_list)
                if i == rand_index:
                    train_list.append(word_list)
                    train_class.append(file_name)
                else:
                    test_list.append(word_list)
                    test_class.append(file_name)

    return train_list, train_class, test_list, test_class


def cul_word_freq(train_list, ):
    all_words_dict = {}  # 统计训练集词频
    for word_list in train_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list


def creat_words_set(words_file):
    words_set = set()  # 创建set集合
    with open(words_file, 'r', encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():  # 一行一行读取
            word = line.strip()  # 去回车
            if len(word) > 0:  # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set


def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []  # 特征词列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  # feature_words的维度为1000
            break
            # 如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(
                all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


def creat_vocab_list(train_list, test_list, feature_words):
    def text_features(text, feature_words):                        #出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_list]
    test_feature_list = [text_features(text, feature_words) for text in test_list]
    return train_feature_list, test_feature_list                #返回结果


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list)
    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy

if __name__ == '__main__':
    train_list, train_class, test_list, test_class = load_data_set()
    all_words_list = cul_word_freq(train_list)

    stopwords_set = creat_words_set('./stopwords_cn.txt')
    feature_words = words_dict(all_words_list, 100, stopwords_set)
