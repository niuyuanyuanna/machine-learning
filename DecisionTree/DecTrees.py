from math import log
import operator
import pickle


def creat_data_set():
    data_set = [[0, 0, 0, 0, 'no'],
                [0, 0, 0, 1, 'no'],
                [0, 1, 0, 1, 'yes'],
                [0, 1, 1, 0, 'yes'],
                [0, 0, 0, 0, 'no'],
                [1, 0, 0, 0, 'no'],
                [1, 0, 0, 1, 'no'],
                [1, 1, 1, 1, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [1, 0, 1, 2, 'yes'],
                [2, 0, 1, 2, 'yes'],
                [2, 0, 1, 1, 'yes'],
                [2, 1, 0, 1, 'yes'],
                [2, 1, 0, 2, 'yes'],
                [2, 0, 0, 0, 'no']]
    features = ['age', 'work', 'house', 'credit']
    return data_set, features


def cal_entropy(data_set):
    num_simple = len(data_set)
    label_count = {}
    for feature_vector in data_set:
        current_label = feature_vector[-1]
        if current_label not in label_count.keys():
            label_count[current_label] = 0
        label_count[current_label] += 1
    entropy = 0.0
    for label in label_count:
        prob = float(label_count[label]) / num_simple
        entropy -= prob * log(prob, 2)
    return entropy


def splite_data_set(data_set, feature, value):
    sub_data_set = []
    for featrue_vector in data_set:
        if featrue_vector[feature] == value:
            reduce_featrue_vec = featrue_vector[:feature]
            reduce_featrue_vec.extend(featrue_vector[feature + 1:])
            sub_data_set.append(reduce_featrue_vec)
    return sub_data_set


def choose_best_feature(data_set):
    num_feature = len(data_set[0]) - 1
    base_entropy = cal_entropy(data_set)
    best_info_gain = 0.0
    best_featrue = -1
    for i in range(num_feature):
        feature_list = [sample[i] for sample in data_set]
        class_value = set(feature_list)
        conditional_entropy = 0.0
        for value in class_value:
            sub_data_set = splite_data_set(data_set, i, value)
            probability = len(sub_data_set) / float(len(data_set))
            conditional_entropy += probability * cal_entropy(sub_data_set)
        info_gain = base_entropy - conditional_entropy
        print('the %d featrue information gain:%.3f' % (i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_featrue = i
    print('the best featrue index:%d' % best_featrue)
    return best_featrue


def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def creat_tree(data_set, features, feat_feature):
    class_list = [sample[-1] for sample in data_set]
    # 如果data_set中所有样本的class_list的value都相同，则返回value
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果data_set中的feature只剩下一个，则返回类别占比更多的value
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)

    best_feature_index = choose_best_feature(data_set)
    best_feature = features[best_feature_index]
    feat_feature.append(best_feature)
    tree = {best_feature: {}}
    del (features[best_feature_index])
    feat_values = [sample[best_feature_index] for sample in data_set]
    unique_value = set(feat_values)
    for value in unique_value:
        tree[best_feature][value] = creat_tree(
            splite_data_set(data_set, best_feature_index, value), features, feat_feature)
    return tree


def store_tree(tree, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)


def load_tree(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def classify(tree, features, test_vector):
    first_str = next(iter(tree))
    second_dict = tree[first_str]
    feat_index = features.index(first_str)
    for key in second_dict.keys():
        if test_vector[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], features, test_vector)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    data_set, features = creat_data_set()
    feat_feature = []
    tree = creat_tree(data_set, features, feat_feature)
    # 训练之后feat_feature是一个从根节点特征开始记录的list，因为在训练过程中features会被删除掉
    print(tree)
    store_tree(tree, 'classifierStorage.txt')
    result = classify(load_tree('classifierStorage.txt'), feat_feature, test_vector=[0, 1])  # test_vector这里代表没有房子但有车子
    print(result)
