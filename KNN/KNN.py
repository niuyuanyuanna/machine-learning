from numpy import *
import operator


def creat_data_set():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify_KNN(input_x, dataset, labels, k):
    dataset_size = dataset.shape[0]  # 4
    diff_mat = tile(input_x, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_dist_index = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_index[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    # operator.itemgetter(1)表示整理的参数按照iteritems的第二项排序
    # 此处class_count.items() = [('B', 2), ('A', 1)]
    # reverse 表示Decending的方式排序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


group, labels = creat_data_set()
sorted_count = classify_KNN([0, 0], group, labels, 3)
print(sorted_count)
