import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def creat_img_vector(filename):
    return_vector = np.zeros((1, 1024))
    with open(filename, 'rb') as f:
        for i in range(32):
            line_str = f.readline()
            for j in range(32):
                return_vector[0, 32 * i + j] = int(line_str[j])
        return return_vector


def hand_writing_test():
    hw_labels = []
    training_file_list = listdir('trainingDigits')
    m = len(training_file_list)
    training_data_mat = np.zeros((m, 1024))
    index = 0

    for training_file_name in training_file_list:
        class_label = int(training_file_name.split('_')[0])
        hw_labels.append(class_label)
        training_data_mat[index, :] = creat_img_vector('trainingDigits/%s' % (training_file_name))
        index += 1
    neigh = KNN(n_neighbors=3, algorithm='auto')
    neigh.fit(training_data_mat, hw_labels)
    test_file_list = listdir('testDigits')
    n = len(test_file_list)
    error_count = 0
    for test_file_name in test_file_list:
        class_label = int(test_file_name.split('_')[0])
        testing_vector = creat_img_vector('testDigits/%s' % (test_file_name))
        classifyer_result = neigh.predict(testing_vector)
        if (classifyer_result != class_label):
            error_count += 1
    print('testing error %d data, error rate: %.2f%%' % (error_count, error_count / n * 100))


if __name__ == '__main__':
    hand_writing_test()
