import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lns
import KNN



def file_matrix():
    with open('datingTestSet.txt', 'rb') as f:
        array_lines = f.readlines()
        num_lines = len(array_lines)
        return_mat = np.zeros((num_lines, 3))
        class_label_vector = []
        index = 0
        for line in array_lines:
            line = line.strip()
            list_from_line = line.split(b'\t')
            return_mat[index, :] = list_from_line[0:3]
            if list_from_line[-1] == b'didntLike':
                class_label_vector.append(1)
            elif list_from_line[-1] == b'smallDoses':
                class_label_vector.append(2)
            elif list_from_line[-1] == b'largeDoses':
                class_label_vector.append(3)
            index += 1
        return return_mat, class_label_vector


def show_data(dating_data_mat, dating_labels):
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(13, 8))
    labels_colors = []
    for i in dating_labels:
        if i == 1:
            labels_colors.append('black')
        if i == 2:
            labels_colors.append('orange')
        if i == 3:
            labels_colors.append('red')
    axs[0][0].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 1], color=labels_colors, s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'time flying miles VS play game')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'flying miles')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'play game')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=dating_data_mat[:, 0], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'time flying miles VS ice cream')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'flying miles')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'ice cream')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=dating_data_mat[:, 1], y=dating_data_mat[:, 2], color=labels_colors, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'time play game VS ice cream')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'play game')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'ice cream')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    didntLike = lns.Line2D([], [], color='black', marker='.',
                           markersize=6, label='didntLike')
    smallDoses = lns.Line2D([], [], color='orange', marker='.',
                            markersize=6, label='smallDoses')
    largeDoses = lns.Line2D([], [], color='red', marker='.',
                            markersize=6, label='largeDoses')
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


def auto_norm(dating_data_mat):
    dataset_size = dating_data_mat.shape[0]
    min_value_vecor = dating_data_mat.min(0)
    max_value_vector = dating_data_mat.max(0)
    old_dev_min = dating_data_mat - np.tile(min_value_vecor, (dataset_size, 1))
    max_dev_min = np.tile(max_value_vector - min_value_vecor, (dataset_size, 1))
    return old_dev_min / max_dev_min, min_value_vecor, max_value_vector


def dating_class_train(dating_data_mat, dating_labels, testing_persentage):
    dataset_size = dating_data_mat.shape[0]
    testing_count = int(dataset_size * testing_persentage / 100)
    erro_count = 0.0
    for i in range(testing_count):
        classify_result = KNN.classify_KNN(
            dating_data_mat[i, :],
            dating_data_mat[testing_count:dataset_size, :],
            dating_labels[testing_count:dataset_size], 4)
        if classify_result != dating_labels[i]:
            erro_count += 1
            print('output erro!')
    print('final erro rate: %.2f %%' % (erro_count / testing_count * 100))


def classify_person():
    result_list = ['讨厌', '有些喜欢', '非常喜欢']

    fly_miles = float(input("flying miles parameter:"))
    play_game = float(input("play game parameter:"))
    ice_cream = float(input("ice cream parameter:"))

    in_arr = np.array([fly_miles, play_game, ice_cream])
    dating_data_mat, dating_labels = file_matrix()
    new_dating_data_mat, min_value_data, max_value_data = auto_norm(dating_data_mat)
    norm_in_arr = (in_arr - min_value_data) / max_value_data

    classifier_result = KNN.classify_KNN(norm_in_arr, new_dating_data_mat, dating_labels, 4)

    # 打印结果
    print("你可能%s这个人" % (result_list[classifier_result - 1]))


if __name__ == '__main__':
    classify_person()
