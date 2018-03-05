from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO

import pandas as pd
import pydotplus


def code_data(lenses, feature_labels):
    lenses_target = []
    for sample in lenses:
        lenses_target.append(sample[-1])
    lenses_dict = {}
    lenses_list = []
    for label in feature_labels:
        for sample in lenses:
            lenses_list.append(sample[feature_labels.index(label)])
        lenses_dict[label] = lenses_list
        lenses_list = []
    print(lenses_dict)
    lenses_pd = pd.DataFrame(lenses_dict)
    print(lenses_pd)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    return lenses_pd, lenses_target


def visual_tree(lenses_pd, clf):
    dot_data = StringIO()

    class_value_list = []
    for class_value in clf.classes_:
        class_value_list.append(class_value.decode('UTF-8'))

    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=lenses_pd.keys(),
                         class_names=class_value_list,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf')


def predict(clf):
    print(clf.predict([[1, 1, 1, 0]]))


if __name__ == '__main__':
    with open('lenses.txt', 'rb') as f:
        lenses = [inst.strip().split(b'\t') for inst in f.readlines()]
    print(lenses)
    feature_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    clf = tree.DecisionTreeClassifier(max_depth=4)
    # fit() 无法接收string类型数据，在fit之前需要对数据进行编码
    lenses_pd, target = code_data(lenses, feature_labels)
    clf = clf.fit(lenses_pd.values.tolist(), target)

    visual_tree(lenses_pd, clf)
    predict(clf)

