import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"
JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"

MODEL_FILE = 'tensorflow_inception_graph.pb'
CACHE_DIR = "F:/project/PyCharm/homework/temp"
INPUT_DATA = "C:/Users/liuyuan/Desktop/dataset"

LEARNING_RATE = 0.01
STEPS = 1000
BATCH = 5


def creat_image_list():
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        label_name = dir_name.lower()

        training_images = []
        testing_images = []
        test_index = []

        for i in range(50):
            chance = np.random.randint(len(file_list))
            test_index.append(chance)
            base_name = os.path.basename(file_list[chance])
            testing_images.append(base_name)

        for index, file_name in enumerate(file_list):
            base_name = os.path.basename(file_name)
            if index in test_index:
                continue
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
        }
    return result


def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # 获取label_name这个类别的所有图片信息
    category_list = label_lists[category]  # 获取指定类中三个数据集中指定的数据集，如：training中的图片信息
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]  # 获取指定编号的图片的文件名
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)  # 最终的path为image_dir/sub_dir/base_name
    return full_path


def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottenneck_values = np.squeeze(bottleneck_values)
    return bottenneck_values


def get_or_creat_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottelneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottelneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottelneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottelneck_values


def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        botteneck = get_or_creat_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottenecks.append(botteneck)
        ground_truths.append(ground_truth)

    return bottenecks, ground_truths


def get_test_bottenecks(sess, image_lists, n_classes, jpeg_data_tensor, botteneck_tensor):
    bottenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            botteneck = get_or_creat_bottleneck(
                sess, image_lists, label_name, index, category, jpeg_data_tensor, botteneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottenecks.append(botteneck)
            ground_truths.append(ground_truth)
    return bottenecks, ground_truths


def main(_):
    image_lists = creat_image_list()
    n_classes = len(image_lists.keys())
    print('class:', n_classes)

    with gfile.FastGFile(MODEL_FILE, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    bottleneck_input = tf.placeholder(tf.float32,
                                      [None, BOTTLENECK_TENSOR_SIZE])
    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, n_classes], name='GroundTruthInput')

    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(
            tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(STEPS + 1):
            train_bottenecks, train_ground_truth = \
                get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor,
                                              bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottenecks, ground_truth_input: train_ground_truth})

            test_bottlenecks, test_ground_truth = get_test_bottenecks(
                sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
            if i % 100 == 0:
                test_accuracy = sess.run(evaluation_step,
                                         feed_dict={bottleneck_input: test_bottlenecks,
                                                    ground_truth_input: test_ground_truth})
                print('After %d training steps,  test accuracy = %.lf%%' % (i, test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
