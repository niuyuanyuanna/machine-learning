import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = "pool_3/_reshape:0"
JPEG_DATA_TENSOR_NAME = "DecodeJpeg/contents:0"

MODEL_DIR = "F:/TensorFlow on Android/TSMigration/inception_dec_2015"
MODEL_FILE = "tensorflow_inception_graph.pb"
CACHE_DIR = "F:/TensorFlow on Android/TSMigration/tmp"
INPUT_DATA = "F:/TensorFlow on Android/TSMigration/flower_photos"

# 验证集和测试集的百分比
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


def creat_image_list(testing_persentage, validation_persentage):
    # result为字典，存储所有图片，key：label_name，value：字典，存储图片名称、数据集属性等
    result = {}
    # 获取当前目录的所有子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        # 得到的第一个为当前目录，跳过
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)  # 文件名，如“daisy”
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, "*." + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 目录名为类的名称，如daisy
        label_name = dir_name.lower()

        # 初始化当前类的训练集、测试集、验证集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # base_name为图片的名字,如：‘5673551_01d1ea993e_n.jpg’
            chance = np.random.randint(100)  # 随机将图片分配给三个数据集
            if chance < validation_persentage:
                validation_images.append(base_name)
            elif chance < (testing_persentage + validation_persentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别数据放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }
    return result


# image_lists所有图片信息
# image_dir根目录
# label_name类别名称
# index图片编号
# category图片所在集合为训练集、测试集还是验证集
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # 获取label_name这个类别的所有图片信息
    category_list = label_lists[category]  # 获取指定类中三个数据集中指定的数据集，如：training中的图片信息
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]  # 获取指定编号的图片的文件名
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)  # 最终的path为image_dir/sub_dir/base_name
    return full_path


# 通过所有图片信息、所属类别（daisy）、图片编号、所属数据集（training）获取通过模型之后的特征向量地址
# 参数比get_image_path方法少了image_dir，替换成CACHE_DIR为保存模型变量的临时文件存储文件夹
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'



# 将当前图片作为输入计算bottleneck_tensor的值，返回值为特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottenneck_values = np.squeeze(bottleneck_values)  # 将四位数组压缩为一个特征向量
    return bottenneck_values


# 获取经过模型之后的特征向量字符串
def get_or_creat_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]  # 获取label_name这个类别的所有图片信息
    sub_dir = label_lists['dir']  # 获取daisy类别的training图片信息
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)  # 创建CHACHE_DIR/sub_dir文件夹
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    # 获取一张图片对应的特征向量文件路径
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 如果没有保存过特征向量文件就计算特征向量并且保存，否则读取文件中的特征向量string值
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()  # 图片内容
        bottelneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor, bottleneck_tensor)  # 得到特征向量
        bottleneck_string = ','.join(str(x) for x in bottelneck_values)  # 将特征向量转化为string存储到.txt文件中
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottelneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottelneck_values


# 随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机选择一个类别和图片编号加入当前的训练数据
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


# 获取全部测试数据
def get_test_bottenecks(sess, image_lists, n_classes, jpeg_data_tensor, botteneck_tensor):
    bottenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有类别以及每个类别中的测试图片
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
    image_lists = creat_image_list(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)  # 读取所有图片
    n_classes = len(image_lists.keys())  # 图片类别
    # 读取训练好的模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载读取的模型，返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的输入为经过模型前向传播后到达瓶颈层时的节点取值，相当于新的特征输入
    bottleneck_input = tf.placeholder(tf.float32,
                                      [None, BOTTLENECK_TENSOR_SIZE])
    ground_truth_input = tf.placeholder(tf.float32,
                                        [None, n_classes], name='GroundTruthInput')

    # 定义最后一层全连接层解决图片分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal(
            [BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(
            tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(STEPS):
            # 随机获取Batch个数据对作为训练数据并进行最后一层的训练
            train_bottenecks, train_ground_truth = \
                get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor,
                                              bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottenecks, ground_truth_input: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                # 随机获取Batch个数据对最为验证集并计算正确率
                validation_bottlenecks, validation_ground_truth = \
                    get_random_cached_bottlenecks(
                        sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={bottleneck_input: validation_bottlenecks,
                                                          ground_truth_input: validation_ground_truth})
                print('Step %d : Validation accuracy on random sampled %d examples = %.lf%%' %
                      (i, BATCH, validation_accuracy * 100))

            # 获取全部测试数据，并计算正确率
            test_bottlenecks, test_ground_truth = get_test_bottenecks(
                sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
            test_accuracy = sess.run(evaluation_step,
                                     feed_dict={bottleneck_input: test_bottlenecks,
                                                ground_truth_input: test_ground_truth})
            print('Final test accuracy = %.lf%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
