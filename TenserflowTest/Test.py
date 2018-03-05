import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 滑动平均模型衰减率
REGULARIZATION_RATE = 0.0001  # 正则化参数，lambda
TRAINING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99


# 计算前向传播的结果函数
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    if avg_class is None:
        # 不使用滑动平均模型
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 使用滑动平均模型
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1))
                            + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) \
               + avg_class.average(biases2)


# 使用variable_scope管理上下文，进而管理变量
def inference(input_tensor, reuse=False):
    with tf.variable_scope("layer1", reuse=reuse):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weight", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=1))
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name="x-input")
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name="y-input")

    # 生成隐藏层和输出层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 不使用滑动平均计算前向传播的结果
    # y = inference(x, None, weights1, biases1, weights2, biases2)
    y = inference(x)

    # 滑动平均衰减，variable_averages为一个类。将神经网络参数进行滑动平均
    global_step = tf.Variable(0, trainable=False)  # 当前迭代轮数
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 使用滑动平均计算前向传播的结果
    average_y = inference(x, variable_averages, True)

    # arg_max(y_, 1)是为了获取y_中1的位置，使用提供的函数计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 正则化
    regularizes = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizes(weights1) + regularizes(weights2)
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step)  # 自动更新loss以及global_step

    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 检验滑动平均模型的神经网络前向传播结果，tf.equal()会对每一个维度返回一个boolean类型的值
    # tf.cast()函数将返回的boolean值强制转换为实数型
    correct_prediction = tf.equal(tf.arg_max(average_y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEP):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s),"
                      "validation accuracy using average model is %g" % (i, validate_acc))
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            if i % 1000 == 0:
                print("After %d training step(s), "
                      "test accuracy using average model is %g" % (i, test_acc))


def main(argv=None):
    tf.get_default_graph().device("/gpu:0")
    mnist = input_data.read_data_sets("F:/TensorFlow on Android", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
