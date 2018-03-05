import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 滑动平均模型衰减率
REGULARIZATION_RATE = 0.0001  # 正则化参数，lambda
TRAINING_STEP = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "F://TensorFlow on Android/MNIST"
MODEL_NAME = "model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.NUM_CHANNELS],
                       name="x-input")





    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name="y-input")

    regularizes = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 前向传播计算y值
    y = mnist_inference.inference(x, regularizes)

    # 滑动平均
    global_step = tf.Variable(0, trainable=False)  # 当前迭代轮数
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # arg_max(y_, 1)是为了获取y_中1的位置，使用提供的函数计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

    train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step)  # 自动更新loss以及global_step
    # 更新每个参数的滑动平均
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s),"
                      "loss on training batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)


def main(argv=None):
    tf.get_default_graph().device("/gpu:0")
    mnist = input_data.read_data_sets("F:/TensorFlow on Android", one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()
