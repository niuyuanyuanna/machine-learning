import tensorflow as tf

# 获取文件列表
files = tf.train.match_filenames_once('path/data.tfrecords-*')
# 创建输入列表
filename_queue = tf.train.string_input_producer(files, shuffle=False, num_epochs=1)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    }
)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6):
        print(sess.run([features['i'], features['j']]))

    coord.request_stop()
    coord.join(threads)