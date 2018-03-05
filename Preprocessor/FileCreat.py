import tensorflow as tf


# TFRecord帮助文件，生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


num_shared = 2  # 定义总共写入多少个文件
instances_per_shard = 2  # 定义每个文件的数据个数
for i in range(num_shared):
    filename = 'path/data.tfrecords-%.5d-of-%.5d' % (i, num_shared)
    writer = tf.python_io.TFRecordWriter(filename)

    for j in range(instances_per_shard):
        example = tf.train.Example(feature=tf.train.Feature(
            feature={
                'i': _int64_feature(i),
                'j': _int64_feature(j)
            }
        ))
        writer.write(example.SerializeToString())
    writer.close()
