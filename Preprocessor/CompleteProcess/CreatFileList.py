import os
import tensorflow as tf
import glob
import numpy as np
from PIL import Image

MODEL_DIR = "F:/TensorFlow on Android/TSMigration/inception_dec_2015"
MODEL_FILE = "tensorflow_inception_graph.pb"
CACHE_DIR = "F:/TensorFlow on Android/TSMigration/tmp"
INPUT_DATA = "F:/TensorFlow on Android/TSMigration/flower_photos"
INDEX = 0
BASERECORDNUM = 1000


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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

def write_records_file(dataset, record_location):
        writer = None
        current_index = 0
        for breed, images_filenames in dataset.items():

            for image_filename in images_filenames:

                if current_index % 100 == 0:

                    if writer:

                        writer.close()

                    record_filename = "{record_location}-{current_index}.tfrecords".format(

                        record_location=record_location,

                        current_index=current_index)

                    writer = tf.python_io.TFRecordWriter(record_filename)

                    print record_filename + "------------------------------------------------------"

                current_index += 1

                image_file = tf.read_file(image_filename)

                try:

                    image = tf.image.decode_jpeg(image_file)

                except:

                    print(image_filename)

                    continue

                grayscale_image = tf.image.rgb_to_grayscale(image)

                resized_image = tf.image.resize_images(grayscale_image, [250, 151])

                image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

                image_label = breed.encode("utf-8")

                example = tf.train.Example(features=tf.train.Features(feature={

                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),

                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))

                }))

                writer.write(example.SerializeToString())

        writer.close()
