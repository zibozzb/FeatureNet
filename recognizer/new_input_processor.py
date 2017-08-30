#########################################################################
'''
    Author: Zhibo Zhang
            MADlab, Mechanical Enineering, University at Buffalo
    Time: 2017.8.30
    Github: https://github.com/zibozzb
    MADlab: http://madlab.eng.buffalo.edu
    
'''
#########################################################################


import tensorflow as tf
import numpy as np
import os


def read_cifar10(data_dir, is_train, batch_size, shuffle):
    img_width = 64
    img_height = 64
    img_depth = 64
    label_bytes = 1
    image_bytes = img_width * img_height * img_depth

    with tf.name_scope('input'):

        if is_train:
            filenames = [os.path.join(data_dir, 'train%d.bin' % ii)
                         for ii in np.arange(0, 50)]
        else:
            # filenames = [os.path.join(data_dir, 'test.bin')]
            filenames = [os.path.join(data_dir, 'test%d.bin' % jj)
                         for jj in np.arange(0, 11)]

        filename_queue = tf.train.string_input_producer(filenames)

        reader = tf.FixedLengthRecordReader(label_bytes + image_bytes)

        key, value = reader.read(filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [label_bytes], [image_bytes])
        image_raw = tf.reshape(image_raw, [img_depth, img_height, img_width, 1])
        image = tf.transpose(image_raw, (1, 2, 0, 3))
        image = tf.cast(image, tf.float32)
        image = (image - 0.5) * 2

        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label],
                batch_size=batch_size,
                num_threads=16,
                capacity=2000,
                min_after_dequeue=1500)
        else:
            images, label_batch = tf.train.batch(
                [image, label],
                batch_size=batch_size,
                num_threads=16,
                capacity=2000)
        n_classes = 24
        label_batch = tf.one_hot(label_batch, depth=n_classes)

        return images, tf.reshape(label_batch, [batch_size, n_classes])
