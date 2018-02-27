import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets(
    'MNIST_data', dtype=tf.uint8, one_hot=True)

images = mnist.train.images
labels = mnist.train.labels
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels = images.shape[1]

num_examples = mnist.train.num_examples
num_test_examples = mnist.test.num_examples

filename = "data/train2.tfrecords"
filename_test = "data/test2.tfrecords"

writer = tf.python_io.TFRecordWriter(filename)
for index in range(1000):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
      'pixels': _int64_feature(pixels),
      'label': _int64_feature(np.argmax(labels[index])),
      'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()

writer = tf.python_io.TFRecordWriter(filename_test)
for index in range(num_test_examples):
    image_raw = images_test[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
      'pixels': _int64_feature(pixels),
      'label': _int64_feature(np.argmax(labels_test[index])),
      'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
writer.close()
