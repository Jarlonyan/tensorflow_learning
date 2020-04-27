#coding=utf-8
import sys
sys.path.append('./CIFAR_TensorFlow/')
import cifar10_input
import tensorflow as tf
import pylab

batch_size = 12
data_dir = 'data/cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

