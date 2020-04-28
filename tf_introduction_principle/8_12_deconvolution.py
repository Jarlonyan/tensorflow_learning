#coding=utf-8
#通过对模拟数据进行卷积与反卷积的操作，来比较卷积与反卷积中padding在SAME、VALID下的变化

import numpy as np
import tensorflow as tf
img = tf.Variable(tf.constant(1.0, shape=[1,4,4,1]))
filter = tf.Variable(tf.constant([1.0, 0, -1, -2], shape=[2,2,1,1]))
conv = tf.nn.conv2d(img, filter, stirdes=[1,2,2,1], padding='VALID')
cons = tf.nn.conv2d(img, filter, strides=[1,2,2,1], padding='SAME')


