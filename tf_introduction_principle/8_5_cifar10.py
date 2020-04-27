#coding=utf-8
import sys
sys.path.append('./CIFAR_TensorFlow/')
import cifar10_input
import tensorflow as tf
import cifar10_input
import pylab

batch_size = 128
data_dir = 'data/cifar-10-batches-bin'
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()
    image_batch, label_batch = sess.run([images_test, labels_test])

    pylab.imshow(image_batch[0])
    pylab.show()


