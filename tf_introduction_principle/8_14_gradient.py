#coding=utf-8
#通过定义两个变量相乘，来演示使用gradients求导

import tensorflow as tf
w1 = tf.Variable([[1,2]])
w2 = tf.Variable([[3,4]])

y = tf.matmul(w1, [[9],[10]])
grads = tf.gradients(y, [w1])

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    grad_val = sess.run(grads)
    print grad_val

