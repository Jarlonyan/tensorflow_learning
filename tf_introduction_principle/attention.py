
import tensorflow as tf
import copy

Q = tf.Variable(tf.constant(\
     [[1.1, 4.2, 1.3, 1.4], \
      [0.1, 2.3, 2.5, 2.7], \
      [3.1, 3.4, 3.8, 3.0]],\
    dtype=tf.float32), name='query')

K = tf.Variable(tf.constant(\
    [[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\
    dtype=tf.float32), name='key')

V = tf.Variable(tf.constant( \
    [[1.1, 4.2, 1.3, 1.4], \
     [0.1, 2.3, 2.5, 2.7], \
     [3.1, 3.4, 3.8, 3.0]],\
    dtype=tf.float32), name='value')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    z = tf.matmul(Q, K, transpose_b=True)
    scaled_z = tf.multiply(z, 1/tf.sqrt(4.0))
    softmax_z = tf.nn.softmax(scaled_z, dim=1)
    print sess.run(tf.tensordot(softmax_z, V, [[1],[0]])))
    print sess.run(tf.tensordot(softmax_z, V, 1))
    print softmax_z.shape
    print sess.run(tf.matmul(softmax_z, V))

    #A = tf.Variable(tf.constant(\
    #    [[2., 2.], [3,3]], dtype=tf.float32))

    #A = tf.constant([[2.0,3.1], [5.0,6.0]])
    #B=tf.nn.softmax(A, dim=1)
    #print sess.run(B)

