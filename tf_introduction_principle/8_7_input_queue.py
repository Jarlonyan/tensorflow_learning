#coding=utf-8
import tensorflow as tf

queue = tf.FIFOQueue(100, 'float')
c = tf.Variable(0.0)
op = tf.assign_add(c, tf.constant(1.0))
enqueue_op = queue.enqueue(c)

qr = tf.train.QueueRunner(queue, enqueue_ops=[op, enqueue_op])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    for i in range(0,10):
        print sess.run(queue.dequeue())

    coord.request_stop()

