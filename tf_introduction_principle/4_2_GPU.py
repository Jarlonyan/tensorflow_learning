
import tensorflow as tf

with tf.Session() as sess:
    with tf.device("/gpu:1"):
        a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add = tf.add(a, b)
    print sess.run(add, feed_dict={a:10, b:12}))

