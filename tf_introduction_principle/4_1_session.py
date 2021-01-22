
import tensorflow as tf

x = tf.constant("x")
with tf.Session() as sess:
    print (sess.run(x))


