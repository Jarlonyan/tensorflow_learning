
import tensorflow as tf

x= [5,3,2.0]

logits = tf.nn.softmax(x)

with tf.Session() as sess:
    ret = sess.run(logits)
    print ret


