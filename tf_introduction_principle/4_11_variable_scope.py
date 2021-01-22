import tensorflow as tf

tf.reset_default_graph()

with tf.variable_scope("test1", ):
    var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)


