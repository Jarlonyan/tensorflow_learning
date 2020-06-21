
import tensorflow as tf
import numpy as np

embed_table = tf.constant([[0,    0,    0   , 0   , 0],
                           [0.11, 0.12, 0.13, 0.14, 0.15],
                           [0.21, 0.22, 0.23, 0.24, 0.25],
                           [0.31, 0.32, 0.33, 0.34, 0.35],
                           [0.41, 0.42, 0.43, 0.44, 0.45]], dtype=tf.float32)

input_batch = tf.constant([2, 3, 0])
ue_raw = tf.nn.embedding_lookup(embed_table, input_batch)

raw_embedding_sum = tf.reduce_sum(ue_raw, axis=1)
hit_count = tf.reduce_sum(tf.cast(raw_embedding_sum != 0, tf.float32))
tf.summary.histogram('hit_count', hit_count)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(ue_raw)
    print sess.run(raw_embedding_sum)
    print sess.run(hit_count)

'''
ue_raw = fc_ue.get_tensor()
tf.summary.histogram(ue_name + "_raw_tensor", ue_raw)

raw_embedding_sum = tf.reduce_sum(ue_raw, axis=1)
hit_count = tf.reduce_sum(tf.cast(raw_embedding_sum != 0, tf.float32))
tf.summary.histogram('hit_count', hit_count)
'''









