
import tensorflow as tf
import numpy as np


embed_table = tf.constant([[0,    0,    0   , 0   , 0,   0],
                           [-0.11, 0.12, 0.13, 0.14, -0.15, 0.16],
                           [-0.21, 0.22, 0.23, 0.24, -0.25, 0.26],
                           [0.31, 0.32, 0.33, 0.34, -0.35, 0.36],
                           [0.41, 0.42, 0.43, 0.44, -0.45, 0.46]], dtype=tf.float32)

input_batch = tf.constant([2, 4, 0, 3])
ue_raw = tf.identity(tf.nn.embedding_lookup(embed_table, input_batch))

raw_embedding_sum = tf.reduce_sum(ue_raw, axis=1)
non_zero = tf.count_nonzero(raw_embedding_sum)
batch_size = tf.reshape(tf.shape(raw_embedding_sum), [])
hit_count_ratio = tf.divide(tf.cast(non_zero,tf.float32),  tf.cast(batch_size,tf.float32))

bool_select_ue = tf.equal(tf.reduce_sum(tf.abs(ue_raw), axis=1), 0.0)
batch_size = ue_raw.shape.as_list()[0]
ue_dim = ue_raw.shape.as_list()[1]
default_ue = tf.Variable(tf.random_normal([ue_dim], name='default_ue'))
#default_ue = tf.tile(tf.expand_dims(default_ue, axis=0), [2,1])

import pdb; pdb.set_trace()
#default_ue = tf.expand_dims(default_ue, axis=0)
#default_ue = tf.tile(default_ue, [2,1])
default_ue = tf.tile(default_ue, [3, 0])

res = tf.where(bool_select_ue, default_ue, ue_raw)

toutiao_dense_uid = ue_raw
is_training = 1.0
bs = tf.shape(toutiao_dense_uid)[0]
rand = tf.random_uniform(shape=[bs])
toutiao_dense_uid = tf.cond(tf.equal(is_training, 1.0),
                            lambda: tf.where(tf.less(rand, tf.fill([bs], 0.01)),
                                             tf.zeros_like(toutiao_dense_uid),
                                             tf.where(tf.less(rand, tf.fill([bs], 0.01 + 0.01)),
                                                              tf.random_uniform(shape=tf.shape(toutiao_dense_uid),
                                                              minval=-0.01,
                                                              maxval=0.01),
                                                      toutiao_dense_uid)),
                            lambda: toutiao_dense_uid)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #print 'raw_embedding_sum=', raw_embedding_sum, sess.run(raw_embedding_sum)
    #print 'non_zero=', type(non_zero), sess.run(non_zero)
    #print 'batch_size=', type(batch_size), sess.run(batch_size)
    #print 'hit_count_ratio=', hit_count_ratio, sess.run(hit_count_ratio)
    #print sess.run(size)
    
    #print 'ue_raw=',sess.run(ue_raw)
    print 'res=',sess.run(res)
    #print 'toutiao_dense_uid=', sess.run(toutiao_dense_uid)


    print "default_ue=", sess.run(default_ue)
    #print "ue=", sess.run(b)

    #print 'non_zero=', type(non_zero), sess.run(non_zero)
    #print "abs_ue=", sess.run(abs_ue)
    #print "default_emb=", sess.run(default_ue)

#default_emb = tf.Variable(tf.random_normal([input_dim, int(batch_size)], name='w'))
#print sess.run(ue_raw)
    

'''
ue_raw = fc_ue.get_tensor()
tf.summary.histogram(ue_name + "_raw_tensor", ue_raw)

raw_embedding_sum = tf.reduce_sum(ue_raw, axis=1)
hit_count = tf.reduce_sum(tf.cast(raw_embedding_sum != 0, tf.float32))
tf.summary.histogram('hit_count', hit_count)
'''









