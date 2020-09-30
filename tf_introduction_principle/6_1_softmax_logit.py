#coding=utf-8
import tensorflow as tf


labels = [[0,0,1], [0,1,0]]  #one-hot
logits = [[2,0.5,6], [0.1,0,3]]
logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

res1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
res2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits_scaled)
res3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)

with tf.Session() as sess:
    print 'res1=',sess.run(res1)
    print 'res2=',sess.run(res2)
    print 'res3=',sess.run(res3)


