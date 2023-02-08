#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

def print_tensor(sess, t, name, prefix=""):
    print("debug_info:"+prefix,",", name+".shape=", t.shape , ", "+name+".tensor=\n", sess.run(t))

def auc_loss(sess):
    surrogate_type = 'xent'    
    labels = tf.constant([[1.0],
                          [0],
                          [1],
                          [0]])
    logits = tf.constant([[0.8],
                          [0.5],
                          [0.6],
                          [0.7]])

    #cc = tf.tile(logits, [1, tf.shape(logits)[0]]) - tf.tile(logits, [1, tf.shape(logits)[0]])
    #print_tensor(sess, cc, "cc", "step1")

    print_tensor(sess, labels, "labels", "step1")
    print_tensor(sess, logits, "logits", "step1")
    labels = tf.reshape(labels, [-1, 1])
    logits = tf.reshape(logits, [-1, 1])
    print_tensor(sess, labels, "labels", "step2")
    print_tensor(sess, logits, "logits", "step2")


    weights = tf.ones_like(logits)
    weights = tf.reshape(weights, [-1, 1])

    #----------------------------------
    tmp_a = tf.expand_dims(logits, 0)
    tmp_b = tf.expand_dims(logits, 1)
    tmp_c = tmp_a - tmp_b

    tmp_a2 = tf.tile(tmp_a, [tf.shape(tmp_a)[1], 1, 1])
    tmp_b2 = tf.tile(tmp_b, [1, tf.shape(tmp_b)[0], 1])
    tmp_c2 = tmp_a2 - tmp_b2
    print_tensor(sess, tmp_a2, "tmp_a2", "step3")
    print_tensor(sess, tmp_b2, "tmp_b2", "step3")
    print_tensor(sess, tmp_c2, "tmp_a2-tmp_b2", "step3")

    #print_tensor(sess, tmp_a, "tmp_a", "step3")
    #print_tensor(sess, tmp_b, "tmp_b", "step3")
    print_tensor(sess, tmp_c, "tmp_a-tmp_b", "step3")
    #----------------------------------

    # create tensors of pairwise differences for logits and labels, and pairwise products of weights. These have shape [batch_size, batch_size, num_labels].
    logits_difference = tf.expand_dims(logits, 0) - tf.expand_dims(logits, 1)
    labels_difference = tf.expand_dims(labels, 0) - tf.expand_dims(labels, 1)
    weights_product = tf.expand_dims(weights, 0) * tf.expand_dims(weights, 1)

    signed_logits_difference = labels_difference * logits_difference

    print_tensor(sess, logits_difference, "logits_difference","step4")
    print_tensor(sess, signed_logits_difference, "signed_logits_difference", "step4")

    if surrogate_type == 'xent':
        raw_loss = tf.add(tf.maximum(-signed_logits_difference, 0.0), tf.log(1.0 + tf.exp(-tf.abs(signed_logits_difference))))
    elif surrogate_type == 'hinge':
        raw_loss = tf.maximum(1.0 - signed_logits_difference, 0)

    weighted_loss = weights_product * raw_loss

    # zero out entries of the loss where labels_difference zero (so loss is only computed on pairs with different labels).
    loss = tf.reduce_mean(tf.abs(labels_difference) * weighted_loss, 0) * 0.5
    loss = tf.reshape(loss, [-1])
    

#2. tf.expand_dims，对tensor插入一个维度
def tf_expand_dims(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    t1 = tf.expand_dims(a, 0)
    t2 = tf.expand_dims(a, 1)
    b = t1 - t2
    print("t1=\n", sess.run(t1))
    print("t2=\n", sess.run(t2))
    print("b=\n", sess.run(b))
    print(t1.shape)
    print(t2.shape)
    return
    t3 = tf.expand_dims(a, 2)
    t4 = tf.expand_dims(a, -1)
    print(t3.shape)
    print(t4.shape)


def main():
    with tf.Session() as sess:
        auc_loss(sess)

if __name__ == '__main__':
    main()


