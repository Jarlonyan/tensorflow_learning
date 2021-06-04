#coding=utf-8
import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

#1. tf.linag.diag_part
def tf_diag_part(sess):
    logits = tf.constant([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6]])
    sfx_prob = tf.linalg.diag_part(logits)
    print("sfx_prob=\n", sess.run(sfx_prob))

def tf_matmul(sess):
    u_emb = tf.constant([[2,2,2], [2,2,2]])  #bs x d
    g_emb = tf.constant([[3,3,3], [3,3,3]])  #bs x d
    logits = tf.matmul(u_emb, g_emb, transpose_b=True)
    print(logits.shape)

def tf_multiply(sess):
    a = tf.constant([[2,2,2], [2,2,2]])
    b = tf.constant([[3,3,3], [3,3,3]])
    c = tf.multiply(a, b)
    print("c=\n", sess.run(c))

def tf_eye(sess):
    a = tf.eye(5, 5, dtype=tf.int32)
    print("tf.eye=\n", sess.run(a))

def tf_softmax_loss(sess):
    labels = [[0,0,1], [0,1,0]]  #one-hot
    logits = [[2,0.5,6], [0.1,0,3]]

    res1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    print("res1.shape=", res1.shape)

def tf_equal(sess):
    a = tf.constant([[1,3,4,5,6]])
    b = tf.constant([[1,3,4,3,2]])
    c = tf.equal(a, b)
    print("tf_equal=", sess.run(c))

def main():
    with tf.Session() as sess:
        #tf_diag_part(sess)
        #tf_matmul(sess)
        #tf_multiply(sess)
        #tf_eye(sess)
        #tf_softmax_loss(sess)
        tf_equal(sess)

if __name__ == '__main__':
    main()

