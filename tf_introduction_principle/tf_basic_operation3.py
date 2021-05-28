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

def tf_funcs(sess):
    a = tf.eye(5, 5, dtype=tf.int32)
    print("tf.eye=\n", sess.run(a))

def main():
    with tf.Session() as sess:
        tf_diag_part(sess)
        #tf_matmul(sess)
        #tf_multiply(sess)
        #tf_funcs(sess)


if __name__ == '__main__':
    main()


