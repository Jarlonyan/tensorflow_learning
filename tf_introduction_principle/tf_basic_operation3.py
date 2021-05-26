#coding=utf-8
import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

#1. tf.linag.diag_part
def tf_diag_part(sess):
    logits = tf.constant([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6]])
    sfx_prob = tf.linalg.diag_part(logits)

def main():
    with tf.Session() as sess:
        tf_diag_part(sess)


if __name__ == '__main__':
    main()


