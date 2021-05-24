#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

#1. tf.slice对tensor进行切片操作。begin表示从哪几个维度开始，size表示在input各个维度抽取的元素个数
def tf_slice(sess):
    a = tf.constant([[[1,1,1], [2,2,2]],  [[3,3,3], [4,4,4]],  [[5,5,5], [6,6,6]]])
    print("a.shape=", a.shape, ", a=", sess.run(a))
    t1 = tf.slice(a, [1,0,0], [1,1,3])
    t2 = tf.slice(a, [1,0,0], [1,2,3])
    #t3 = tf.slice(a, [1,0,0], [2,3,1])
    print("t1=",sess.run(t1))
    print("t2=",sess.run(t2))
    #print("t3=",sess.run(t3))

#2. tf.split沿着某一个维度将tensor分割成xxx
def tf_split(sess):
    a = tf.constant([[1,1,1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2,2,2,], [3,3,3,3,3,3,3,3,3,3,3,3]]) #3,12 
    t0,t1,t2 = tf.split(a, [4,6,2], 1)
    print("t0=", sess.run(t0))
    print("t1=", sess.run(t1))
    print("t2=", sess.run(t2))

#3. tf.concat沿着某一维度连接tensor
def tf_concat(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    b = tf.constant([[7,8,9], [10,11,12]])
    t = tf.concat([a,b], 0)
    print("t=", sess.run(t))

#4. tf.transpose调换tensor的维度顺序，其实就是矩阵转置
def tf_transpose(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    b = tf.transpose(a, perm=[1,0])
    print("a=", sess.run(a))
    print("a.transpose=", sess.run(b))

#5. tf.pack是沿着某个维度pack
def tf_stack(sess):
    t1 = tf.constant([1,4])
    t2 = tf.constant([2,5])
    t3 = tf.constant([3,6])
    a = tf.stack([t1, t2, t3])
    print("tf.stack([t1,t2,t3])=", sess.run(a))

def main():
    with tf.Session() as sess:
        #tf_slice(sess)
        #tf_split(sess)
        #tf_concat(sess)
        #tf_transpose(sess)
        tf_stack(sess)

if __name__ == '__main__':
    main()


