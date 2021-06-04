#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

'''
1                  #维度为0的标量
[1, 2, 3]          #维度为1，包含3个元素. 注意shape=(3,)，而不是(3)。在python中tuple必须带有一个逗号
[[1, 2], [3, 4]]   #维度为2， shape=(2, 2)
'''

#1. tf.reshape对tensor按照指定形状进行变化
def tf_reshape(sess):
    a = tf.constant([1,2,3,4,5,6,7,8,9])
    t1 = tf.reshape(a, [3,3])
    t2 = tf.reshape(a, [1,-1]) #-1表示该维度下按照原有维度自动计算
    print("t1=\n", sess.run(t1))
    print("t2=\n", sess.run(t2))

#2. tf.expand_dims，对tensor插入一个维度
def tf_expand_dims(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    t1 = tf.expand_dims(a, 0)
    t2 = tf.expand_dims(a, 1)
    t3 = tf.expand_dims(a, 2)
    t4 = tf.expand_dims(a, -1)
    print(t1.shape)
    print(t2.shape)
    print(t3.shape)
    print(t4.shape)

#3. tf.squeeze删除tensor中所有大小是1的维度
def tf_squeeze(sess):
    #a = tf.Variable([[[[1],[1],[1]], [[2],[2],[2]]]])
    a = tf.Variable([1,1,1,1,1]) #shape=(bs,)
    init = tf.global_variables_initializer()
    sess.run(init)
    b = tf.squeeze(a)
    print ("a.shape=", a.shape, ", tf.rank=", sess.run(tf.rank(a)))
    print ("b.shape=", b.shape)
    print ("a=\n", sess.run(a))
    print ("tf.squeeze(a)=\n", sess.run(b))
    c = tf.expand_dims(b, 0)
    print("tf.expand_dims=\n", sess.run(c))
    d = tf.tile(c, [5,1])
    print("tf.tile=\n", sess.run(d))

#4. tf.tile是平铺的意思，用于在同一维度上的复制. tf.tile(input, multiples, name=None) input是输入，multiples是同一维度上复制的次数
def tf_tile(sess):
    a = tf.constant([[1,2], [3,4]], name="a")
    b = tf.tile(a, [3,1])
    print("a=\n", sess.run(a))
    print("tf.tile(a,[3,1])=\n", sess.run(b))

#5. tf.split对value张量沿着axis维度，按照num_or_size_splits个数切分
def tf_split(sess):
    a = tf.Variable([[[[1],[1],[1]], [[2],[2],[2]]]])
    #b = tf.split(a, num_or_size_splits ,axis=0)

def main():
    with tf.Session() as sess:
        #tf_reshape(sess)
        #tf_expand_dims(sess)
        tf_squeeze(sess)
        #tf_tile(sess)

if __name__ == '__main__':
    main()


