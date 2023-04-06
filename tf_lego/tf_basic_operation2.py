#coding=utf-8
import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

#1. tf.add, tf.add_n
def tf_add_x(sess):
    a = tf.constant([2,2,2,2])
    b = tf.constant([3,3,3,3])
    c = tf.add_n([a, b]) #实现列表的元素的相加，列表中可以是张量
    d = tf.add(a,b)  #张量相加
    print("c=\n", sess.run(c))
    print("d=\n", sess.run(d))

#2. tf.matmul，矩阵乘法
def tf_matmul(sess):
    u_emb = tf.constant([[2,2,2], [2,2,2]])  #bs x d
    g_emb = tf.constant([[3,3,3], [3,3,3]])  #bs x d
    logits = tf.matmul(u_emb, g_emb, transpose_b=True)
    print(logits.shape)

#3. tf.multiply，元素相乘
def tf_multiply(sess):
    a = tf.constant([[2,2,2], [2,2,2]])
    b = tf.constant([[3,3,3], [3,3,3]])
    c = tf.multiply(a, b)
    print("c=\n", sess.run(c))

#4. tf.eye，单位矩阵
def tf_eye(sess):
    a = tf.eye(5, 5, dtype=tf.int32)
    print("tf.eye=\n", sess.run(a))

#5. tf.concat沿着某一维度连接tensor
def tf_concat(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    b = tf.constant([[7,8,9], [10,11,12]])
    t = tf.concat([a,b], 1)
    print("a=\n", sess.run(a))
    print("b=\n", sess.run(b))
    print("tf.concat([a,b],0)=\n", sess.run(t))

#6. tf.nn.softmax_cross_entropy_with_logits, softmax损失函数
def tf_softmax_loss(sess):
    labels = [[0,0,1], [0,1,0]]  #one-hot
    logits = [[2,0.5,6], [0.1,0,3]]

    res1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    print("res1.shape=", res1.shape)

#7. tf.equal，对比这两个tensor相等的元素，相等则返回True，否则返回False
def tf_equal(sess):
    #普通用法
    a = tf.constant([[3,3,4,1,2,2]])
    b = tf.constant([[1,3,4,3,2,5]])
    c = tf.equal(a, b)
    print("tf.equal(a,b)=\n", sess.run(tf.cast(c, dtype=tf.int32)))

    #高级用法
    x = tf.constant([[3],[3],[4],[1],[2],[2]])
    z = tf.equal(x, tf.transpose(x)) #inbatch_softmax修复有这个代码
    print("x=\n", sess.run(x))
    print("tf.equal(x, x.T)=\n", sess.run(tf.cast(z, dtype=tf.int32)))

def main():
    with tf.Session() as sess:
        #tf_slice(sess)
        #tf_split(sess)
        #tf_concat(sess)
        #tf_transpose(sess)
        #tf_stack(sess)
        #tf_unstack(sess)
        #tf_reverse(sess)
        tf_gather(sess)
        #tf_onehot(sess)


if __name__ == '__main__':
    main()


