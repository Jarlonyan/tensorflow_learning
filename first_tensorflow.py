#coding=utf-8

import tensorflow as tf

def main():
    w1 = tf.Variable(tf.random_normal((2,3), stddev=1, seed=1))  # 2*3 矩阵
    w2 = tf.Variable(tf.random_normal((3,1), stddev=1, seed=1))  # 3*1 矩阵
    x = tf.constant([[0.7, 0.9], #模拟输入向量，一行一个样本
                     [0.1, 0.2],
                     [0.3, 0.5]])
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    with tf.Session() as sess:
        sess.run(w1.initializer)  # 初始化w1
        sess.run(w2.initializer)  # 初始化w2
        ret = sess.run(y)
        print ret

if __name__ == "__main__":
    main()
