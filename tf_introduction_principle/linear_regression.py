#coding=utf-8
'''
3-1: 实现线性回归模型
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]


train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape)*0.3

plt.plot(train_X, train_Y, 'ro', label='raw data')
plt.legend()
plt.show()

print train_X
print train_Y

X = tf.placeholder(dtype="float32", name="X")
Y = tf.placeholder(dtype="float32", name="Y")

W = tf.Variable(tf.random_normal([1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")

y_head = tf.multiply(X, W) + b






