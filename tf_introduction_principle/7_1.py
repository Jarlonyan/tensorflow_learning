#coding=utf-8
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf

def generate(sample_size, mean, cov, label_dim, diff, one_hot_flag):
    num_classes = 1+len(diff)
    samples_per_class = int(sample_size/num_classes)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for i,d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (i+1)*np.ones(samples_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
    if one_hot_flag == True: #True表示one-hot编码标签
        class_ind = [Y0==class_number for class_number in range(num_classes)]
        Y0 = np.asarray(np.hstack(class_ind), dtype=np.float32)
    Y0 = np.reshape(Y0, (-1, label_dim))
    X,Y = shuffle(X0,Y0)
    return X,Y


def main():
    np.random.seed(0)
    input_dim = 2
    label_dim = 3
    num_classes = 2
    mean = np.random.randn(num_classes)
    cov = np.eye(num_classes)
    X,Y = generate(10, mean, cov, label_dim, [3.0, 4.0], True)
    '''
    colors = []
    for i in Y[:]:
        if i ==0:
            colors.append('r')
        elif i==1:
            colors.append('b')
        else:
            colors.append('g')
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.xlabel('years')
    plt.ylabel('size')
    plt.show()
    '''

    input_features = tf.placeholder(tf.float32, [None, input_dim]) 
    input_labels = tf.placeholder(tf.float32, [None, label_dim])
    W = tf.Variable(tf.random_normal([input_dim, label_dim], name='w'))
    b = tf.Variable(tf.zeros([label_dim]), name='b')

    output = tf.matmul(input_features, W) + b
    output = tf.nn.sigmoid(output)
    cross_entropy = -(input_labels*tf.log(output) + (1-input_labels)*tf.log(1-output))
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(output, feed_dict={input_features:X, input_labels:Y})
       

if __name__ == "__main__":
    main()

