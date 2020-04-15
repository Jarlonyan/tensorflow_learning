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
        #import pdb; pdb.set_trace()
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (i+1)*np.ones(samples_per_class)
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
    if one_hot_flag == True: #True表示one-hot编码标签, 将0转成[1 0]
        hot_list = []
        for idx in Y0[:]:
            one_hot = [int(idx==class_number) for class_number in range(num_classes)]
            hot_list.append(one_hot)
        Y0 = np.asarray(np.hstack([hot_list]), dtype=np.float32)
        Y0 = np.reshape(Y0, (-1, label_dim))
    X,Y = shuffle(X0,Y0)
    return X,Y


def main():
    np.random.seed(0)
    input_dim = 2
    label_dim = 3
    num_classes = 3
    mini_batch_size = 64
    mean = np.random.randn(input_dim)
    cov = np.eye(input_dim)
    X,Y = generate(1000, mean, cov, label_dim, [3.0, 5.0], True)


    #'''
    cates = [np.argmax(i) for i in Y]
    colors = ['r' if i==0 else 'b' if i==1 else 'y' for i in cates[:]]
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.xlabel('years')
    plt.ylabel('size')
    plt.show()
    #'''

    input_features = tf.placeholder(tf.float32, [None, input_dim]) 
    input_labels = tf.placeholder(tf.float32, [None, label_dim])
    W = tf.Variable(tf.random_normal([input_dim, label_dim], name='w'))
    b = tf.Variable(tf.zeros([label_dim]), name='b')

    output = tf.matmul(input_features, W) + b
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)
    loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

    #err = tf.reduce_mean(tf.square(input_labels - output))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(10):
            sum_err = 0.0
            for i in range(np.int32(len(Y)/mini_batch_size)):
                X_mini_batch = X[i*mini_batch_size : (i+1)*mini_batch_size, :]
                Y_mini_batch = Y[i*mini_batch_size : (i+1)*mini_batch_size, :]
                _,temp_loss = sess.run([optimizer,loss], feed_dict={input_features:X_mini_batch, input_labels:Y_mini_batch})
                
                print 'epoch=',epoch, ', loss=',temp_loss

if __name__ == "__main__":
    main()

