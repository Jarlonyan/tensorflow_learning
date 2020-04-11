import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf

def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2
    samples_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci,d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)

        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
    if regression == False:
        class_ind = [Y==class_number_ for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    X,Y = shuffle(X0,Y0)
    return X,Y


def main():
    np.random.seed(0)
    num_classes = 2
    mean = np.random.randn(num_classes)
    cov = np.eye(num_classes)
    X,Y = generate(10, mean, cov, [3.0], True)

    '''
    colors = ['r' if i==0 else 'b' for i in Y[:]]
    plt.scatter(X[:,0], X[:,1], c=colors)
    plt.xlabel('years')
    plt.ylabel('size')
    plt.show()
    '''

    input_dim = 2
    lab_dim = 1
    input_features = tf.placeholder(tf.float32, [None,input_dim]) 
    input_labels = tf.placeholder(tf.float32, [None,lab_dim])
    W = tf.Variable(tf.random_normal([input_dim, lab_dim], name='w'))
    b = tf.Variable(tf.zeros([lab_dim]), name='b')

    output = tf.nn.sigmoid(tf.matmul(input_features, W) + b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print  sess.run(output, feed_dict={input_features:X, input_labels:Y})
       

if __name__ == "__main__":
    main()

