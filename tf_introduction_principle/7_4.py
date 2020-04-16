#coding=utf-8
import tensorflow as tf

learning_rate = 0.001
epochs = 20
batch_size = 100
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10

x = tf.placeholder(tf.float32, [None,n_input])
y = tf.placeholder(tf.float32, [None,n_classes])

def multi_layer_perceptron(x, weights, biases):
    layer_1 = tf.matmul(x, weights['h1']) + biases['b1']
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b1']
    layer_2 = tf.nn.relu(layer_2)

    output = tf.matmul(layer_2, weights['out']) + biases['out']
    return output 

weights = {
    'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out' : tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1' : tf.Variable(tf.random_normal([n_hidden_1]),
    'b2' : tf.Variable(tf.random_normal([n_hidden_2]),
    'out' : tf.Variable(tf.random_normal([n_classes])
}

pred = multi_layer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_encropy_with_logits(logits=pred, label=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
     

