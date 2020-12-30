#coding=utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist_data/", one_hot=True)
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10
batch_size = 8

tf.reset_default_graph()

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
x1 = tf.unstack(x, n_steps, 1)

lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
outputs,states = tf.contrib.rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
pred = tf.contrib.layers.fully_connected(outputs[-1], n_classes, activation_fn=None)

learning_rate = 0.001
training_epoch = 30
display_step = 2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 1
    while step*batch_size < training_epoch:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})

        if step%display_step==0:
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(loss, feed_dict={x:batch_x, y:batch_y})
            print "iter="+str(step*batch_size)+", minibatch_loss="+str(loss)+", training_accuracy="+str(acc)
        step += 1
    print "end training"
#end-while


