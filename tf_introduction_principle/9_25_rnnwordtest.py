#coding=utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
import collections

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec)+" sec"
    elif sec<(60*60):
        return str(sec/60)+" min"
    else:
        return str(sec/(60.0*60.0))+" hour"

tf.reset_default_graph()
training_file = 'data/9_25_wordstest.data'

def get_ch_label(txt_file):
    labels = ""
    with open(txt_file, 'rb') as f:
        for label in f:
            #print label.encode('utf-8')
            labels = labels + label.encode("utf-8")
            #labels += label
    #print labels.encode("utf-8")
    return labels

def get_ch_label_v(txt_file, word_num_map, txt_label=None):
    words_size = len(word_num_map)
    to_num = lambda word: word_num_map.get(word, words_size)
    if txt_file != None:
        txt_label = get_ch_label(txt_file)
    labels_vector = list(map(to_num, txt_label))
    return labels_vector

training_data = get_ch_label(training_file)
print ("loaded training data...")

counter = collections.Counter(training_data)
words = sorted(counter)
words_size = len(words)
word_num_map = dict(zip(words, range(words_size)))

print "字表大小=".encode('utf-8'), words_size
wordlabel = get_ch_label_v(training_file, word_num_map)
print wordlabel


#-----------model----------------- 
learning_rate = 0.001
training_epoch = 100
display_step = 60
n_input = 4
n_hidden1 = 256
n_hidden2 = 512
n_hidden3 = 512

x = tf.placeholder("float", [None, n_input, 1])
wordy = tf.placeholder("float", [None, words_size])

x1 = tf.reshape(x, [-1, n_input])
x2 = tf.split(x1, n_input, 1)

rnn_cell = rnn.MultiRNNCell([rnn.LSTMCell(n_hidden1), rnn.LSTMCell(n_hidden2), rnn.LSTMCell(n_hidden3)])
outputs, states = rnn.static_rnn(rnn_cell, x2, dtype=tf.float32)

pred = tf.contrib.layers.fully_connected(outputs[-1], words_size, activation_fn=None)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=wordy)) 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(size_input,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

savedir = "log/rnnword/"
saver = tf.train.Saver(max_to_keep=1)

