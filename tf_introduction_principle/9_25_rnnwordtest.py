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
word_label = get_ch_label_v(training_file, word_num_map)
print word_label


#-----------train model----------------- 
learning_rate = 0.001
training_epoch = 200
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

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(wordy,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

savedir = "log/rnnword/"
saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    step = 0
    offset = random.randint(0, n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    ckpt = tf.train.latest_checkpoint(savedir)
    print "ckpt:", ckpt
    start_epo = 0
    if ckpt != None:
        saver.restore(session, ckpt)
        idx = ckpt.find("-")
        start_epo = int(ckpt[idx+1:])
        step = start_epo

    while step < training_epoch:
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        in_words = [[word_label[i]] for i in range(offset, offset+n_input)]
        in_words = np.reshape(np.array(in_words), [-1, n_input, 1])
        out_onehot = np.zeros([words_size], dtype=float)
        out_onehot[word_label[offset+n_input]] = 1.0
        out_onehot = np.reshape(out_onehot, [1,-1])

        _,acc,lossval,onehot_pred = session.run([optimizer, accuracy, loss, pred], feed_dict={x:in_words, wordy: out_onehot})
        loss_total += lossval
        acc_total += acc

        if (step+1) % display_step == 0:
            print ("iter="+str(step+1)+", average loss="+"{:.6f}".format(loss_total/display_step))
            acc_total = 0
            loss_total = 0

        step += 1
        offset += (n_input+1)

    print "finish training"
    saver.save(session, savedir+"rnnwordtest.ckpt", global_step=step)

    #--------------test model--------------------------
    prompt = 32
    sentence = u"生活平凡"
    input_word = sentence.strip()
    input_word = get_ch_label_v(None, word_num_map, input_word)
    for i in range(32):
        keys = np.reshape(np.array(input_word), [-1,n_input,1])
        onehot_pred = session.run(pred, feed_dict={x:keys})
        onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
        #sentence = sentence.decode("utf-8") + words[onehot_pred_index].decode("utf-8")
        print words[onehot_pred_index].decode("utf-8")
        #sentence += words[onehot_pred_index] #.decode("utf-8")
        input_word = input_word[1:]
        input_word.append(onehot_pred_index)
    print sentence


