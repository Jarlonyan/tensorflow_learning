#coding=utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import time
import collections

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
            labels = labels + labels.decode("gb2312")
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

import sys
print sys.getdefaultencoding()
print ("字表大小=", words_size)
wordlabel = get_ch_label_v(training_file, word_num_map)




