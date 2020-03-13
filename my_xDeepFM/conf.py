#coding=utf-8

instance_train_file = 'data/instance_train.data'
instance_valid_file = 'data/instance_valid.data'
instance_test_file = 'data/instance_test.data'

# model
FM_layer = True
DNN_layer = True
CIN_layer = False

use_numerical_embedding = False

embedding_size = 16

dnn_net_size = [128,64,32]
cross_layer_size = [10,10,10]
cross_direct = False
cross_output_size = 1

# train
batch_size = 4096
epochs = 4000
learning_rate = 0.01


