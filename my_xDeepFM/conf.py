#coding=utf-8

#pre-process
numerical_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day', \
                      'all_action_count', 'last_action', 'all_action_day', 'register_day']
single_features = ['register_type', 'device_type']
multi_features = []


#instance
train_instance_file = 'data/train_instance.data'
valid_instance_file = 'data/valid_instance.data'
test_instance_file = 'data/test_instance.data'

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


