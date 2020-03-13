"""
    - 数据处理，数值型必须是float,离散型必须是int,多值离散是str中间用|隔开，eg. "1|2|3"
    - 暂时不能有缺失值

"""

import Config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# features
numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day', \
                    'all_action_count', 'last_action', 'all_action_day', 'register_day']
single_features = ['register_type', 'device_type']
multi_features = []

num_embedding = True
single_feature_frequency = 10
multi_feature_frequency = 0

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

def get_dict():
    global_emb_idx = 0
    num_dict = {}
    single_dict = {}
    multi_dict = {}
    backup_dict = {}

    if num_features and num_embedding:
        for s in num_features:
            num_dict[s] = global_emb_idx
            global_emb_idx += 1
            #for NaN
            backup_dict[s] = global_emb_idx
            global_emb_idx += 1
    #end-if

    if single_features:
        for s in single_features:
            #eacho field
            frequency_dict = {}
            current_dict = {}
            values = pd.concat([train, valid, test])
            for v in values:
                if v in frequency_dict:
                    frequency_dict[v] += 1
                else:
                    frequency_dict[v] = 1
            for k,v in frequency_dict.items():
                if v > single

def pre_process_data():
    label_num = 0
    single_num = 0
    multi_num = 0

    train = pd.read_csv('data/raw_train_data.csv', index_col=0)
    valid = pd.read_csv('data/raw_valid_data.csv', index_col=0)
    test = pd.read_csv('data/raw_test_data.csv', index_col=0)

    scalar = MinMaxScaler()
    all_data = pd.concat([train, valid, test])

    for s in numeric_features:
        scalar.fit(all_data[s].values.reshape(-1,1))
        train[s] = scalar.transorm(tran[s].values.reshape(-1,1))
        valid[s] = scalar.transorm(valid[s].values.reshape(-1,1))
        test[s] = scalar.transorm(test[s].values.reshape(-1,1))
    if (train.shape[1] == valid.shape[1] == test.shape[1]) is False:
        print 'error shape'
        return -1

    num 


def main():
    pre_process_data()

if __name__ == '__main__':
    main()
