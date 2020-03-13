"""
    - 数据处理，数值型必须是float,离散型必须是int,多值离散是str中间用|隔开，eg. "1|2|3"
    - 暂时不能有缺失值

"""

import Config
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# features
Numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day', \
                    'all_action_count', 'last_action', 'all_action_day', 'register_day']
Single_features = ['register_type', 'device_type']
Multi_features = []

Nnum_embedding = True
Single_feature_frequency = 10
Multi_feature_frequency = 0

# model
FM_layer = True
DNN_layer = True
CIN_layer = False

Uuse_numerical_embedding = False

Embedding_size = 16
Dnn_net_size = [128, 64, 32]
Cross_layer_size = [10, 10, 10]
Cross_direct = False
Cross_output_size = 1

# train
Batch_size = 4096
Epochs = 4000
Learning_rate = 0.01

def get_dict(train, valid, test):
    global_emb_idx = 0
    num_dict = {}
    single_dict = {}
    multi_dict = {}
    backup_dict = {}

    if Num_features and Num_embedding:
        for s in Num_features:
            #each field
            num_dict[s] = global_emb_idx
            global_emb_idx += 1
            #for NaN
            backup_dict[s] = global_emb_idx
            global_emb_idx += 1

    if Single_features:
        for s in Single_features:
            #each field
            frequency_dict = {}
            current_dict = {}
            values = pd.concat([train, valid, test])
            for v in values:
                if v in frequency_dict:
                    frequency_dict[v] += 1
                else:
                    frequency_dict[v] = 1
            for k,v in frequency_dict.items():
                if v > Single_feature_frequency:
                    current_dict[k] = global_emb_idx
                    global_emb_idx += 1
            single_dict[s] = current_dict
            backup_dict[s] = global_emb_idx
    if Multi_features:
        #each field
        for s in Multi_features:
            frequency_dict = {}
            current_dict = {}
            values = pd.concat([train, valid, test)[s]
            for vs in values:
                for v in vs.split('|'):
                    v = int(v)
                    if v in frequency_dict:
                        frequency_dict[v] += 1
                    else:
                        frequency_dict[v] = 1
            for k,v in frequency_dict.items():
                if v>Multi_feature_frequency:
                    current_dict[k] = global_emb_idx
                    global_emb_ix += 1
            multi_dict[s] = current_dict
            backup_dict[s] = global_emb_idx

    return global_emb_idx, num_dict, single_dict, multi_dict, backup_dict

def trains_data(data, num_dict, single_dict, multi_dict, instance_file):
    single_num = 0
    multi_num = 0
    with open(instance_file, 'w') as f:
        #label, index:value
        def instance_write_to_file(line):
            label = line['label']
            f.write(str(label)+',')
            for s in Num_features:
                now_v = line[s]
                f.write(str(num_dict[s])+':'+str(now_v)+',')
                single_num += 1
            for s in Single_features:
                now_v = line[s] 
                if now_v in single_dict[s]:
                    now_idx = 
        #end-func
        data.apply(lambda x: instance_write_to_file(x), axis=1)
    #end-with

def pre_process_data():
    label_num = 0
    single_num = 0
    multi_num = 0

    train = pd.read_csv('data/raw_train_data.csv', index_col=0)
    valid = pd.read_csv('data/raw_valid_data.csv', index_col=0)
    test = pd.read_csv('data/raw_test_data.csv', index_col=0)

    scalar = MinMaxScaler()
    all_data = pd.concat([train, valid, test])

    for s in g_numeric_features:
        scalar.fit(all_data[s].values.reshape(-1,1))
        train[s] = scalar.transorm(tran[s].values.reshape(-1,1))
        valid[s] = scalar.transorm(valid[s].values.reshape(-1,1))
        test[s] = scalar.transorm(test[s].values.reshape(-1,1))
    if (train.shape[1] == valid.shape[1] == test.shape[1]) is False:
        print 'error shape'
        return -1

    global_emb_idx, num_dict, single_dict, multi_dict, backup_dict = get_dict(train, valid, test)

    trains_data(train, num_dict, single_dict, multi_dict, 'data/train_instance.data') 
    trains_data(valid, num_dict, single_dict, multi_dict, 'data/valid_instance.data') 
    trains_data(test, num_dict, single_dict, multi_dict, 'data/test_instance.data') 
    


def main():
    pre_process_data()

if __name__ == '__main__':
    main()
