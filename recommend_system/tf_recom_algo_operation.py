#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

g_emb_size = 4
g_dict_len = 10

#1. 简单的onehot embedding方式
def onehot_embedding(sess, slot_id):
    slotx_emb_table = tf.get_variable(name="onehot_emb_slot%s"%str(slot_id), shape=(g_dict_len, g_emb_size), initializer=tf.glorot_uniform_initializer())
    slotx_index = tf.constant([2,1,3], dtype=tf.int64)
    slotx_emb = tf.reshape(tf.nn.embedding_lookup(slotx_emb_table, slotx_index), shape=[-1, g_emb_size])
    sess.run(tf.global_variables_initializer())
    #print("emb_table(slot"+str(slot_id)+")=\n", sess.run(slotx_emb_table))
    print("emb(slot"+str(slot_id)+")=\n", sess.run(slotx_emb))
    return slotx_emb

#2. multihot embedding方式
def multihot_embedding(sess, slot_id):
    slotx_emb_table = tf.get_variable(name='multi_hot_emb_slot_%s'%str(slot_id), shape=(g_dict_len, g_emb_size), initializer=tf.glorot_uniform_initializer())
    '''
    slotx_emb_table = tf.constant([[6.4, 1.2, 0.5, 3.3],
                                   [0.3, 0.4, 0.5, 0.8],
                                   [1.5, 0.3, 2.2, 1.9],
                                   [0.4, 0.9, 1.1, 4.3]])
    '''

    #定义稀疏矩阵, indices是位置[0,0]表示矩阵的第0行第0列，这样拼出来稀疏矩阵. values是对应emb_table中的索引。dense_shape是稀疏矩阵的长*宽
    #这个稀疏矩阵就是下面这个样子，每一行是一个multihot，行数代表batch_size，列数代表multihot最多允许多少个hot。N表示稀疏矩阵这位置没有存
    #[[1, 2, 3, N, N],
    # [N, N, 2, N, N],
    # [N, N, 3, 1, N]]
    slotx_idx = tf.SparseTensor(indices=[[0,0], [0,1], [0,2], [1,2], [2,2], [2,3]], values=[1,2,3,2,3,1], dense_shape=(10, 5))
    print("slotx_emb_table.shape=",slotx_emb_table.shape)

    slotx_emb = tf.nn.embedding_lookup_sparse(slotx_emb_table, slotx_idx, sp_weights=None, combiner="sum") #combiner=sum表示multihot用sum方式聚合
    sess.run(tf.global_variables_initializer())
    #print("emb_table(slot"+str(slot_id)+")=\n", sess.run(slotx_emb_table))
    print("emb(slot"+str(slot_id)+")=\n", sess.run(slotx_emb))
    return slotx_emb

#2. 用item_emb对multihot做加权求和的attention
def attention_func(Q, K, V):
    Z = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
    Z = tf.divide(Z, tf.sqrt(dk))
    Z = tf.nn.softmax(Z, dim=-1)
    res = tf.matmul(Z, V)
    res = tf.reduce_mean(res, axis=0)
    return res

def multihot_attention_embedding(sess, slot_id, batch_ids, item_emb):
    slotx_emb_table = tf.get_variable(name='multi_hot_atten_emb_slot_%s'%str(slot_id), shape=(g_dict_len, g_emb_size), initializer=tf.glorot_uniform_initializer())
    sess.run(tf.global_variables_initializer())
    batch_emb = []
    item_emb_list = tf.split(item_emb, 3, axis=0) #item_emb是batch的，先拆分开来
    for (ids,Q) in zip(batch_ids, item_emb_list):
        ids = tf.constant(ids, dtype=tf.int64)
        V = tf.nn.embedding_lookup(slotx_emb_table, ids) #V.shape=m*d, m是这个样本的这个slot是m-hot，d是emb维度
        res = attention_func(Q, V, V)      #Q.shape=1*d, d是emb维度
        batch_emb.append(res)
        #print("res=", sess.run(V))
    slotx_emb = tf.stack(batch_emb, axis=0)
    print("emb(slot"+str(slot_id)+")=\n", sess.run(slotx_emb))
    return slotx_emb

#3. SENet
def SENet(sess, emb_matrix, field_size, emb_size, ratio):
    z = tf.reduce_mean(emb_matrix, axis=2)  # bs*field*emb_size  ->  bs*field
    z1 = tf.layers.dense(z, units=field_size/ratio, activation='relu')
    w = tf.layers.dense(z1, units=field_size, activation='relu')  #bs*field
    sess.run(tf.global_variables_initializer()) #使用过tf.layers.dense的后面，要初始化
    #print("debug_senet, z.shape=", z.shape, ", z1.shape=", z1.shape, ", a.shape=", a.shape)
    senet_emb = tf.multiply(emb_matrix, tf.expand_dims(w, axis=-1))   #(bs*field*emb) * (bs*field*1)
    return senet_emb, w

#4. LHUCNet
def mlp(sess, mlp_input, mlp_dims):
    x = mlp_input # bs*d
    if len(mlp_dims) > 1:
        for idx,dim in enumerate(mlp_dims[0:-1]):
            x = tf.layers.dense(x, units=dim, activation='relu')
    x = tf.layers.dense(x, units=mlp_dims[-1], activation=None)
    sess.run(tf.global_variables_initializer())
    return x

def LHUCNet(sess, lhuc_inputs, lhuc_dims, scale_last=False):
    mlp_dims = [256, 256, 128, 64]
    cur_layer = lhuc_inputs
    for idx,dim in enumerate(mlp_dims[:-1]):
        lhuc_output = mlp(sess, lhuc_inputs, lhuc_dims+[int(cur_layer.shape[1])])
        lhuc_scale = 1.000 + 5.000 * tf.nn.tanh(0.200 * lhuc_output)
        cur_layer = mlp(sess, cur_layer*lhuc_scale, [dim])

    if scale_last:
        lhuc_output = mlp(lhuc_inputs, lhuc_dims+[mlp_dims[-1]])
        lhuc_scale = 1.000 + 5.000 * tf.nn.tanh(0.200 * lhuc_output)
        cur_layer = cur_layer * lhuc_scale

    cur_layer = mlp(sess, cur_layer, [mlp_dims[-1]])
    return cur_layer

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            emb_slot1 = onehot_embedding(sess, 1)
            emb_slot2 = multihot_embedding(sess, 2)

            batch_ids = [ #batch_size=3,每一行表示一个user_recent序列
                [5,2,6,1],
                [3,2,5],
                [6,8]
            ]
            emb_slot3 = multihot_attention_embedding(sess, 3, batch_ids, emb_slot1)

            #'''
            #SENet
            x = tf.stack([emb_slot1, emb_slot2], axis=1)
            senet_emb_matrix, f_weight = SENet(sess, x, 2, g_emb_size, 0.2) #2表示有2个slot
            print('senet_emb=\n', sess.run(senet_emb_matrix))
            print('f_weight=\n', sess.run(f_weight))
            #'''

            #'''
            #LHUC
            lhuc_inputs = tf.concat([emb_slot1, emb_slot2], axis=1)
            lhuc_dims = [256, 256, 128, 64]
            lhuc_output = LHUCNet(sess, lhuc_inputs, lhuc_dims, False)
            print("lhuc_output=\n", sess.run(lhuc_output))
            #'''


        #end-with
    #end-with

if __name__ == '__main__':
    main()


