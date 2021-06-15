#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

g_emb_size = 4
g_dict_len = 10


def onehot_embedding(sess, slot_id):
    slotx_emb_table = tf.get_variable(name="onehot_emb_slot%s"%str(slot_id), shape=(g_dict_len, g_emb_size), initializer=tf.glorot_uniform_initializer())
    slotx_index = tf.constant([2,1,3], dtype=tf.int64)
    slotx_embed = tf.reshape(tf.nn.embedding_lookup(slotx_emb_table, slotx_index), shape=[-1, g_emb_size])
    sess.run(tf.global_variables_initializer())
    #print("emb_table(slot"+str(slot_id)+")=\n", sess.run(slotx_emb_table))
    print("emb(slot"+str(slot_id)+")=\n", sess.run(slotx_embed))
    return slotx_embed 

def multihot_embedding(sess, slot_id):
    slotx_emb_table = tf.get_variable(name='multi_hot_embeds_slot_%s'%str(slot_id), shape=(g_dict_len, g_emb_size), initializer=tf.glorot_uniform_initializer())
    '''slotx_emb_table = tf.constant([[6.4, 1.2, 0.5, 3.3],
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

    print(slotx_emb_table.shape)

    slotx_embed = tf.nn.embedding_lookup_sparse(slotx_emb_table, slotx_idx, sp_weights=None, combiner="sum") #combiner=sum表示multihot用sum方式聚合
    #slotx_embed = tf.nn.embedding_lookup_sparse(slotx_emb_table, slotx_idx, None, combiner=None)

    sess.run(tf.global_variables_initializer())
    #print("emb_table(slot"+str(slot_id)+")=\n", sess.run(slotx_emb_table))
    print("emb(slot"+str(slot_id)+")=\n", sess.run(slotx_embed))
    return slotx_embed


def SENet(sess, embed_matrix, field_size, emb_size, ratio):
    z = tf.reduce_mean(embed_matrix, axis=2)  # bs*field*emb_size  ->  bs*field
    z1 = tf.layers.dense(z, units=field_size/ratio, activation='relu')
    w = tf.layers.dense(z1, units=field_size, activation='relu')  #bs*field
    sess.run(tf.global_variables_initializer()) #使用过tf.layers.dense的后面，要初始化
    #print("debug_senet, z.shape=", z.shape, ", z1.shape=", z1.shape, ", a.shape=", a.shape)
    senet_embed = tf.multiply(embed_matrix, tf.expand_dims(w, axis=-1))   #(bs*field*emb) * (bs*field*1)
    return senet_embed, w

def mlp(sess, mlp_input, mlp_dims):
    x = mlp_input
    if len(mlp_dims) > 1:
        for idx,dim in enumerate(mlp_dims[0:-1]):
            x = tf.layers.dense(x, units=dim, activation='relu')
    x = tf.layers.dense(x, units=mlp_dims[-1], activation=None)
    sess.run(tf.global_variables_initializer())
    return x

def lhuc_net(sess, lhuc_inputs, lhuc_dims, scale_last=False):
    mlp_dims = [256, 256, 128, 64]
    cur_layer = lhuc_inputs
    for idx,dim in enumerate(mlp_dims[:-1]):
        sess.run(tf.global_variables_initializer())
        lhuc_output = mlp(sess, lhuc_inputs, lhuc_dims+[int(cur_layer.shape[1])])
        lhuc_scale = 1.0 + 5.0 * tf.nn.tanh(0.2 * lhuc_output)
        cur_layer = mlp(sess, cur_layer*lhuc_scale, [dim])

    if scale_last:
        lhuc_output = mlp(lhuc_inputs, lhuc_dims+[nn_dims[-1]])
        lhuc_scale = 1.0 + 5.0 * tf.nn.tanh(0.2 * lhuc_output)
        cur_layer = cur_layer * lhuc_scale

    cur_layer = mlp(sess, cur_layer, [mlp_dims[-1]])
    return cur_layer


def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            emb_slot1 = onehot_embedding(sess, 1)
            emb_slot2 = multihot_embedding(sess, 2)

            '''
            #SENet
            x = tf.stack([emb_slot1, emb_slot2], axis=1)
            senet_embed_matrix, f_weight = SENet(sess, x, 2, g_emb_size, 0.2) #2表示有2个slot
            print('senet_emb=\n', sess.run(senet_embed_matrix))
            print('f_weight=\n', sess.run(f_weight))
            '''

            #'''
            #LHUC
            lhuc_inputs = tf.concat([emb_slot1, emb_slot2], axis=1)
            lhuc_dims = [256, 256, 128, 64]
            lhuc_output = lhuc_net(sess, lhuc_inputs, lhuc_dims, False)
            print("lhuc_output=\n", sess.run(lhuc_output))
            #'''

        #end-with
    #end-with

if __name__ == '__main__':
    main()


