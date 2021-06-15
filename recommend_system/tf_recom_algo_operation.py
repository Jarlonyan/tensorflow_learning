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
    print("emb_table(slot"+str(slot_id)+")=\n", sess.run(slotx_emb_table))
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
    print("emb_table(slot"+str(slot_id)+")=\n", sess.run(slotx_emb_table))
    print("emb(slot"+str(slot_id)+")=\n", sess.run(slotx_embed))
    return slotx_embed

def get_senet_weights(sess, embeddings):
    slots_num = len(embeddings)
    inputs = []
    for embed in embeddings:
        inputs.append(tf.reduce_mean(embed, axis=1, keepdims=True))
    sequeeze_embedding = tf.concat(inputs, axis=1)
    #print("sequeeze_embedding.shape=",sequeeze_embedding.shape)

    sequeeze_embedding = tf.layers.dense(inputs=sequeeze_embedding, units=32, activation=tf.nn.relu)
    weight_out = tf.layers.dense(inputs=sequeeze_embedding, units=2, activation=tf.nn.relu)    

    sess.run(tf.global_variables_initializer())
    print("weight_out=\n", sess.run(weight_out))

    return tf.split(weight_out, 2, axis=1)

def main():
    with tf.Graph().as_default():
        with tf.Session() as sess:
            emb_slot1 = multihot_embedding(sess, 1)
            #emb_slot2 = onehot_embedding(sess, 2)

            """
            emb_slot1 = multihot_embedding(sess, 1)
            emb_slot2 = multihot_embedding(sess, 2)
            t0, t1 = get_senet_weights(sess, [emb_slot1, emb_slot2])
            sess.run(tf.global_variables_initializer())
            print("emb_slot1=\n", sess.run(emb_slot1))
            print("emb_slot2=\n", sess.run(emb_slot2))        
            print("t0=\n", sess.run(t0))
            print("t1=\n", sess.run(t1))
            """

        #end-with
    #end-with

if __name__ == '__main__':
    main()


