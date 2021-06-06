#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

embed_size = 5

def embeddings(sess):
    slot1_embed_table = tf.get_variable(name='multi_hot_embeds', shape=(100, embed_size), initializer=tf.glorot_uniform_initializer())
    #print("embeds=\n", sess.run(embeds))


    #slot1_index = tf.placeholder(dtype=tf.int64, shape=[None, 2])
    #slot1_value = tf.placeholder(dtype=tf.int64, shape=[None])
    slot1_index = tf.constant([[1,1], [2,2], [3,3]], dtype=tf.int64)
    slot1_value = tf.constant([3,1,2], dtype=tf.int64)

    print(slot1_embed_table.shape)
    print(slot1_index.shape)
    print(slot1_value.shape)

    slot1_embed = tf.nn.embedding_lookup_sparse(
                    slot1_embed_table, 
                    tf.SparseTensor(indices=slot1_index, values=slot1_value, dense_shape=(3, embed_size)),
                    None,
                    combiner="sum")

    sess.run(tf.global_variables_initializer())
    print("embeds=\n", sess.run(slot1_embed))

def get_senet_weights(embeddings):
    slots_num = len(embeddings)
    inputs = []
    for embed in embeddings:
        inputs.append(tf.reduce_mean(embed, axis=1, keepdims=True))
    sequeeze_embedding = tf.concat(inputs, axis=1)
    print(sequeeze_embedding.shape)

    weight_out = modules.DenseTower(name='senet_layer', output_dims=[32, slots_num],
                                    initializers=initializers.GlorotNormal(mode='fan_avg'),
                                    activations=[layers.Relu(), layers.Sigmoid()])(sequeeze_embedding)
    return tf.split(weight_out, slots_num, axis=1)


def main():
    with tf.Session() as sess:
        embeddings(sess)

if __name__ == '__main__':
    main()


