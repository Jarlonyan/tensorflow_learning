#coding=utf-8
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
#tf.disable_v2_behavior()

def mlp_layer(input, mlp_dims=[64, 16, 4]):
    x = input # bs*d
    for idx,dim in enumerate(mlp_dims[0:-1]):
        x = tf.compat.v1.layers.dense(x, units=dim, activation='relu')
    x = tf.compat.v1.layers.dense(x, units=mlp_dims[-1], activation=None)
    return x

#1. SENet
def SENet(embed_list, field_size, emb_size, ratio):
    z = tf.reduce_mean(input, axis=2)  # bs*field*emb_size  ->  bs*field
    z1 = tf.layers.dense(z, units=field_size/ratio, activation='relu')
    w = tf.layers.dense(z1, units=field_size, activation='relu')  #bs*field
    #    tf.global_variables_initializer()) #使用过tf.layers.dense的后面，要初始化
    #print("debug_senet, z.shape=", z.shape, ", z1.shape=", z1.shape, ", a.shape=", a.shape)
    senet_emb = tf.multiply(input, tf.expand_dims(w, axis=-1))   #(bs*field*emb) * (bs*field*1)
    return senet_emb, w


def main():
    #先用随机向量模拟一个all_concat_embedding，实际是各个slot的embedding层的concat起来
    all_concat_embedding = tf.compat.v1.get_variable(name="all_concat_embedding", shape=(4, 8), initializer=tf.compat.v1.glorot_uniform_initializer()) #bs x emb_size

    tf.compat.v1.global_variables_initializer()
    print('input=\n', all_concat_embedding)
    print('intpu.shape=', all_concat_embedding.shape)
    nn_output = mlp_layer(all_concat_embedding, [64, 16, 4])
    print('nn_output=\n', nn_output)

    output = SENet(all_concat_embedding, 20, 8, 0.3)
    print('output=\n', output)

if __name__ == '__main__':
    main()


