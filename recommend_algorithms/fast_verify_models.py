#coding=utf-8
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
#tf.disable_v2_behavior()

def mlp_layer(input, mlp_dims=[64, 16, 4]):
    x = input # bs*d
    for idx,dim in enumerate(mlp_dims[0:-1]):
        x = tf.compat.v1.layers.dense(x, units=dim, activation='relu')
    x = tf.compat.v1.layers.dense(x, units=mlp_dims[-1], activation=None)
    return x

def main():
    #先用随机向量模拟一个all_concat_embedding，实际是各个slot的embedding层的concat起来
    all_concat_embedding = tf.compat.v1.get_variable(name="all_concat_embedding", shape=(4, 8), initializer=tf.compat.v1.glorot_uniform_initializer()) #bs x emb_size

    tf.compat.v1.global_variables_initializer()
    print('input=\n', all_concat_embedding)
    print('intpu.shape=', all_concat_embedding.shape)
    nn_output = mlp_layer(all_concat_embedding, [64, 16, 4])
    print('nn_output=\n', nn_output)

if __name__ == '__main__':
    main()


