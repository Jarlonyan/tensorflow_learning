
import tensorflow as tf
import numpy as np


def SENet_layer(embed_matrix, field_size, emb_size, pool_op, ratio):
    with tf.variable_scope('SENet_layer'):
        with tf.variable_scope('pooling'):
            if pool_op == 'max':
                z = tf.reduce_max(embed_matrix, axis=2)  # bs*field_size*emb_size  ->  bs*field_size
            elif pool_op == 'avg':
                z = tf.reduce_mean(embed_matrix, axis=2)

        with tf.variable_scope('excitation'):
            z1 = tf.layers.dense(z, units=field_size//ratio, activation='relu')
            a = tf.layers.dense(z1, units=field_size, activation='relu')  #bs*field_size

        with tf.variable_scope('reweight'):
            senet_embed = tf.multiply(embed_matrix, tf.expand_dims(a, axis=-1))   #(bs*field*emb) * (bs*field*1)

        return senet_embed

def main():
    embed_table = tf.constant([[0,    0,    0   , 0   , 0,   0],
                               [-0.11, 0.12, 0.13, 0.14, -0.15, 0.16],
                               [-0.21, 0.22, 0.23, 0.24, -0.25, 0.26],
                               [0.31, 0.32, 0.33, 0.34, -0.35, 0.36],
                               [0.41, 0.42, 0.43, 0.44, -0.45, 0.46]], dtype=tf.float32)

    input_idx = tf.constant([2, 4, 0, 3])
    x = tf.identity(tf.nn.embedding_lookup(embed_table, input_idx))


    senet_embed_matrix = SENet_layer(x, 4, 6, 'avg', 0.2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print 'x=', sess.run(x)
        print 'senet_emb=', sess.run(senet_embed_matrix)
    

if __name__ == '__main__':
    main()


