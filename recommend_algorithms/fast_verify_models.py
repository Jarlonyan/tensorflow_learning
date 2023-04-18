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

def new_embedding(self, slot, dim):
    return self.fc_dict[slot].get_vector(self.fc_dict[slot].feature_slot.add_slice(dim))

#1. all_interaction
def all_interaction(vec_slots, allint_slots=None, allint_dim=None, compress_dim=None):
    '''
    vec_slots: VEC_SLOTS = [(1,3), (2,4), (3,2), (4,2), (5,3) ], 第一维是slot_id,第二维是dim，每个slot的dim不一样
    all_int_slots: ALLINT_SLOTS = [1, 2, 3, 10, 11, 12, 22]，下面会给他们分配相同dim, 即allint_dim
    '''
    def all_interaction_layer(slot_embeddings, compress_dim=8, tag='default'):
        """
        :param input_embedding: bs * slot * k
        :param compress_dim: output dim
        :return: flattened interaction
        """
        tag = "allint_%s_" % tag
        stack_embeddings = tf.stack(slot_embeddings, axis=1) # bs * slot * fm_dim
        fm_dim = int(stack_embeddings.shape[2])

        transposed = tf.transpose(stack_embeddings, perm=[0, 2, 1])  # bs * fm_dim * slot
        compress_wt = M.get_variable(shape=[int(transposed.shape[-1]), compress_dim],
                                    initializer=initializers.GlorotNormal(),
                                    name="compress_wt") # 1 * slot * compress_dim
        compress_bias = M.get_variable(shape=[1, fm_dim, compress_dim],
                                        initializer=initializers.Zeros(),
                                        name="compress_bias")
        
        # subs tensordot
        transposed = tf.reshape(transposed, [-1, int(transposed.shape[-1])]) # (bs x fm_dim) * slot
        embed_transformed = tf.matmul(transposed, compress_wt) # (bs x fm_dim) * compress_dim
        embed_transformed = tf.reshape(embed_transformed, [-1, fm_dim, compress_dim]) + compress_bias # bs * fm_dim * compress_dim
        #embed_transformed = tf.tensordot(transposed, compress_wt, axes=1) + compress_bias  # bs * fm_dim * compress_dim
        interaction = tf.matmul(stack_embeddings, embed_transformed, name=tag + "interaction_result") # bs * slot * compress_dim
        tf.summary.histogram("all_interaction result", interaction)
        return tf.layers.flatten(interaction)

    embeddings = []
    for slot, dim in vec_slots:
        embedding = self.new_embedding(slot, dim)
        embeddings.append(embedding)

    if allint_slots:
        allint_embeddings = [ self.new_embedding(slot, allint_dim) for slot in allint_slots ]
        return tf.concat(embeddings, axis=1), all_interaction_layer(allint_embeddings, compress_dim)
    else:
        return tf.concat(embeddings, axis=1), None


#2. SENet
def SENet(input, field_size, emb_size, ratio):
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


