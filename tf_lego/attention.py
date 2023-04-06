
import tensorflow as tf
import copy
import numpy as np

def sb_gate_net(exp_out):
    #import pdb; pdb.set_trace()
    exp_out = tf.stack(exp_out, axis=2)
    gate_out = tf.ones([1, 3])
    gate_out = tf.tile(gate_out, [tf.shape(exp_out)[0], 1])
    exp_out = tf.matmul(exp_out, tf.expand_dims(gate_out, -1))
    exp_out = tf.squeeze(exp_out, axis=2)
    return exp_out/3

def self_attention_gate_net(exp_out):
    Q = tf.reshape(exp_out, shape=[-1, 3, 4])
    K = tf.identity(Q)
    V = tf.identity(Q)

    z = tf.matmul(Q, K, transpose_b=True)
    #scaled_z = tf.multiply(z, 1/tf.sqrt(4.0))
    scaled_z = tf.divide(z, tf.sqrt(4.0))
    softmax_z = tf.nn.softmax(scaled_z, dim=1)
    result = tf.matmul(softmax_z, V)
    result = tf.reduce_mean(result, 1)
    return result

def static_mmoe_gate_net(exp_out):
    exp_out = tf.stack(exp_out, axis=2)
    w = tf.Variable(tf.random_normal(shape=[1,3], mean=0, stddev=1))
    gate_out = tf.tile(w, [tf.shape(exp_out)[0], 1])
    gate_out = tf.nn.softmax(gate_out)
    exp_out = tf.matmul(exp_out, tf.expand_dims(gate_out, -1))
    exp_out = tf.squeeze(exp_out, axis=2)
    return exp_out


def attention_fun(exp_out,scaled=True,masked=False):
    Q = tf.reshape(exp_out,shape=[-1, 3, 4])
    K = tf.identity(Q)
    V = tf.identity(Q)
    z = tf.matmul(Q,K,transpose_b=True)
    #d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
    d_k = 4.0
    scaled_z = tf.divide(z, tf.sqrt(d_k))
    softmax_z = tf.nn.softmax(scaled_z, dim=-1)

    output = tf.matmul(softmax_z, V)
    output = tf.reduce_mean(output, axis=1)
    return output

def cin_gate_net(exp_out):
    exp_out = tf.reshape(exp_out, shape=[-1, 3, 4])
    field_nums = [3]
    D = int(exp_out.get_shape().as_list()[-1])
    cross_layer_size = [16,8,4]
    cin_layers = [exp_out]
    final_len = 0
    final_result = []
    split_tensor_0 = tf.split(exp_out, D * [1], 2)
    for idx,layer_size in enumerate(cross_layer_size):
        now_tensor = tf.split(cin_layers[-1], D * [1], 2)
        # Hk x m
        dot_result_m = tf.matmul(split_tensor_0, now_tensor, transpose_b=True)
        dot_result_o = tf.reshape(dot_result_m, shape=[D, -1, field_nums[0] * field_nums[-1]])
        dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])
        filters = tf.get_variable(name="f_" + str(idx), shape=[1, field_nums[-1] * field_nums[0], layer_size], dtype=tf.float32)
        curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
        b = tf.get_variable(name="f_b" + str(idx), shape=[layer_size], dtype=tf.float32, initializer=tf.zeros_initializer())
        curr_out = tf.nn.relu(tf.nn.bias_add(curr_out, b))
        curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

        direct_connect = curr_out
        next_hidden = curr_out
        final_len += layer_size
        field_nums.append(int(layer_size))

        final_result.append(direct_connect)
        cin_layers.append(next_hidden)
    result = tf.concat(final_result, axis=1)
    result = tf.reduce_mean(result, 1)
    return result

def main():
    #w = tf.get_variable(name='w',shape=[1,3], dtype=tf.float32, initializer=tf.zeros_initializer())
    #w = tf.Variable(tf.random_normal(shape=[1,3], mean=0, stddev=1))

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        exp1 = tf.constant([6, 4, 2, 1], dtype=tf.float32)
        exp2 = tf.constant([6, 1, 100, 3], dtype=tf.float32)
        exp3 = tf.constant([900, 2, 1, 8], dtype=tf.float32)
        exp1 = tf.expand_dims(exp1, 0)
        exp2 = tf.expand_dims(exp2, 0)
        exp3 = tf.expand_dims(exp3, 0)
        experts = [exp1, exp2, exp3]
    
        exp_out1 = sb_gate_net(experts)
        exp_out2 = self_attention_gate_net(experts)
        exp_out3 = static_mmoe_gate_net(experts)
        exp_out4 = cin_gate_net(experts)        
        exp_out5 = attention_fun(experts)

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()]) 
        #print sess.run(exp_out1)
        print (sess.run(exp_out2))
        #print sess.run(exp_out3)
        #print sess.run(exp_out4)
        print (sess.run(exp_out5))

def attention():
    Q = tf.Variable(tf.constant(\
         [[1.1, 4.2, 1.3, 1.4], \
          [0.1, 2.3, 2.5, 2.7], \
          [3.1, 3.4, 3.8, 3.0]],\
            dtype=tf.float32), name='query')

    K = tf.Variable(tf.constant(\
        [[1.1, 4.2, 1.3, 1.4], \
        [0.1, 2.3, 2.5, 2.7], \
        [3.1, 3.4, 3.8, 3.0]],\
        dtype=tf.float32), name='key')

    V = tf.Variable(tf.constant( \
        [[1.1, 4.2, 1.3, 1.4], \
        [0.1, 2.3, 2.5, 2.7], \
        [3.1, 3.4, 3.8, 3.0]],\
        dtype=tf.float32), name='value')

    def attention_fun(Q,K,scaled=True,masked=False):
        attention = tf.matmul(Q,K,transpose_b=True)
        if scaled:
            d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
            attention = tf.divide(attention, tf.sqrt(d_k))

        attention = tf.nn.softmax(attention, dim=-1)
        return attention

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        attention = attention_fun(Q,K)
        #print sess.run(attention)
        output = tf.matmul(attention, V)
        output = tf.reduce_mean(output, axis=1)
        print (sess.run(output))

if __name__ == "__main__":
    main() 

