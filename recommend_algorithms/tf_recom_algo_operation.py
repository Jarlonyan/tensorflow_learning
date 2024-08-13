#coding=utf-8
import numpy as np
import tensorflow as tf

g_dict_len = 10

#1. 简单的onehot embedding方式
def onehot_embedding(slot_id, emb_size=4):
    slotx_emb_table = tf.Variable(tf.random.normal([g_dict_len, emb_size], stddev=0.35), name="onehot_emb_slot%s"%slot_id)
    slotx_index = tf.constant([2,1,3], dtype=tf.int64)
    slotx_emb = tf.reshape(tf.nn.embedding_lookup(slotx_emb_table, slotx_index), shape=[-1, emb_size])
    print("emb(slot"+str(slot_id)+")=\n", slotx_emb)
    return slotx_emb

#2. multihot embedding, sum pooling方式
def multihot_embedding(sess, slot_id):
    slotx_emb_table = tf.Variable(tf.random.normal([g_dict_len, emb_size], stddev=0.35), name="onehot_emb_slot%s"%slot_id)
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
    print("emb(slot"+str(slot_id)+")=\n", slotx_emb)
    return slotx_emb

#3. multihot embedding with attention: 用item_emb对multihot做加权求和的attention。用Q、K计算权重，对V重新赋值
#Q: query( to match others)
#K: key (to be mathed)
#V: information to be extrated
def attention_func(Q, K, V):
    Z = tf.matmul(Q, K, transpose_b=True)
    print("Q=\n", Q)
    print("K=\n", K)
    print("V=\n", V)
    print("Z=\n", Z)
    dk = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
    Z = tf.divide(Z, tf.sqrt(dk))
    Z = tf.nn.softmax(Z, dim=-1)
    res = tf.matmul(Z, V)
    res = tf.reduce_mean(res, axis=0)
    return res

def multihot_attention_embedding(slot_id, batch_ids, item_emb, emb_size):
    slotx_emb_table = tf.Variable(tf.random.normal([g_dict_len, emb_size], stddev=0.35), name="onehot_emb_slot%s"%slot_id)
    batch_emb = []
    item_emb_list = tf.split(item_emb, 3, axis=0) #item_emb是batch的，先拆分开来
    print("item_emb=\n", item_emb)
    print("emb_table=\n", slotx_emb_table)
    for (ids,Q) in zip(batch_ids, item_emb_list):
        print("ids=", ids)
        ids = tf.constant(ids, dtype=tf.int64)
        V = tf.nn.embedding_lookup(slotx_emb_table, ids) #V.shape=m*d, m是这个样本的这个slot是m-hot，d是emb维度
        res = attention_func(Q, V, V)      #Q.shape=1*d, d是emb维度
        batch_emb.append(res)
        print('=====================================')
    slotx_emb = tf.stack(batch_emb, axis=0)
    print("emb(slot"+str(slot_id)+")=\n", slotx_emb)
    return slotx_emb

#4. SENet
def SENet(emb_matrix, field_size, emb_size, ratio):
    z = tf.reduce_mean(emb_matrix, axis=2)  # bs*field*emb_size  ->  bs*field
    z1 = tf.layers.dense(z, units=field_size/ratio, activation='relu')
    w = tf.layers.dense(z1, units=field_size, activation='relu')  #bs*field
    #sess.run(tf.global_variables_initializer()) #使用过tf.layers.dense的后面，要初始化
    #print("debug_senet, z.shape=", z.shape, ", z1.shape=", z1.shape, ", a.shape=", a.shape)
    senet_emb = tf.multiply(emb_matrix, tf.expand_dims(w, axis=-1))   #(bs*field*emb) * (bs*field*1)
    return senet_emb, w

#5. LHUCNet
def mlp(mlp_input, mlp_dims):
    x = mlp_input # bs*d
    if len(mlp_dims) > 1:
        for idx,dim in enumerate(mlp_dims[0:-1]):
            x = tf.keras.layers.Dense(units=dim, activation='relu')(x)
    x = tf.keras.layers.Dense(units=mlp_dims[-1], activation=None)(x)
    #sess.run(tf.global_variables_initializer())
    return x

def LHUCNet(lhuc_inputs, lhuc_dims, scale_last=False):
    mlp_dims = [256, 256, 128, 2]
    cur_layer = lhuc_inputs
    for idx,dim in enumerate(mlp_dims[:-1]):
        lhuc_output = mlp(lhuc_inputs, lhuc_dims+[int(cur_layer.shape[1])])
        lhuc_scale = 1.000 + 5.000 * tf.nn.tanh(0.200 * lhuc_output)
        cur_layer = mlp(cur_layer*lhuc_scale, [dim])

    if scale_last:
        lhuc_output = mlp(lhuc_inputs, lhuc_dims+[mlp_dims[-1]])
        lhuc_scale = 1.000 + 5.000 * tf.nn.tanh(0.200 * lhuc_output)
        cur_layer = cur_layer * lhuc_scale

    cur_layer = mlp(cur_layer, [mlp_dims[-1]])
    return cur_layer

#6. NAS-----------------------------------------------------------------------------------------
def alloc_emb_for_nas_v1(slots=[], target_vec_sizes=[0,1,2,4], T=0.2):
    print("total slots for nas",len(slots),"target vec size for search",target_vec_sizes,"temperature", T)
    max_size= max(target_vec_sizes)
    masks =[]
    for i,mask_size in enumerate(target_vec_sizes):  #主要变化了6-9行
        if i>0:
            cur_mask = [[0.]*target_vec_sizes[i-1]+[1.0]*(mask_size-target_vec_sizes[i-1])+[0.0]*(max_size-mask_size)]
        else:
            cur_mask=[[1.0]*mask_size +[0.0]*(max_size-mask_size)]
        print('cur_mask=', cur_mask)
        masks.append(cur_mask)  # 1* emb
    import numpy as np
    masks = np.concatenate(masks,axis=0)
    print("before avg", masks)
    masks = masks/np.sum(masks,axis=0,keepdims=True)
    print("after avg", masks)
    total_mask = tf.constant(masks,name="masks",dtype=tf.float32)
    print("alloc masks", total_mask)
    embeddings =[]
    for slot in slots:
        emb = onehot_embedding(slot, max_size)
        emb = tf.keras.layers.BatchNormalization(axis=-1, center=True, scale=True, trainable=True)(emb) #加上BN层
        embeddings.append(emb)
    embeddings = tf.stack(embeddings,axis=1,name="original_embeds") # BN * slots * emb
    logits = tf.Variable(tf.zeros((len(slots),len(target_vec_sizes))), name="nas_choice_logits")
    choice_probs = tf.nn.softmax(logits/T, axis=1,name="nas_choice_prob") # slots * 3
    tf.summary.histogram("nas_choice_probs",choice_probs)
    choice_matrix = tf.matmul(choice_probs,total_mask,name="choice_probs") # slots* emb
    output_embs = tf.expand_dims(choice_matrix,axis=0) * embeddings
    print("output_embs=", output_embs)
    print("flatten output_embs=", tf.keras.layers.Flatten()(output_embs))
    return tf.keras.layers.Flatten()(output_embs), logits

def alloc_emb_for_nas_v2(slots, emb_sizes=[0,1,2,3,4], T=0.2):
    print ("total slots for nas", len(slots))
    print ("target vec size for search", emb_sizes)
    print ("Temperature", T)
    # embeds
    emb_size = max(emb_sizes)
    if emb_sizes[0] != 0:
        emb_sizes.insert(0,0)
    embeds = [ b_norm(self.new_embedding(slot, emb_size)) for slot in slots ]
    embeds = tf.stack(embeds, axis=1, name="original_embeds")
    # embeds
    # mask
    masks = [[np.zeros(emb_size)]]
    for i in range(1,len(emb_sizes)):
        mask = np.zeros(emb_size)
        mask[emb_sizes[i-1]:emb_sizes[i]] = 1.0
        masks.append([mask])
    masks = np.concatenate(masks, axis=0)
    print(masks)
    mask_matrix = tf.constant(masks, name="masks", dtype=tf.float32)
    # (bs, slots, emb_size)
    logits = M.get_variable(name="nas_choice_logits",
                            shape=(int(embeds.shape[1]), len(emb_sizes)),
                            initializer=initializers.Zeros())
    choice_probs = tf.nn.softmax(logits / T, axis=1, name="nas_choice_prob")
    tf.summary.histogram("nas_choice_probs", choice_probs)
    # (slots, emb_size)
    choice_matrix = tf.matmul(choice_probs, mask_matrix, name="choice_matrix")
    # mask
    print ("choice_matrix.shape", choice_matrix.shape)
    print ("embeds.shape", embeds.shape)
    # (bs, slots, emb_size)
    output_embs = tf.expand_dims(choice_matrix, axis=0) * embeds
    # 加速小概率的归0
    alpha = 0.3
    threshold = 1.0 / len(emb_sizes) * alpha  # 均值* alpha，小于这个值的加速变0 --> [0.06]
    print("threshold=", threshold)
    need_boost = tf.where(choice_probs < threshold, tf.ones_like(choice_probs), tf.zeros_like(choice_probs))
    boost_sum = tf.reduce_sum(need_boost)
    boost_sum= tf.Print(boost_sum, [ boost_sum, boost_sum / (len(slots)* len(emb_sizes)) ], message="nas_need_boost_op_sum_and_ratio")
    tf.summary.scalar("loss_stat/need_boost_ops_sum", boost_sum)
    tf.summary.scalar("loss_stat/need_boost_ops_ratio", boost_sum/ (len(slots)* len(emb_sizes)))
    boost_loss = tf.reduce_sum(need_boost * choice_probs) # 都是正数，加起来就行
    # 0 op不能处于舒适区
    comfort_min, comfort_max = 0.6, 0.8
    zeros_probs = choice_probs[:, :1] # 默认第0位为0 op
    zero_in_comfort_zone = tf.cast(tf.logical_and(zeros_probs>comfort_min, zeros_probs<comfort_max), dtype=tf.float32)
    zero_in_comfort_zone_sum = tf.reduce_sum(zero_in_comfort_zone)
    zero_in_comfort_zone_sum = tf.Print(zero_in_comfort_zone_sum,
                                        [zero_in_comfort_zone_sum, zero_in_comfort_zone_sum/len(slots)], message="nas_zero_in_comfort_zone_sum_and_ratio")
    bigger_than_comfort_max = tf.cast(zeros_probs>comfort_max, dtype=tf.float32)
    bigger_than_comfort_max_sum = tf.reduce_sum(bigger_than_comfort_max)
    tf.summary.scalar("loss_stat/zero_in_comfort_zone_sum", zero_in_comfort_zone_sum)
    tf.summary.scalar("loss_stat/zero_in_comfort_zone_sum_ratio", zero_in_comfort_zone_sum/len(slots))
    tf.summary.scalar("loss_stat/bigger_than_comfort_zone_sum", bigger_than_comfort_max_sum)
    tf.summary.scalar("loss_stat/bigger_than_comfort_zone_sum_ratio", bigger_than_comfort_max_sum/len(slots))
    comfort_zone_loss = tf.reduce_sum(1.0 - zero_in_comfort_zone * zeros_probs)
    return tf.layers.flatten(output_embs), logits, {'boost_loss':boost_loss, 'comfort_loss':comfort_zone_loss}

def nas_model_two_stage(stage=0):
    if stage == 0:
        nas_emb, _ = alloc_emb_for_nas_v1(slots=[1, 3, 8])
        bias_input = tf.Variable(tf.random.normal([3,1], stddev=0.35), name="bias_input")
        concat_input = tf.concat([nas_emb, bias_input], axis=1, name="concat_input")
    elif stage == 1:
        nas_emb = xxxx
        bias_input = tf.Variable(tf.random.normal([3,1], stddev=0.35), name="bias_input")
        concat_input = tf.concat([nas_emb, bias_input], axis=1, name="concat_input")
    mlp(concat_input, [32, 16, 8, 1])

#---------------------------------------------------------------------------------------------------

def main():
    #emb_slot1 = onehot_embedding(slot_id=1, emb_size=4)
    #emb_slot2 = multihot_embedding(slot_id=2, emb_size=4)

    batch_ids = [ #batch_size=3,每一行表示一个user_recent序列
        [5,2,6,1],
        [3,2,5],
        [6,8]
    ]
    #emb_slot3 = multihot_attention_embedding(3, batch_ids, emb_slot1, 4)

    #SENet
    #x = tf.stack([emb_slot1, emb_slot2], axis=1)
    #senet_emb_matrix, f_weight = SENet(sess, x, 2, g_emb_size, 0.2) #2表示有2个slot
    #print('senet_emb=\n', sess.run(senet_emb_matrix))
    #print('f_weight=\n', sess.run(f_weight))
    

    #LHUC
    #lhuc_inputs = tf.concat([emb_slot1, emb_slot2], axis=1)
    #lhuc_dims = [128, 128]
    #lhuc_output = LHUCNet(sess, lhuc_inputs, lhuc_dims, False)
    #print("lhuc_output=\n", sess.run(lhuc_output))
    #print("lhuc_output.shape=", lhuc_output.shape)

    #NAS
    nas_model_two_stage(stage=0)


if __name__ == '__main__':
    main()


