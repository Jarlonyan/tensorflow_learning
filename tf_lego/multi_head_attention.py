#coding=utf-8
import tensorflow as tf

# 定义多头注意力层
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # 计算每个头的注意力维度.这里的d_model是W^o的行数
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        
        # 初始化查询、键、值的线性层
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        
        # 初始化最后的线性层
        self.dense = tf.keras.layers.Dense(units=d_model)
 
    def split_heads(self, inputs, batch_size):
        # 将输入的最后一个维度分割成(num_heads, depth)
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        # 获取输入的形状
        query, key, value, mask = inputs
        
        # 获取批量大小
        batch_size = tf.shape(query)[0]
        
        # 通过线性层进行映射得到查询、键、值
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)
        print('inputs.shape=', inputs[0].shape)
        print('query.shape=', query.shape)
        
        # 将查询、键、值分割成多个头
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        print('query.shape=', query.shape, ', after split multi head')
        # 计算注意力权重
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)
        # 将头连接并经过最后的线性层
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        print('concat_attention.shape=', concat_attention.shape)
        outputs = self.dense(concat_attention)
        print('outputs.shape=', outputs.shape)
        return outputs, attention_weights
    
    def scaled_dot_product_attention(self, query, key, value, mask):
        # 计算注意力权重
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
        # 应用掩码（如果有）
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        # 计算注意力权重
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, value)
        return attention_output, attention_weights


# 测试代码
# 创建多头注意力层实例
num_heads = 3
d_model = 12
attention_layer = MultiHeadAttention(d_model, num_heads)

# 生成测试输入
batch_size = 2
seq_length = 5
embedding_dim = d_model #这里的emb_dim=12，然后有3个head，相当于是每个head利用了12中的4维，所以前面会判断d_model是num_head的倍数
test_input = tf.random.normal(shape=(batch_size, seq_length, embedding_dim))

# 调用多头注意力层
attention_output, attention_weights = attention_layer([test_input, test_input, test_input, None])

# 打印输出结果
print("Attention Output Shape:", attention_output.shape)
print("Attention Weights Shape:", attention_weights.shape)


