import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.d_k = 12
        self.d_v = 16
        
    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], self.d_k),
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.W_k = self.add_weight(shape=(input_shape[-1], self.d_k),
                                          initializer='glorot_uniform',
                                          trainable=True)
        self.W_v = self.add_weight(shape=(input_shape[-1], self.d_v),
                                          initializer='glorot_uniform',
                                          trainable=True)
        
    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)
        
        attention_weights = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32)))
        output = tf.matmul(attention_weights, V)
        
        return output

# 测试数据
inputs = tf.random.normal(shape=(2, 5, 8))  # (batch_size, seq_length, embedding_dim)
# 创建自注意力层实例
attention_layer = SelfAttention()

# 调用自注意力层并输出结果
outputs = attention_layer(inputs)

print('===================>')
# 打印结果
print(outputs)
