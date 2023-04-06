#coding=utf-8
import numpy as np
#import tensorflow as tf

import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

'''
1                  #维度为0的标量
[1, 2, 3]          #维度为1，包含3个元素. 注意shape=(3,)，而不是(3)。在python中tuple必须带有一个逗号
[[1, 2], [3, 4]]   #维度为2， shape=(2, 2)
'''

#1. tf.reshape对tensor按照指定形状进行变化
def tf_reshape(sess):
    a = tf.constant([1,2,3,4,5,6,7,8,9])
    t1 = tf.reshape(a, [3,3])
    t2 = tf.reshape(a, [1,-1]) #-1表示该维度下按照原有维度自动计算
    print("t1=\n", sess.run(t1))
    print("t2=\n", sess.run(t2))

#2. tf.expand_dims，对tensor插入一个维度
def tf_expand_dims(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    t1 = tf.expand_dims(a, 0)
    t2 = tf.expand_dims(a, 1)
    b = t1 - t2
    print("t1=\n", sess.run(t1))
    print("t2=\n", sess.run(t2))
    print("b=\n", sess.run(b))
    print(t1.shape)
    print(t2.shape)
    t3 = tf.expand_dims(a, 2)
    t4 = tf.expand_dims(a, -1)
    print(t3.shape)
    print(t4.shape)

#3. tf.squeeze删除tensor中所有大小是1的维度
def tf_squeeze(sess):
    #a = tf.Variable([[[[1],[1],[1]], [[2],[2],[2]]]])
    a = tf.Variable([1,1,1,1,1]) #shape=(bs,)
    init = tf.global_variables_initializer()
    sess.run(init)
    b = tf.squeeze(a)
    print ("a.shape=", a.shape, ", tf.rank=", sess.run(tf.rank(a)))
    print ("b.shape=", b.shape)
    print ("a=\n", sess.run(a))
    print ("tf.squeeze(a)=\n", sess.run(b))
    c = tf.expand_dims(b, 0)
    print("tf.expand_dims=\n", sess.run(c))
    d = tf.tile(c, [5,1])
    print("tf.tile=\n", sess.run(d))

#4. tf.tile是平铺的意思，用于在同一维度上的复制. tf.tile(input, multiples, name=None) input是输入，multiples是同一维度上复制的次数
def tf_tile(sess):
    a = tf.constant([[1,2], [3,4]], name="a")
    b = tf.tile(a, [3,1])
    print("a=\n", sess.run(a))
    print("tf.tile(a,[3,1])=\n", sess.run(b))

#5. tf.split对value张量沿着axis维度，按照num_or_size_splits个数切分
def tf_split(sess):
    a = tf.Variable([[[[1],[1],[1]], [[2],[2],[2]]]])
    #b = tf.split(a, num_or_size_splits ,axis=0)

#6. tf.transpose调换tensor的维度顺序，其实就是矩阵转置
def tf_transpose(sess):
    a = tf.constant([[1,2,3], [4,5,6]]) #shape=(2,3)
    b = tf.transpose(a, perm=[1,0])
    print("a=\n", sess.run(a))
    print("tf.transpose(a,perm=[1,0])=\n", sess.run(b))

#7. tf.stack是沿着某个维度pack
def tf_stack(sess):
    t1 = tf.constant([1,4]) #shape=(2,)
    t2 = tf.constant([2,5]) #shape=(2,)
    t3 = tf.constant([3,6]) #shape=(2,)
    a = tf.stack([t1, t2, t3])
    b = tf.stack([t1, t2, t3], axis=1)
    print("t1.shape=", t1.shape)
    print("t1=", sess.run(t1))
    print("t2=", sess.run(t2))
    print("t3=", sess.run(t3))
    print("tf.stack([t1,t2,t3])=\n", sess.run(a))         #shape=(3,2)
    print("tf.stack([t1,t2,t3], axis=1)=\n", sess.run(b)) #shape=(2,3)

#8. tf.unstack将输入的tensor按照指定的行或列进行拆分，并输出含有num个tensor的list
def tf_unstack(sess):
    a = tf.constant([[1,2,3], [4,5,6]]) #shape=(2,3)
    b = tf.unstack(a, axis=1)  #所以拆分后有3个tensor
    print("a=\n", sess.run(a))
    print("tf.unstack(a,axis=1)(type=",type(b),")=>\n", b)
    for x in b:
        print (x)

#9. tf.reverse沿着维度进行序列反转
def tf_reverse(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    b = tf.reverse(a, [True, False])
    print("a=\n", sess.run(a))
    print("tf.reverse(a,[True,False])=\n", sess.run(b))

#10. tf.gather根据indices所指示的参数获取tensor中的切片
def tf_gather(sess):
    a = tf.constant([11,22,33,44,55])
    b = tf.gather(a, [2,0,3])
    c = tf.gather(a, [1,4,2])
    print("a=\n", sess.run(a))
    print("tf.gather(a,[2,0,3])=\n", sess.run(b))
    print("tf.gather(a,[1,4,2])=\n", sess.run(c))

#11. tf.one_hot生成符合onehot编码的张量
def tf_onehot(sess):
    indices = [0,2,-1,1]  #要生成的张量
    depth = 3             #在depth长度的数组中，哪个索引的值为onehot值
    on_value = 1.0        #为onehot值时，该值为多少
    off_value = 0.0       #非onehot值时，该值为多少
    axis = -1             #axis=-1时，生成的shape=[indices长度,depth]。axis=0时，shape=[depth,indices长度]
    a = tf.one_hot(indices, depth, on_value, off_value, axis)
    print("indices=", indices)
    print("tf.one_hot(indices, ...)=\n", sess.run(a))

#12. tf.nn.softmax_cross_entropy_with_logits, softmax损失函数
def tf_softmax_loss(sess):
    labels = [[0,0,1], [0,1,0]]  #one-hot
    logits = [[2,0.5,6], [0.1,0,3]]
    res1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    print("res1.shape=", res1.shape)

#13. tf.linag.diag_part
def tf_diag_part(sess):
    logits = tf.constant([[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6]])
    sfx_prob = tf.linalg.diag_part(logits)
    print("sfx_prob=\n", sess.run(sfx_prob))

#14. tf.eye，单位矩阵
def tf_eye(sess):
    a = tf.eye(5, 5, dtype=tf.int32)
    print("tf.eye=\n", sess.run(a))

def main():
    with tf.Session() as sess:
        #tf_reshape(sess)
        tf_expand_dims(sess)
        #tf_squeeze(sess)
        #tf_tile(sess)
        ##tf_split(sess)
        #tf_transpose(sess)
        #tf_stack(sess)
        #tf_unstack(sess)
        #tf_reverse(sess)
        #tf_gather(sess)
        #tf_onehot(sess)
        #tf_softmax_loss(sess)
        #tf_diag_part(sess)
        #tf_eye(sess)

if __name__ == '__main__':
    main()


