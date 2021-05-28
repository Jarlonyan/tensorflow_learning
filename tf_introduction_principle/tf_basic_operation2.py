#coding=utf-8
import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf #在tensorflow2的环境下使用tensorflow1.x
tf.disable_v2_behavior()

#1. tf.slice对tensor进行切片操作。begin表示从哪几个维度开始，size表示在input各个维度抽取的元素个数
def tf_slice(sess):
    a = tf.constant([[[1,1,1], [2,2,2]],  [[3,3,3], [4,4,4]],  [[5,5,5], [6,6,6]]])
    print("a.shape=", a.shape, ", a=\n", sess.run(a))
    t1 = tf.slice(a, [1,0,0], [1,1,3])
    t2 = tf.slice(a, [1,0,0], [1,2,3])
    #t3 = tf.slice(a, [1,0,0], [2,3,1])
    print("tf.slice1=\n",sess.run(t1))
    print("tf.slice2=\n",sess.run(t2))
    #print("t3=",sess.run(t3))

#2. tf.split沿着某一个维度将tensor分割成xxx
def tf_split(sess):
    a = tf.constant([[1,1,1,1,1,1,1,1,1,1,1,1], [2,2,2,2,2,2,2,2,2,2,2,2,], [3,3,3,3,3,3,3,3,3,3,3,3]]) #3,12 
    t0,t1,t2 = tf.split(a, [4,6,2], 1)
    print("a=\n", sess.run(a))
    print("t0=\n", sess.run(t0))
    print("t1=\n", sess.run(t1))
    print("t2=\n", sess.run(t2))

#3. tf.concat沿着某一维度连接tensor
def tf_concat(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    b = tf.constant([[7,8,9], [10,11,12]])
    t = tf.concat([a,b], 0)
    print("a=\n", sess.run(a))
    print("b=\n", sess.run(b))
    print("tf.concat([a,b],0)=\n", sess.run(t))

#4. tf.transpose调换tensor的维度顺序，其实就是矩阵转置
def tf_transpose(sess):
    a = tf.constant([[1,2,3], [4,5,6]]) #shape=(2,3)
    b = tf.transpose(a, perm=[1,0])
    print("a=\n", sess.run(a))
    print("tf.transpose(a,perm=[1,0])=\n", sess.run(b))

#5. tf.stack是沿着某个维度pack
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

#6. tf.unstack将输入的tensor按照指定的行或列进行拆分，并输出含有num个tensor的list
def tf_unstack(sess):
    a = tf.constant([[1,2,3], [4,5,6]]) #shape=(2,3)
    b = tf.unstack(a, axis=1)  #所以拆分后有3个tensor
    print("a=\n", sess.run(a))
    print("tf.unstack(a,axis=1)(type=",type(b),")=>\n", b)
    for x in b:
        print (x)

#7. tf.reverse沿着维度进行序列反转
def tf_reverse(sess):
    a = tf.constant([[1,2,3], [4,5,6]])
    b = tf.reverse(a, [True, False])
    print("a=\n", sess.run(a))
    print("tf.reverse(a,[True,False])=\n", sess.run(b))

#8. tf.gather根据indices所指示的参数获取tensor中的切片
def tf_gather(sess):
    a = tf.constant([11,22,33,44,55])
    b = tf.gather(a, [2,0,3])
    c = tf.gather(a, [1,4,2])
    print("a=\n", sess.run(a))
    print("tf.gather(a,[2,0,3])=\n", sess.run(b))
    print("tf.gather(a,[1,4,2])=\n", sess.run(c))

#9. tf.one_hot生成符合onehot编码的张量
def tf_onehot(sess):
    indices = [0,2,-1,1]  #要生成的张量
    depth = 3             #在depth长度的数组中，哪个索引的值为onehot值
    on_value = 1.0        #为onehot值时，该值为多少
    off_value = 0.0       #非onehot值时，该值为多少
    axis = -1             #axis=-1时，生成的shape=[indices长度,depth]。axis=0时，shape=[depth,indices长度]
    a = tf.one_hot(indices, depth, on_value, off_value, axis)
    print("indices=", indices)
    print("tf.one_hot(indices, ...)=\n", sess.run(a))


def main():
    with tf.Session() as sess:
        #tf_slice(sess)
        #tf_split(sess)
        #tf_concat(sess)
        #tf_transpose(sess)
        tf_stack(sess)
        #tf_unstack(sess)
        #tf_reverse(sess)
        #tf_gather(sess)
        #tf_onehot(sess)


if __name__ == '__main__':
    main()


