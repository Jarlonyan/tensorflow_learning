#coding=utf-8

#import tensorflow as tf
import tensorflow.compat.v1 as tf

_TENSORS_TO_LOG = dict((x,x) for x in ['learning_rate', 'cross_entropy', 'train_accracy'])

def get_logging_tensor_hook(every_n_iter=100, tensors_to_log=None, **kwargs):
    if tensors_to_log is None:
        tensors_to_log = _TENSORS_TO_LOG

    return tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=every_n_iter)
