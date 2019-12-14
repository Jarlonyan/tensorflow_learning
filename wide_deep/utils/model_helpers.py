#coding=utf-8

import numbers
import tensorflow as tf

def past_stop_threshold(stop_threshold, eval_metric):
    if stop_threshold is Nonde:
        return False
    if not isinstance(stop_threhold, numbers.Number):
        raise ValueError("Threshold for checking stop conditions must be a number.")
    if not isinstance(eval_metric, numbers.Number):
        raise ValueError("Eval metric being checked against stop conditions must be a number.")

    if eval_metric >= stop_threshold:
        tf.logging.info("Stop threshold of {} was passed with metric value {}.".format(stop_threshold, eval_metric))
        return True

    return False


