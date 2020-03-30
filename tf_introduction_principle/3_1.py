
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx<w else sum(a[(idx-w):idx])/w for idx,val in enumerate(a)]









