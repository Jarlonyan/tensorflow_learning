#coding=utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import tensorflow as tf

img_path = 'data/xxx.jpg'
my_img = mping.imread(os.path.expanduser(img_path))

length = my_img.shape[0]
width = my_img.shape[1]



