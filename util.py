# -*- coding: utf-8 -*-
# @Time    : 2020/4/20 19:44
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : util.py
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import functools

import numpy as np
from scipy import misc
import os
from glob import glob
import tensorflow as tf
import tensorflow_gan as tfgan


# Limit values from -1 to 1
def get_images(img_path, img_size):
    filenames = glob(os.path.join(img_path, '*.*'))
    x = [misc.imresize(misc.imread(filename), size=[img_size, img_size])for filename in filenames]
    x = np.array(x) / 127.5 - 1
    return x
