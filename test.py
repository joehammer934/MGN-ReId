"""
@Time   : 2019/3/19
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : test.py
"""
import tensorflow as tf

flags = tf.flags
flags.DEFINE_list("a", default=[0.2, 0.3, 0.4], help="")
FLAGS = flags.FLAGS

print(FLAGS.a)