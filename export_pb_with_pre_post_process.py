"""
@Time   : 2019/2/18
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : train_reid_new.py.py
"""
import torch
import tensorflow as tf
from model.mgn import Model

import numpy as np
import os

from tensorflow.python.framework import graph_util
from config import config as CFG
slim = tf.contrib.slim

flags = tf.flags
flags.DEFINE_string("dataset", default="/data/dataset/reid/Market-1501-v15.09.15", help="image data path")
flags.DEFINE_string("train_dir", default="./mgn_reid/exp/train/ckpt", help="ckpt path")
flags.DEFINE_string("summary_path", default="./mgn_reid/exp/train/summary", help="summary path")
flags.DEFINE_string("tf_name_path", default="./tensor_name/fianl_tf_name_v2.txt", help="File that holds the tensorflow tensor name")
flags.DEFINE_string("pt_name_path", default="./tensor_name/final_pt_name.txt", help="File that holds the pytorch tensor name")
flags.DEFINE_string("pt_model_path", default="./pretrain_model/MGN_12_27_M.pt", help="pytorch model path")
flags.DEFINE_string("pb_save_path", default="./mgn_reid/exp/train/save_pb", help="pb file path")
FLAGS = flags.FLAGS

root = FLAGS.dataset
train_dir = FLAGS.train_dir
tf_name_path = FLAGS.tf_name_path
pt_name_path = FLAGS.pt_name_path
pt_model_path = FLAGS.pt_model_path
summary_save_path = FLAGS.summary_path
pb_path = FLAGS.pb_save_path

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(summary_save_path):
    os.makedirs(summary_save_path)
if not os.path.exists(pb_path):
    os.makedirs(pb_path)

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

def combined_static_and_dynamic_shape(tensor):
  """Returns a list containing static and dynamic values for the dimensions.

  Returns a list of static and dynamic values for shape dimensions. This is
  useful to preserve static shapes when available in reshape operation.

  Args:
    tensor: A tensor of any type.

  Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
  """
  static_tensor_shape = tensor.shape.as_list()
  dynamic_tensor_shape = tf.shape(tensor)
  combined_shape = []
  for index, dim in enumerate(static_tensor_shape):
    if dim is not None:
      combined_shape.append(dim)
    else:
      combined_shape.append(dynamic_tensor_shape[index])
  return combined_shape

def restore_model_v2(sess,
                  global_variables,
                  pt_dict,
                  tf_name_path,
                  pt_name_path):
    tf_names = []
    with open(tf_name_path) as f:
        for line in f:
            tf_names.append(line.strip())

    pt_name = []
    with open(pt_name_path) as f:
        for line in f:
            pt_name.append(line.split(" ")[0])

    tf2pt = dict(zip(tf_names, pt_name))

    for var in global_variables:
        if str(var) in tf_names:
            value = pt_dict[tf2pt[str(var)]].numpy()
            if len(np.array(value.shape)) == 4:
                value = np.transpose(value, [2, 3, 1, 0])
            elif len(np.array(value.shape)) == 2:
                value = np.transpose(value, [1, 0])
            print(str(var))
            _ops = tf.assign(var, value)
            sess.run(_ops)



def train():
    batch_image = tf.placeholder(tf.float32, shape=[None, None, None, CFG.channel], name='image_tensor')
    batch_label = tf.placeholder(tf.int32, shape=[None, ], name='label_tensor')

    batch_image_resize = tf.image.resize_images(batch_image, [384, 128])
    batch_image_resize = tf.identity(batch_image_resize, "resize_image")
    #RGB
    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

    batch_image_norm = tf.div((batch_image_resize/255-mean), std, name="image_norm")
    batch_image_flip = tf.image.flip_left_right(batch_image_norm)

    batch_image_input = tf.concat([batch_image_norm, batch_image_flip], 0)

    reid_model = Model(is_training=False, num_class=751)

    outputs = reid_model.predict(batch_image_input, 'mgn')

    tmp_embedding = outputs[0]
    num_image = combined_static_and_dynamic_shape(tmp_embedding)[0]//2

    original_embedding = tmp_embedding[:num_image, :]
    original_embedding = tf.identity(original_embedding, name="original_embedding")
    flip_embedding = tmp_embedding[num_image:, :]
    flip_embedding = tf.identity(flip_embedding, name="flip_embedding")
    sum_embedding = original_embedding+flip_embedding

    norm_embedding = tf.norm(sum_embedding, keep_dims=True, axis=1, name="embedding_norm")
    result_embedding = tf.div(sum_embedding, norm_embedding, name="result_embedding")

    with tf.Session() as sess:
        saver = tf.train.Saver()

        if len(os.listdir(train_dir)) > 0:
            latest_checkpoint_path = tf.train.latest_checkpoint(train_dir)
            saver.restore(sess, latest_checkpoint_path)
            print("restore pretrained checkpoint from %s" % (latest_checkpoint_path))
        else:
            # 从pytorch的预训练模型中加载参数
            restore_model_v2(sess,
                             tf.global_variables(),
                             torch.load(pt_model_path, map_location='cpu'),
                             tf_name_path,
                             pt_name_path)
            print('restore from pytorch best pt..........')

        constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                   ["result_embedding"])
        with tf.gfile.FastGFile(os.path.join(pb_path, 'model_pre_and_post.pb'), mode='wb') as f:
            f.write(constant_graph.SerializeToString())



if __name__ == '__main__':
    train()