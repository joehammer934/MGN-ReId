"""
@Time   : 2019/3/19
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : export_pb.py
"""
import torch
import tensorflow as tf
from model.mgn import Model

import numpy as np
import os

from config import config as CFG
from tensorflow.python.framework import graph_util

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

    reid_model = Model(is_training=False, num_class=751)
    outputs = reid_model.predict(batch_image, 'mgn')

    triplet_losses = [reid_model.loss(triplet_logits, batch_label, margin=1.2) for triplet_logits in outputs[1]]
    softmax_losses = [tf.losses.sparse_softmax_cross_entropy(batch_label, softmax_logits) for softmax_logits in outputs[2]]

    triplet_loss = sum(triplet_losses)/len(triplet_losses)
    softmax_loss = sum(softmax_losses)/len(softmax_losses)

    total_loss = triplet_loss + softmax_loss

    epoch_id = tf.Variable(0, name='global_step', trainable=False)
    inc_op = tf.assign_add(epoch_id, 1, name='increment_global_step')

    lr = tf.train.piecewise_constant(epoch_id, boundaries=CFG.lr_steps,
                                     values=CFG.learning_rate,
                                     name='lr_schedule')
    optimizer = tf.train.MomentumOptimizer(lr, momentum=CFG.momentum).minimize(total_loss)

    #save summary
    triplet_loss_summary = tf.summary.scalar(name="triplet_loss", tensor=triplet_loss)
    softmax_loss_summary = tf.summary.scalar(name="softmax_loss", tensor=softmax_loss)
    total_loss_summary = tf.summary.scalar(name="total_loss", tensor=total_loss)
    learning_rate_summary = tf.summary.scalar(name="learning_rate", tensor=lr)
    param_his_summary_list = []
    for var in tf.trainable_variables():
        if "weights" in var.name or \
                "gamma" in var.name or \
                "beta" in var.name or \
                "moving_mean" in var.name or \
                "moving_variance" in var.name:
            param_his_summary_list.append(tf.summary.histogram(var.name, var))
    param_his_summary_list.extend([triplet_loss_summary,
                                   softmax_loss_summary,
                                   total_loss_summary,
                                   learning_rate_summary])
    train_summary = tf.summary.merge(param_his_summary_list)


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
                                                                   ["person_embedding"])
        with tf.gfile.FastGFile(os.path.join(pb_path, "model.pb"), mode='wb') as f:
            f.write(constant_graph.SerializeToString())


if __name__ == '__main__':
    train()
