"""
@Time   : 2019/2/20
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : market_eval.py
"""
import os
import cv2
import time
import torch
import tensorflow as tf
import numpy as np
import PIL.Image as Image

from utils.eval_utils import *
from scipy.spatial.distance import cdist
from torchvision import transforms
from data.market1501 import Market1501, RandomIdSampler
from torch.utils.data import dataloader
import multiprocessing

flags = tf.flags
flags.DEFINE_string("pb_path", default="./model_bn.pb", help="saved pb path")
flags.DEFINE_string("data_path", default='/data01/dataset/ReId/Market-1501-v15.09.15', help="data path")
flags.DEFINE_integer("batch_size", default=32, help="test batch size")
FLAGS = flags.FLAGS

pb_path = FLAGS.pb_path
root = FLAGS.data_path
batch_test = FLAGS.batch_size
workers = int(multiprocessing.cpu_count() / 2)


def fliphor(inputs):
    inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
    return inputs.index_select(3, inv_idx)


def inference_dataset(loader):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)

            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Get handles to input and output tensors
            output = tf.get_default_graph().get_tensor_by_name("person_embedding:0")
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            embedding_list = []
            for batch_image, _ in loader:
                batch_image_flip = fliphor(batch_image)

                image = np.transpose(batch_image.data.numpy(), [0,2,3,1])
                image_flip = np.transpose(batch_image_flip.numpy(), [0,2,3,1])

                output_ = sess.run(output, feed_dict = {image_tensor: image})
                output_flip = sess.run(output, feed_dict = {image_tensor: image_flip})

                output_sum = output_ + output_flip
                output_norm = np.linalg.norm(output_sum, axis=1, keepdims=True, ord=2)
                output_result = output_sum/output_norm
                embedding_list.append(output_result)

    return np.concatenate(embedding_list, axis=0)

if __name__ == '__main__':
    # query_features = inference_dataset(query_path)
    # test_features = inference_dataset(test_path)
    test_transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    query_dataset = Market1501(root + '/query', transform=test_transform)
    query_loader = dataloader.DataLoader(query_dataset, batch_size=batch_test, num_workers=workers)

    test_dataset = Market1501(root + '/bounding_box_test', transform=test_transform)
    test_loader = dataloader.DataLoader(test_dataset, batch_size=batch_test, num_workers=workers)

    query_features = inference_dataset(query_loader)
    test_features = inference_dataset(test_loader)
    dist = cdist(query_features, test_features)

    m_ap = mean_ap(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras)
    r = cmc(dist, query_dataset.ids, test_dataset.ids, query_dataset.cameras, test_dataset.cameras,
                        separate_camera_set=False,
                        single_gallery_shot=False,
                        first_match_break=True)

    print('mAP=%f, r@1=%f, r@3=%f, r@5=%f, r@10=%f' % (m_ap, r[0], r[2], r[4], r[9]))
    # mAP = 0.873147, r @ 1 = 0.947447, r @ 3 = 0.974762, r @ 5 = 0.983967, r @ 10 = 0.990499
