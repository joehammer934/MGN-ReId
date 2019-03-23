"""
@Time   : 2019/2/15
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : mgn.py
"""
import functools
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.python.framework import ops

from model import resnet_utils
from model import resnet_v1

slim = tf.contrib.slim

class Model():
    def __init__(self,
                 num_class,
                 is_training,
                 ):
        self._is_training = is_training
        self._num_class = num_class
        self.batch_norm_fn = functools.partial(layers.batch_norm,
                                               center=True,
                                               scale=True,
                                               epsilon=1e-5,
                                               decay=0.997,
                                               updates_collections=ops.GraphKeys.UPDATE_OPS,
                                               is_training=self._is_training)

    @property
    def __num_class__(self):
        return self._num_class

    def bottleneck(self, x, inner_depth, output_channel, name, downsample=False, stride=1):
        """
        residual network bottleneck
        :param x: input tensor
        :param inner_depth: bottleneck depth
        :param output_channel: output channel
        :param downsample: is down sample
        :param name: bottleneck name
        :return:
        """
        residual = x
        with tf.variable_scope("bottleneck_" + name):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=self.batch_norm_fn,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                out = slim.conv2d(x, inner_depth, [1, 1], stride=1)

                if downsample:
                    out = tf.pad(out, [[0,0], [1,1], [1,1], [0,0]])
                    out = slim.conv2d(out, inner_depth, [3, 3], stride=stride, padding="VALID")
                else:
                    out = slim.conv2d(out, inner_depth, [3, 3], stride=1)

            with slim.arg_scope([slim.conv2d],
                                activation_fn=None,
                                normalizer_fn=self.batch_norm_fn):
                if downsample:
                    residual = slim.conv2d(x, output_channel, [1, 1], stride=2, padding="VALID", scope="downsample")
                if not downsample and x.shape.as_list()[-1] != output_channel:
                    residual = slim.conv2d(x, output_channel, [1, 1], stride=1, scope="downsample")

                out = slim.conv2d(out, output_channel, [1, 1], stride=1)

            out += residual
        return tf.nn.relu(out)

    def parametric_relu(self, x):
        """
        PRelu proposed by:
            "Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification. arXiv:1502.01852"
        :param x: input tensor
        :return: activation result
        """
        alphas = tf.get_variable('Prelu_alpha', x.shape.as_list()[-1],
                                 initializer=tf.constant_initializer(0.25),
                                 dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5

        return pos + neg

    def reduction(self, x, name, depth=256):
        """
        The module to reduce feature map channels
        :param x: input tensor
        :param name: module name
        :param depth: output channel
        :return:
        """
        with tf.variable_scope("reduction_" + name):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=self.batch_norm_fn,
                                activation_fn=None):
                out = slim.conv2d(x, depth, [1, 1])
                out = self.parametric_relu(out)
        return out

    def fully_connect(self, x, num_classes, name):
        out = tf.layers.dense(x,
                              num_classes,
                              activation=None,
                              use_bias=True,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                              kernel_regularizer=slim.l2_regularizer(0.0001),
                              bias_regularizer=slim.l2_regularizer(0.0001),
                              activity_regularizer=None,
                              kernel_constraint=None,
                              bias_constraint=None,
                              trainable=True,
                              name=name,
                              reuse=False)
        return out

    def general_branch(self, input, downsample):
        """
        branch block
        :param input: inout tensor
        :param downsample: is down sample
        :return:
        """
        with tf.variable_scope("block_0"):
            p = self.bottleneck(x=input, inner_depth=256, output_channel=1024, name="v0")
            p = self.bottleneck(x=p, inner_depth=256, output_channel=1024, name="v1")
            p = self.bottleneck(x=p, inner_depth=256, output_channel=1024, name="v2")
            p = self.bottleneck(x=p, inner_depth=256, output_channel=1024, name="v3")
            p = self.bottleneck(x=p, inner_depth=256, output_channel=1024, name="v4")

        with tf.variable_scope("block_1"):
            if downsample:
                p = self.bottleneck(x=p, inner_depth=512, output_channel=2048, name="v0", downsample=downsample, stride=2)
            else:
                p = self.bottleneck(x=p, inner_depth=512, output_channel=2048, name="v0")

            p = self.bottleneck(x=p, inner_depth=512, output_channel=2048, name="v1")
            p = self.bottleneck(x=p, inner_depth=512, output_channel=2048, name="v2")

        return p

    def predict(self, preprocessed_inputs, model):
        """Predict prediction tensors from inputs tensor.

        Outputs of this function can be passed to loss or postprocess functions.

        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.

        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        num_classes = 751
        predict = []
        triplet_losses = []
        softmax_losses = []
        #resnet-50
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, endpoints = resnet_v1.resnet_v1_50(
                preprocessed_inputs, num_classes=None,
                is_training=self._is_training)

        backbone = endpoints['resnet_v1_50/block1/unit_3/bottleneck_v1']

        with tf.variable_scope("resnet_v1_50/sd_block2/unit_1"):
            backbone = self.bottleneck(backbone, 128, 512, "v1", downsample=True, stride=2)
        with tf.variable_scope("resnet_v1_50/sd_block2/unit_2"):
            backbone = self.bottleneck(backbone, 128, 512, "v1", downsample=False, stride=1)
        with tf.variable_scope("resnet_v1_50/sd_block2/unit_3"):
            backbone = self.bottleneck(backbone, 128, 512, "v1", downsample=False, stride=1)
        with tf.variable_scope("resnet_v1_50/sd_block2/unit_4"):
            backbone = self.bottleneck(backbone, 128, 512, "v1", downsample=False, stride=1)

        with tf.variable_scope("resnet_v1_50/sd_block3/unit_1"):
            backbone = self.bottleneck(backbone, 256, 1024, "v1", downsample=True, stride=2)

        #model branch
        if model in {'mgn', 'p1_single'}:
            with tf.variable_scope("p1"):
                p1 = self.general_branch(backbone, downsample=True)
                zg_p1 = layers.avg_pool2d(p1, [12, 4], scope='avg_pool_zg_p1')
                fg_p1 = tf.squeeze(tf.squeeze(self.reduction(zg_p1, name="0"), squeeze_dims=1), squeeze_dims=1)
                l_p1 = self.fully_connect(fg_p1, num_classes, name="fc_id_2048_0")
                predict.append(fg_p1)
                triplet_losses.append(fg_p1)
                softmax_losses.append(l_p1)

        if model in {'mgn', 'p2_single'}:
            with tf.variable_scope("p2"):
                p2 = self.general_branch(input=backbone, downsample=False)
                zg_p2 = layers.avg_pool2d(p2, [24, 8], scope='avg_pool_zg_p2')
                fg_p2 = tf.squeeze(tf.squeeze(self.reduction(zg_p2, name="1"), squeeze_dims=1), squeeze_dims=1)
                l_p2 = self.fully_connect(fg_p2, num_classes, name='fc_id_2048_1')

                zp2 = layers.avg_pool2d(p2, [12, 8], [12, 8],scope='avg_pool_zp2')
                print("zp2", zp2)
                z0_p2 = tf.slice(zp2, [0, 0, 0, 0], [-1, 1, -1, -1])
                z1_p2 = tf.slice(zp2, [0, 1, 0, 0], [-1, 1, -1, -1])
                print('z0_p2:', z0_p2)
                f0_p2 = tf.squeeze(tf.squeeze(self.reduction(z0_p2, "2"), squeeze_dims=1), squeeze_dims=1)
                f1_p2 = tf.squeeze(tf.squeeze(self.reduction(z1_p2, "3"), squeeze_dims=1), squeeze_dims=1)
                l0_p2 = self.fully_connect(f0_p2, num_classes, name='fc_id_256_1_0')
                l1_p2 = self.fully_connect(f1_p2, num_classes, name='fc_id_256_1_1')
                print('l0_p2:', l0_p2)
                print('l1_p2:', l1_p2)
                predict.extend([fg_p2, f0_p2, f1_p2])
                triplet_losses.append(fg_p2)
                softmax_losses.extend([l_p2, l0_p2, l1_p2])

        if model in {'mgn', 'p3_single'}:
            with tf.variable_scope("p3"):
                p3 = self.general_branch(backbone, downsample=False)
                print('p3:', p3)
                zg_p3 = layers.avg_pool2d(p3, [24, 8], scope='avg_pool_zg_p3')
                print('zg_p3:', zg_p3)
                fg_p3 = tf.squeeze(tf.squeeze(self.reduction(zg_p3, name="4"), squeeze_dims=1), squeeze_dims=1)
                print('fg_p3:', fg_p3)
                l_p3 = self.fully_connect(fg_p3, num_classes, name='fc_id_2048_2')
                print('l_p3:', l_p3)
                zp3 = layers.avg_pool2d(p3, [8, 8], [8, 8],scope='avg_pool_zp3')
                print("zp3", zp3)
                z0_p3 = tf.slice(zp3, [0, 0, 0, 0], [-1, 1, -1, -1])  # z_p0^P3
                z1_p3 = tf.slice(zp3, [0, 1, 0, 0], [-1, 1, -1, -1])  # z_p1^P3
                z2_p3 = tf.slice(zp3, [0, 2, 0, 0], [-1, 1, -1, -1])  # z_p2^P3
                f0_p3 = tf.squeeze(tf.squeeze(self.reduction(z0_p3, name="5"), squeeze_dims=1), squeeze_dims=1)
                f1_p3 = tf.squeeze(tf.squeeze(self.reduction(z1_p3, name="6"), squeeze_dims=1), squeeze_dims=1)
                f2_p3 = tf.squeeze(tf.squeeze(self.reduction(z2_p3, name="7"), squeeze_dims=1), squeeze_dims=1)
                l0_p3 = self.fully_connect(f0_p3, num_classes, name='fc_id_256_2_0')
                l1_p3 = self.fully_connect(f1_p3, num_classes, name='fc_id_256_2_1')
                l2_p3 = self.fully_connect(f2_p3, num_classes, name='fc_id_256_2_2')
                print('l0_p3:', l0_p3)
                print('l1_p3:', l1_p3)
                print('l2_p3:', l2_p3)
            predict.extend([fg_p3, f0_p3, f1_p3, f2_p3])
            triplet_losses.append(fg_p3)
            softmax_losses.extend([l_p3, l0_p3, l1_p3, l2_p3])

        predict = tf.concat(predict, 1, name="person_embedding")
        return predict, triplet_losses, softmax_losses

    def loss(self, input, target, margin):
        # sess = tf.Session()
        # target shape [64]
        y_true = tf.expand_dims(target, dim=-1)
        print('y true:', y_true)
        # y_true shape [64, 1]
        # y_true.t() shape [1, 64]
        # same_id shape [64, 64], 矩阵中对角线上值为1
        same_id = tf.cast(tf.equal(y_true, tf.transpose(y_true)), tf.float32)
        print('same id:', same_id)
        # same id矩阵中 若[2,3]位置为1，代表labels中第2个值和第3个值得label相同
        pos_mask = same_id
        neg_mask = 1 - same_id
        print('pos_mask:', pos_mask)
        print('neg_mask:', neg_mask)

        def _mask_max(input_tensor, mask, axis=None, keepdims=False):
            # mask = 1(相同) 距离=input_tensor - 0
            # mask = 0(不同) 距离=input_tensor - 1
            # 距离最远的pos对
            input_tensor = input_tensor - 1e6 * (1 - mask)
            _max = tf.reduce_max(input_tensor, axis=axis, keepdims=keepdims)
            return _max

        def _mask_min(input_tensor, mask, axis=None, keepdims=False):
            input_tensor = input_tensor + 1e6 * (1 - mask)
            _min = tf.reduce_min(input_tensor, axis=axis, keepdims=keepdims)
            return _min

        dist_squared = tf.reduce_sum(input ** 2, axis=1, keep_dims=True) + \
                       tf.reduce_sum(tf.transpose(input) ** 2, axis=0, keep_dims=True) - \
                       2.0 * tf.matmul(input, tf.transpose(input))
        dist = tf.maximum(dist_squared, 1e-16)
        # tmp_b = tf.cast(tf.less(dist_squared, tf.constant(1e-16)), tf.int32)
        # tmp_c = 1 - tmp_b
        # tmp_b = tf.multiply(tf.cast(tmp_b, tf.float32),1e-16)
        # tmp_c = tf.multiply(dist_squared, tf.cast(tmp_c, tf.float32))
        dist = tf.sqrt(dist)

        pos_max = _mask_max(dist, pos_mask, axis=-1)
        # print('pos_max:', sess.run(pos_max))
        neg_min = _mask_min(dist, neg_mask, axis=-1)
        # print('neg_min:', sess.run(neg_min))

        # loss(x, y) = max(0, -y * (x1 - x2) + margin)
        basic_loss = tf.add(tf.subtract(pos_max, neg_min), margin)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        # print('loss:', sess.run(loss))
        return loss