"""
@Time   : 2019/3/16
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : config.py
"""
#input image
image_height = 384
image_width = 128
channel = 3

#learning rate config
init_learning_rate = 0.01
momentum = 0.9
weight_decay = 5e-4
num_epoch=100000
lr_steps =[40, 60]
learning_rate = [0.01, 0.001, 0.0001]


batch_id = 16
batch_image = 4
batch_test = 32
num_class = 751