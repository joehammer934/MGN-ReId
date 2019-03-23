"""
@Time   : 2019/3/19
@Author : Li YongHong
@Email  : lyh_robert@163.com
@File   : export_tf_serving_with_pre_post.py
"""
import os
import glog as log
import tensorflow as tf
flags = tf.flags
flags.DEFINE_string("pb_path", "model_pre_and_post.pb", "model path")
flags.DEFINE_string("export_path","./serving_model","")
FLAGS = flags.FLAGS

pb_path = FLAGS.pb_path
export_path = FLAGS.export_path

def main(_):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session() as sess:
            #get tensor by tensor_name
            input_image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            result_embedding = tf.get_default_graph().get_tensor_by_name('result_embedding:0')
            print('........', input_image_tensor)
            # 将模型保存为可用于线上服务的文件（一个.pb文件，一个variables文件夹）
            if os.path.exists(export_path):
                os.rmdir(export_path)
            log.info('Exporting trained model to', export_path)

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # 建立签名映射
            """
            build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
            输入：tensorflow graph中的tensor；
            输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
            """
            input_image = tf.saved_model.utils.build_tensor_info(input_image_tensor)
            reid_embedding = tf.saved_model.utils.build_tensor_info(result_embedding)

            """
            signature_constants：SavedModel保存和恢复操作的签名常量。

            如果使用默认的tensorflow_model_server部署模型，
            这里的method_name必须为signature_constants中CLASSIFY,PREDICT,REGRESS的一种
            """

            # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
            reid_mgn_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "input_image": input_image
                    },
                    outputs={
                        "reid_embedding": reid_embedding
                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            """
            tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
            """
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            """
            add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                          输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                          对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                          对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
            """
            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={
                    "reid_mgn_serving" : reid_mgn_signature
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            log.info("Build Done")
if __name__ == '__main__':
    tf.app.run()