import os

import tensorflow as tf
from tensorflow.keras import layers, models
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

from tensorflow.python.framework import importer
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def freeze_keras_model2pb(keras_model, pb_filepath, input_variable_name_list=None, output_variable_name_list=None):
    """
    karas 模型转pb，BertOnnxEncoder加载的keras模型需要用该函数先转换后才能加载
    :param keras_model: 待转换模型
    :param pb_filepath: 模型pb文件保存路径
    :param input_variable_name_list: 输入变量名称列表
    :param output_variable_name_list: 输出变量名称列表
    :return:
    """
    assert hasattr(keras_model, 'inputs'), "the keras model must be built with functional api or sequential"
    # save pb
    if input_variable_name_list is None:
        input_variable_name_list = list()
    if output_variable_name_list is None:
        output_variable_name_list = list()

    if len(input_variable_name_list) == len(keras_model.inputs):
        input_variable_list = input_variable_name_list
    else:
        input_variable_list = ['x%d' % i for i in range(len(keras_model.inputs))]

    input_func_signature_list = [
        tf.TensorSpec(item.shape, dtype=item.dtype, name=name) for name, item in zip(input_variable_list, keras_model.inputs)]
    full_model = tf.function(lambda *x: keras_model(x, training=False))
    # To obtain an individual graph, use the get_concrete_function method of the callable created by tf.function.
    # It can be called with the same arguments as func and returns a special tf.Graph object
    concrete_func = full_model.get_concrete_function(input_func_signature_list)
    # Get frozen ConcreteFunction
    frozen_graph = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_graph.graph.as_graph_def()
    out_idx = 0
    for node in graph_def.node:
        node.device = ""
        if node.name.startswith('Identity'):
            out_idx += 1
    if len(output_variable_name_list) == out_idx:
        output_variable_list = output_variable_name_list
    else:
        output_variable_list = ['y%d' % i for i in range(out_idx)]
    out_idx = 0
    for node in graph_def.node:
        node.device = ""
        if node.name.startswith('Identity'):
            node.name = output_variable_list[out_idx]
            out_idx += 1
    new_graph = tf.Graph()
    with new_graph.as_default():
        importer.import_graph_def(graph_def, name="")

    return tf.io.write_graph(graph_or_graph_def=new_graph,
                             logdir=os.path.dirname(pb_filepath),
                             name=os.path.basename(pb_filepath),
                             as_text=False), input_variable_list, output_variable_list


if __name__ == "__main__":
    max_len = 16
    model_name = "roberta"

    conf = f"{model_name}/small/bert_config.json"
    ckpt = f"{model_name}/small/bert_model.ckpt"
    vocab = f"{model_name}/small/vocab.txt"

    tokenizer = Tokenizer(vocab, do_lower_case=True)

    base = build_transformer_model(conf, ckpt)
    output = layers.Lambda(lambda tensor: tensor[:, 0], name='bert_encoder')(base.output)
    model = models.Model(base.inputs, output)

    ind_input, seg_input = tf.keras.layers.Input([max_len]), tf.keras.layers.Input([max_len])
    model_new = tf.keras.Model(inputs=[ind_input, seg_input], outputs=model([ind_input, seg_input]))

    # 存储pb文件，方便下次使用，
    # 这里的input_vaiable_list, ouput_vaiable_list也在调用模型的时候需要用到
    _, input_vaiable_list, ouput_vaiable_list = freeze_keras_model2pb(model_new, "roberta.pb")
