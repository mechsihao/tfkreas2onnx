from typing import List

import time
import numpy as np
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from core.encoder.vecs_whitening import VecsWhitening


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    graph = tf.Graph()

    def _imports_graph_def():
        tf.graph_util.import_graph_def(graph_def, name="")

    with graph.as_default():
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    if print_graph:
        print("-" * 50)
        print("Frozen model layers: ")
        layers = [op.name for op in import_graph.get_operations()]
        for layer in layers:
            print(layer)
        print("-" * 50)
    return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs),
                                tf.nest.map_structure(import_graph.as_graph_element, outputs))


def pb_file_to_concrete_function(pb_file, inputs, outputs, print_graph=False):
    """
    pb_file 转 concrete function
    :param pb_file:
    :param inputs:
    :param outputs:
    :param print_graph:
    :return:
    """
    with tf.io.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                        inputs=inputs,
                                        outputs=outputs,
                                        print_graph=print_graph)
    return graph_def, frozen_func


class BertFreezeEncoder(object):
    """
    为了加速线上推理，离线生产的tf/keras/pytorch版本的bert模型，均可以通过此方法加载
    需要提前约定好输入list、输出list、词典、maxlen
    """
    def __init__(self,
                 pb_model_file: str,
                 vocab_path: str,
                 whitening_dim: int = None,
                 whitening_path: str = None,
                 input_variable_names: List[str] = None,
                 output_variable_names: List[str] = None,
                 max_len: int = 16):
        """加载onnx存储的pb模型文件，并且封装encoder方法，原生支持白化操作
        """
        input_variable_list = [f"{i}:0" for i in input_variable_names] if input_variable_names else ['x0:0', 'x1:0']
        output_variable_list = [f"{i}:0" for i in output_variable_names] if output_variable_names else ['y0:0']
        graph, onnx_model = pb_file_to_concrete_function(pb_model_file, input_variable_list, output_variable_list)
        self.tokenizer_ = Tokenizer(vocab_path, do_lower_case=True)
        self.max_len_ = max_len
        self.onnx_model_ = onnx_model
        self.graph_ = graph
        self.vw_model_ = VecsWhitening(whitening_dim).load_bw_model(whitening_path) if whitening_path else None
        self.vecs_not_whitening_ = None
        self.vocab_path = vocab_path
        self.whitening_path = whitening_path
        self.whitening_dim = whitening_dim
        self.input_variable_list = input_variable_names
        self.output_variable_list = output_variable_names
        self.pb_model_file = pb_model_file
        return self

    def __token_encode__(self, text):
        ids, segs = self.tokenizer_.encode(text, maxlen=self.max_len_)
        return tf.cast([ids], tf.float32), tf.cast([segs], tf.float32)

    def __batch_token_encode__(self, texts, batch_size=1024):
        ids, segs = [], []
        n = len(texts)
        for i, text in enumerate(texts, 1):
            ids_, segs_ = self.tokenizer_.encode(text, maxlen=self.max_len_)
            ids.append(ids_)
            segs.append(segs_)
            if len(ids) == batch_size or i == n:
                ids = sequence_padding(ids)
                segs = sequence_padding(segs)
                yield tf.cast(ids, tf.float32), tf.cast(segs, tf.float32)
                ids, segs = [], []

    def encode(self, text: str, verbose=True):
        """单文本embedding
        """
        start = time.time()
        token, segment = self.__token_encode__(text)
        vec = self.onnx_model_(token, segment)[0].numpy()
        start_vw = time.time()
        if verbose:
            print(f"Bert encode cost: {start_vw - start:.8f}s")
        if self.vw_model_:
            vec = self.vw_model_.transform(vec)
            if verbose:
                print(f"Whitening cost: {time.time() - start_vw:.8f}s")
        return vec

    def batch_encode(self, texts: List[str], batch_size=1024, verbose=True):
        """多文本批量embedding
        """
        start = time.time()
        vecs_list = []
        for token, segment in self.__batch_token_encode__(texts, batch_size):
            vecs_ = self.onnx_model_(token, segment)[0].numpy()
            vecs_list.append(vecs_)
        self.vecs_not_whitening_ = np.vstack(vecs_list)  # 这一步的目的是方便拿出未白化前的向量，方便后续做处理
        vecs = self.vecs_not_whitening_
        start_batch_vw = time.time()
        if verbose:
            print(f"Bert batch encode cost: {start_batch_vw - start:.8f}s")
        if self.vw_model_:
            vecs = self.vw_model_.transform(vecs)
            if verbose:
                print(f"Whitening batch transform cost: {time.time() - start_batch_vw:.8f}s")
        return vecs
