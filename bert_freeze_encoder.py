from typing import List

import time
import numpy as np
import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding


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
    def __init__(self,
                 pb_model_file: str,
                 vocab_path: str,
                 input_variable_names: List[str] = None,
                 output_variable_names: List[str] = None,
                 max_len: int = 16):
        
        input_variable_list = [f"{i}:0" for i in input_variable_names] if input_variable_names else ['x0:0', 'x1:0']
        output_variable_list = [f"{i}:0" for i in output_variable_names] if output_variable_names else ['y0:0']
        graph, onnx_model = pb_file_to_concrete_function(pb_model_file, input_variable_list, output_variable_list)
        self.tokenizer_ = Tokenizer(vocab_path, do_lower_case=True)
        self.max_len_ = max_len
        self.onnx_model_ = onnx_model
        self.graph_ = graph
        self.vocab_path = vocab_path
        self.input_variable_list = input_variable_names
        self.output_variable_list = output_variable_names
        self.pb_model_file = pb_model_file

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
        return vec

    def batch_encode(self, texts: List[str], batch_size=1024, verbose=True):
        """多文本批量embedding
        """
        vecs_list = []
        for token, segment in self.__batch_token_encode__(texts, batch_size):
            vecs_ = self.onnx_model_(token, segment)[0].numpy()
            vecs_list.append(vecs_)
        vecs = np.vstack(vecs_list)  # 这一步的目的是方便拿出未白化前的向量，方便后续做处理
        return vecs
