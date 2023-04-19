from typing import List

import time
import numpy as np
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
import onnxruntime


class BertONNXEncoder(object):
    
    def __init__(self,
                 onnx_model_file: str,
                 vocab_path: str,
                 max_len: int = 16):
        self.tokenizer_ = Tokenizer(vocab_path, do_lower_case=True)
        self.max_len_ = max_len
        self.onnx_model_ = onnxruntime.InferenceSession(onnx_model_file)
        self.vocab_path = vocab_path
        self.onnx_model_file = onnx_model_file

    def __token_encode__(self, text):
        ids, segs = self.tokenizer_.encode(text, maxlen=self.max_len_)
        ids = sequence_padding([ids], length=self.max_len_).astype(np.float32)
        segs = sequence_padding([segs], length=self.max_len_).astype(np.float32)
        return ids, segs

    def __batch_token_encode__(self, texts, batch_size=1024):
        ids, segs = [], []
        n = len(texts)
        for i, text in enumerate(texts, 1):
            ids_, segs_ = self.tokenizer_.encode(text, maxlen=self.max_len_)
            ids.append(ids_)
            segs.append(segs_)
            if len(ids) == batch_size or i == n:
                ids = sequence_padding(ids, length=self.max_len_).astype(np.float32)
                segs = sequence_padding(segs, length=self.max_len_).astype(np.float32)
                yield ids, segs
                ids, segs = [], []

    def encode(self, text: str, verbose=True):
        """单文本embedding
        """
        token, segment = self.__token_encode__(text)
        ort_inputs = {self.onnx_model_.get_inputs()[0].name: token, self.onnx_model_.get_inputs()[1].name: segment}
        vec = self.onnx_model_.run(None, ort_inputs)[0]
        return vec

    def batch_encode(self, texts: List[str], batch_size=1024, verbose=True):
        """多文本批量embedding
        """
        vecs_list = []
        for token, segment in self.__batch_token_encode__(texts, batch_size):
            ort_inputs = {self.onnx_model_.get_inputs()[0].name: token, self.onnx_model_.get_inputs()[1].name: segment}
            vecs = self.onnx_model_.run(None, ort_inputs)[0]
            vecs_list.append(vecs) 
        vecs = np.vstack(vecs_list)
        return vecs
