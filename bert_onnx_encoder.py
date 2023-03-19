class BertOnnxEncoder(object):
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
                 max_len: int = 16,
                 max_cache_len: int = None):
        """加载onnx存储的pb模型文件，并且封装encoder方法，原生支持白化操作
        """
        input_variable_list = [f"{i}:0" for i in input_variable_names] if input_variable_names else ['x0:0', 'x1:0']
        output_variable_list = [f"{i}:0" for i in output_variable_names] if output_variable_names else ['y0:0']
        graph, onnx_model = pb_file_to_concrete_function(pb_model_file, input_variable_list, output_variable_list)
        self.tokenizer = Tokenizer(vocab_path, do_lower_case=True)
        self.max_len = max_len
        self.onnx_model = onnx_model
        self.graph = graph
        self.max_cache_len = max_cache_len
        self.vw_model = self.__init_whitening_model(whitening_dim, whitening_path)
        self.lru_encoder = self.__inti_lru_encoder(max_cache_len)
        self.vecs_not_whitening = None

    @staticmethod
    def __init_whitening_model(dim, whitening_path):
        vw_model = VecsWhitening(dim)
        if whitening_path:
            vw_model.load_bw_model(whitening_path)
        else:
            vw_model = None
        return vw_model

    def __inti_lru_encoder(self, max_cache_len):
        """用一条样本激活lru_encoder，因为lru_encoder第一条encode非常慢，构建好缓存后速度恢复正常
        """
        @lru_cache(max_cache_len, typed=False)
        def lru_encoder_(text: str, verbose: bool = True) -> ndarray:
            """增加cache的单文本embedding，将高频且近期使用的query存储在缓存中，方便下次调用
            """
            res = self.encode(text, verbose)
            return res
        lru_encoder_("欢迎使用Bert ONNX Encoder", False)
        return lru_encoder_

    def __token_encode__(self, text):
        ids, segs = self.tokenizer.encode(text, maxlen=self.max_len)
        return tf.cast([ids], tf.float32), tf.cast([segs], tf.float32)

    def __batch_token_encode__(self, texts, batch_size=1024):
        ids, segs = [], []
        n = len(texts)
        for i, text in enumerate(texts, 1):
            ids_, segs_ = self.tokenizer.encode(text, maxlen=self.max_len)
            ids.append(ids_)
            segs.append(segs_)
            if len(ids) == batch_size or i == n:
                ids = sequence_padding(ids)
                segs = sequence_padding(segs)
                yield tf.cast(ids, tf.float32), tf.cast(segs, tf.float32)
                ids, segs = [], []

    def encode(self, text, verbose=True):
        """单文本embedding
        """
        start = time.time()
        token, segment = self.__token_encode__(text)
        vec = self.onnx_model(token, segment)[0].numpy()
        start_vw = time.time()
        if verbose:
            print(f"Bert encode cost: {start_vw - start:.8f}s")
        if self.vw_model:
            vec = self.vw_model.transform(vec)
            if verbose:
                print(f"Whitening cost: {time.time() - start_vw:.8f}s")
        return vec

    def batch_encode(self, texts, batch_size=1024):
        """多文本批量embedding
        """
        start = time.time()
        vecs_list = []
        for token, segment in self.__batch_token_encode__(texts, batch_size):
            vecs_ = self.onnx_model(token, segment)[0].numpy()
            vecs_list.append(vecs_)
        self.vecs_not_whitening = np.vstack(vecs_list)  # 这一步的目的是方便拿出未白化前的向量，方便后续做处理
        vecs = self.vecs_not_whitening
        start_batch_vw = time.time()
        print(f"Bert batch encode cost: {start_batch_vw - start:.8f}s")
        if self.vw_model:
            vecs = self.vw_model.transform(vecs)
            print(f"Whitening batch transform cost: {time.time() - start_batch_vw:.8f}s")
        return vecs

    def lru_encode(self, text, verbose=True):
        start = time.time()
        res = self.lru_encoder(text, verbose)
        print(f"Lru encode cost: {time.time() - start:.8f}s")
        return res

    def clear_cache(self):
        """清空缓存"""
        self.lru_encoder.cache_clear()

    def reload(self,
               pb_model_file: str = None,
               vocab_path: str = None,
               whitening_dim: int = None,
               whitening_path: str = None,
               input_variable_names: List[str] = None,
               output_variable_names: List[str] = None,
               max_len: int = None,
               max_cache_len: int = None):
        """
        重载相关核心文件，init中的参数哪些改动了就传入哪些即可
        """
        if pb_model_file:
            input_variable_list = input_variable_names or [f'x{i}:0' for i in range(2)]
            output_variable_list = output_variable_names or [f'y{i}:0' for i in range(1)]
            graph, onnx_model = pb_file_to_concrete_function(pb_model_file, input_variable_list, output_variable_list)
            self.onnx_model = onnx_model
            self.graph = graph

        if whitening_dim and whitening_path:
            self.vw_model = self.__init_whitening_model(whitening_dim, whitening_path)

        if vocab_path:
            self.tokenizer = Tokenizer(vocab_path, do_lower_case=True)

        if max_len:
            self.max_len = max_len

        self.clear_cache()

        if max_cache_len:
            self.lru_encoder = self.__inti_lru_encoder(max_cache_len)

        return self
