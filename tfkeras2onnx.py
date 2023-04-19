import os

import tensorflow as tf
from tensorflow.keras import layers, models
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer


if __name__ == "__main__":
    max_len = 8
    model_name = "roberta"
    save_root = "onnx_model_root"

    conf = f"{model_name}/small/bert_config.json"
    ckpt = f"{model_name}/small/bert_model.ckpt"
    vocab = f"{model_name}/small/vocab.txt"

    tokenizer = Tokenizer(vocab, do_lower_case=True)

    base = build_transformer_model(conf, ckpt)
    output = layers.Lambda(lambda tensor: tensor[:, 0], name='bert_encoder')(base.output)
    model = models.Model(base.inputs, output)

    ind_input, seg_input = tf.keras.layers.Input([max_len]), tf.keras.layers.Input([max_len])
    model_new = tf.keras.Model(inputs=[ind_input, seg_input], outputs=model([ind_input, seg_input]))

    # 这里保存的文件名必须是xxxx/saved_model.pb，tf2onnx只支持这样的名称转换
    tf.keras.models.save_model(model_new, f"{model_name}_onnx/saved_model.pb")

    # 这一步操作也可以在终端中操作，需要事先安装tf2onnx，将其中的变量替换即可：
    # python -m tf2onnx.convert --saved-model "{model_name}_onnx/saved_model.pb" --output "{save_root}/{model_name}.onnx"
    os.system(f'python -m tf2onnx.convert --saved-model "{model_name}_onnx/saved_model.pb" --output "{save_root}/{model_name}.onnx"')
