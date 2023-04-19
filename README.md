# tf/keras模型线上推理加速方法： 
## 1.tf2onnx转换tf/keras模型至onnx格式，用onnxruntime来加速在线推理
### freeze转换方法及模型使用方法
 - 参数freeze：
    - freeze_pb.py文件，按照其中的示例来转换出模型的pb文件
    - 需要将模型用keras的model包一层，输入输出的长度固定死，然后再用freeze_keras_model2pb转换
    - 加载保存模型方法，见bert_freeze_encoder.py
 - 使用onnxruntime：
    - 示例：tfkeras2onnx.py文件，需要事先安装tf2onnx，如下：
    ```shell 
    pip install tf2onnx
    ```
    - 需要首先将模型存储成pb文件，保存的文件名必须是xxxx/saved_model.pb，tf2onnx只支持这样的名称转换
    - 然后使用如下方法转换即可：
    ```shell
    python -m tf2onnx.convert --saved-model "your_onnx_model_name/saved_model.pb" --output "save_root/your_model.onnx"
    ```
    - 加载保存模型方法，见bert_onnx_encoder.py

## 3.将tf/keras参数冻结，tf加载静态图推理
这里用到的bert是苏剑林开源的roberta small，框架用的也是苏神的bert4keras，需要事先安装：
```shell
pip install bert4keras==0.10.8
```
#### 压测实验效果：
用上面的两种方法将一个4层bert模型分别转换后压测实验，实验方法是用二者分别对相同的1w条query进行推理，查看压测时间，效果如下：
| 优化方法  | P90/s | P95/s | P99/s | P999/s | 
| --- | :---: | :---: | :---: | :---: |
| onnxruntime | 0.00549 | 0.00672 | 0.00932 | 0.07780 | 
| freeze参数 | 0.00349 | 0.00373 | 0.00439 | 0.00897 | 
#### 结论： 
- 从压测来看，效果好像是freeze参数更好一些
- 二者相同点是大部分推理耗时都在2.5ms以内
- 但`onnxruntime`有个优点，有75%的推理耗时在3ms以内，而 `freeze参数` 则是只有60%的推理耗时在3ms以内
- 个人推测onnxruntime效果差的原因可能是笔者用的cpu比较垃圾，导致前面的计算阻塞了，从而后面的计算堆积了起来。在真实场景下效果应该会比freeze参数稳定。