本篇主要内容是wav2vec模型的代码笔记、工程向的阅读笔记
源码地址：https://github.com/pytorch/fairseq/tree/main/examples/wav2vec

# 1.模型构建
函数调用顺序：fairseq/train.py--->fairseq/fairseq_cli/train.py(此处进行迭代训练、模型构建)--->fairseq/fairseq/task/audio_pretrainning.py--->fairseq/fairseq/task/fairseq_task.py--->fairseq/fairseq/models/wav2vec/wav2vec.py

好好缕一缕函数调用顺序。
表层的函数调用（包括init.py这种）
里层的函数调用

# 2.模型训练
从fairseq/fairseq/models/wav2vec/wav2vec.py/Class Wav2VecModel看训练的步骤。
```
features = self.feature_extractor(source) # 对应论文，encoder network
if self.vector_quantizer:
    q_res = self.vector_quantizer(features)
    features = q_res["x"]
    for k in q_res.keys():
        if k != "x":
            result[k] = q_res[k]

x = self.dropout_feats(features)
x = self.feature_aggregator(x) # 对应论文，context network
x = self.dropout_agg(x)

if self.project_features is not None:
    features = self.project_features(features)
x, targets = self.wav2vec_predictions(x, features) # 对应论文objective，前半截公式，不是loss!
result["cpc_logits"] = x
result["cpc_targets"] = targets
```
# 3.模型结构

```
encoder network(raw audio) ---> low frequency feat
context network(low frequency feat) ---> contextualized tensor

loss：是怎么计算的呢？
```



