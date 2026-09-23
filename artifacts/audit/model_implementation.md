# 鲁棒预测模型实现审计片段

## 范围与文件

本实现只新增/修改以下职责范围内文件：

- `src/data/dataset.py`、`src/data/__init__.py`：三类对齐数据统一加载、`text_bert`校验、padding/内部缺失掩码、BERT文本特征复现与缓存。
- `src/models/multitask.py`、`src/models/__init__.py`：单模态/concat基线、mask-aware gated multimodal multi-task模型。
- `src/training/augment.py`、`losses.py`、`trainer.py`、`cli.py`、`__init__.py`：连续span缺失增强、一致性损失、确定性训练、checkpoint。
- `src/evaluation/metrics.py`、`cli.py`、`predict_special.py`、`__init__.py`：Accuracy、Macro-F1、Weighted-F1、MAE、Pearson及评估/专项预测CLI。
- `configs/model_single_text.yaml`、`model_concat.yaml`、`model_gated.yaml`：轻量可控配置。
- `tests/test_data.py`、`test_models.py`、`test_training.py`：数据、掩码、模型、增强、损失及指标测试。

开始实现时上述目标目录为空；未覆盖之后由其他任务写入的 `src/explain`、`src/utils`与`tests/test_statistics.py` 等文件。

## 核心接口

- `load_records(path, split=None, text_provider=None)`：统一读取对齐特征集分割字典、模态缺失评测集的`{"test": ...}`包装结构和可解释性评测集逐样本直接字典。
- `MultimodalDataset(...)`：输出`features`、`observed`、`valid`、`text_bert`、任务标签和ID。
- `BertTextProvider.encode(text_bert)`：用同一`google-bert/bert-base-uncased`最后隐层复现缺失的768维文本特征并在`cache/bert_text`按token哈希缓存。
- `build_model(config)`：构建`single`、`concat`或`gated`模型。
- `Trainer.train_epoch/evaluate/predict/save`：训练、评估、无标签预测和checkpoint接口。

训练示例：

```bash
env/run.sh env/venv/bin/python -m src.training.cli \
  --config configs/model_gated.yaml --device cpu
```

评估示例：

```bash
env/run.sh env/venv/bin/python -m src.evaluation.cli \
  --checkpoint artifacts/models/gated_multitask.pt \
  --data "data/features_aligned/aligned_50.pkl" \
  --split test --device cpu
```

两个评测集无标签专项预测示例：

```bash
env/run.sh env/venv/bin/python -m src.evaluation.predict_special \
  --checkpoint artifacts/models/gated_multitask.pt \
  --input "<对齐版本目录或单个pkl>" \
  --output artifacts/predictions/special.csv \
  --device cpu --bert-device cuda --local-files-only
```

## 数据与文本选择证据

1. `aligned_50.pkl`为主数据，形状为：文本`[N,50,768]`、音频`[N,50,74]`、视觉`[N,50,35]`、token三元组`[N,3,50]`。
2. 模态缺失评测集共30个对齐样本，`text_bert`存储为`float32`；逐样本检查均有限且严格为整数，加载后才安全转换为`int64`。模态缺失评测集没有`text`字段。
3. 可解释性评测集共20个对齐样本，是逐文件直接字典而非split字典，且包含已有`text`。其pickle由NumPy 2.x生成，加载器为当前NumPy 1.x运行时提供了模块名兼容层。
4. 已从镜像获取/缓存`google-bert/bert-base-uncased`并用对齐特征集前两个样本核验各层输出。最后一层与对齐特征集给定`text`的MAE为`4.85e-7`、RMSE为`6.32e-7`、最大误差为`9.06e-6`、相关系数约`0.999999999999`；其他层不匹配。因此模态缺失评测集使用相同token三元组和同一BERT最后隐层复现，不使用伪造文本特征，对齐特征集与模态缺失评测集保持同分布。
5. 对齐特征集与可解释性评测集优先使用给定`text`以避免重复计算；模态缺失评测集复现结果按模型名和token内容哈希缓存。

## 掩码与标签安全

- `valid`（非padding）严格来自`text_bert`的attention mask。
- 音频/视觉`observed = valid AND 全维有限 AND 非全零`；所以padding与有效区间内的全零缺失span被明确区分，即使padding位置意外非零也不会被视为观测。
- 文本观测掩码来自attention mask；BERT特征不以数值是否为零判断。
- 连续span增强只在`valid`区域采样，覆盖文本、音频和视觉；直接将对应连续帧特征置零并同步更新各自`observed`，所以文本遮蔽会真实进入编码器，而不只是修改未参与前向计算的token。padding位置和值均不改变。
- 两个评测集没有标准标签字段时统一填入忽略值`-100/NaN`；专项预测仅读取特征，不读取文件名、ID、原文或任何外部专项标签来推断真值。

## 模型与目标

- 基线：配置`modalities: [text]`即单模态；`[text,audio,vision]`即concat。
- 鲁棒模型：每个模态独立轻量MLP时序编码和masked mean；门控输入包括模态表示、观测覆盖率与可用性；完全缺失模态使用学习型missing token且在门控softmax中屏蔽。配置支持文本/音频/视觉的任一单模态、任意双模态和三模态组合。
- 多任务头：三分类logits；回归通过`3*tanh`严格限制在`[-3,3]`。
- 损失：交叉熵 + Smooth-L1 + 增强前后分类KL/回归MSE一致性损失；权重均由YAML控制。
- checkpoint只按验证集固定联合准则选择：`0.35*Macro-F1 + 0.15*Accuracy + 0.25*((Pearson+1)/2) + 0.25*(1-clip(MAE,0,6)/6)`。四项均在`[0,1]`；权重在配置中固定且不看test调整。Macro-F1获最高权重以应对类别不均衡，分类/回归两任务合计权重各0.5，既保留准确率又同时约束趋势和绝对误差。每个epoch在history记录`selection_score`。
- 随机种子固定为2026；DataLoader shuffle显式使用固定generator。
- 带`--max-samples`的烟测自动输出到`runs/smoke_<model>_<N>/artifacts`，不会覆盖正式配置的checkpoint/metrics路径。

## 已运行验证

- `python -m compileall -q src tests`：通过。
- `python -m pytest -q`：最终`40 passed in 7.46s`。新增测试覆盖文本/音频/视觉连续段遮蔽、padding边界、全部1/2/3模态组合、联合选择准则归一化/权重校验。
- CPU训练烟测：gated模型，对齐特征集小样本12 epoch，完成并生成隔离的checkpoint及训练指标；此结果仅证明链路可运行，不代表正式性能。验证确认history每个epoch均含`selection_score`，且烟测未创建/覆盖正式产物路径。
- CPU评估烟测：对齐特征集 test前8条，Accuracy `0.625`、Macro-F1 `0.2564`、Weighted-F1 `0.4808`、MAE `0.5906`、Pearson `0.2819`；小样本结果不可用于论文结论。
- 模态缺失评测集全30个对齐文件专项预测：成功，文本由BERT重建；输出31行（含表头）。
- 可解释性评测集全20个对齐文件专项预测：成功，使用给定BERT文本特征；输出21行（含表头）。

## 风险与后续使用注意

- Pickle反序列化只能用于可信的本地数据文件，不应加载未知来源文件。
- 模态缺失评测集首次文本复现依赖本地BERT缓存或可达镜像；可用`--local-files-only`确保运行时不访问网络。复现较慢但后续命中工程缓存。
- 全零作为音频/视觉内部缺失的判据符合本批数据结构；若未来真实观测可能整帧严格全零，应提供显式观测掩码以消除歧义。
- 当前checkpoint与CSV均来自8条样本烟测，专用于接口验证。正式实验必须移除`--max-samples`、按验证集选模并保存独立随机种子/消融结果。
- 默认未做冗长正式训练；GPU运行应始终通过项目`env/run.sh`使用匹配驱动库。
