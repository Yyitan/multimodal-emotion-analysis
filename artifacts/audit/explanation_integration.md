# 可解释预测真实模型集成审计

## 1. 冻结模型与选择依据

正式解释使用 `artifacts/models.pt`，运行前、运行后均校验 SHA-256：

`989b84b5e507a97186179e80e0f7f8c8e29714f3c84a84aa1d60b3a2ee5bb601`

模型选择仅依据对齐特征集 valid 固定联合分数，依据文件为 `artifacts/model_card.json`。三个 gated 种子的 valid 分数分别为：seed 2026 = 0.7396859205、seed 3407 = 0.7371254045、seed 20260923 = 0.7320056857，因此选择 seed 2026、best epoch 2。两个评测集的标签、ID、文本或特征匹配均未参与选择。

冻结模型 valid 基础指标：Accuracy 0.620879、Macro-F1 0.609382、Weighted-F1 0.624585、MAE 0.585749、Pearson 0.661413。

## 2. 模型适配与解释目标

`src/explain/model_adapter.py` 加载真实 gated checkpoint。每次联盟或遮挡预测均同时执行：

1. 被移除模态特征置零；
2. 被移除位置的 observed mask 置 false；
3. 与原始 observed mask、valid mask、有限值和非零行再次求交；
4. 模型保持 eval/inference 模式。

因此不存在“值已置零但门控仍认为模态可用”的伪联盟。

分类解释在完整输入上确定最终预测类别，随后所有联盟、局部遮挡和删除实验固定选择该类别的 probability，命名为 `classification_predicted_probability` 或详细结果中的 `predicted_class_probability:<class>`。回归解释固定选择模型 regression value，命名为 `regression_value`。两者分别计算三模态精确 Shapley；可解释性评测集分类结果进入33列CSV和逐样本detail JSON，回归Shapley独立保存，避免混淆两个任务。

## 3. 对齐特征集 valid 忠实性评估

对 valid 全部 728 条样本真实推理，分类和回归各一条评估记录，共1456条。随机删除50次，seed固定为20260923；轻微扰动为各模态观测值标准差的1%高斯噪声。正式评估在主训练结束后使用 `CUDA_VISIBLE_DEVICES=3`、项目 `env/run.sh` 和 `cuda:0` 完整重跑；此前未落盘的CPU进程已停止，未并行写正式输出。

分类固定预测类别概率：

- Top evidence 删除 AOPC：0.047663；随机删除 AOPC：0.003179；增益：0.044483。
- 排名删除优于随机删除的样本比例：98.9011%。
- 主要模态删除 score drop：0.251128；次要模态删除：-0.020757；差值：0.271885。
- 主要模态删除大于次要模态删除比例：98.2143%。
- 稀疏性：正重要性窗口比例0.535015；top-5重要性质量占比0.561756；主要模态top-5覆盖位置比例0.277455。
- 1%扰动稳定性：Shapley余弦相似度0.999986；主要模态不变率100%；top evidence位置Jaccard 0.987304。

回归值：

- Top evidence 删除 AOPC：0.092391；随机删除 AOPC：0.003040；增益：0.089351。
- 排名删除优于随机删除的样本比例：99.7253%。
- 主要模态删除 score drop：0.397357；次要模态删除：0.026278；差值：0.371079。
- 主要模态删除大于次要模态删除比例：63.0495%。
- 稀疏性：正重要性窗口比例0.516107；top-5重要性质量占比0.542878；主要模态top-5覆盖位置比例0.287404。
- 1%扰动稳定性：Shapley余弦相似度0.999993；主要模态不变率99.7253%；top evidence位置Jaccard 0.982845。

完整均值、标准差、逐样本曲线和定义见：

- `artifacts/metrics/faithfulness_valid.json`
- `artifacts/metrics/faithfulness_valid.csv`

## 4. 可解释性评测集全量解释与定位

对齐版本20条全部真实运行，未读取或通过对齐特征集匹配推断任何专项标签。输出20行、固定33列。预测类别分布为 Negative 9、Neutral 3、Positive 8；主要模态为 text 16、vision 4、audio 0。

时间定位逐条使用可解释性评测集原视频和该PKL的 `raw_text`，复用特征提取 `ctc_forced_align`。本地 Wav2Vec2 CTC 对齐20/20成功，无均匀映射fallback；每条本地 tokenizer 编码均与所给 `text_bert` 有效 token IDs 完全一致。CTC平均置信度0.863339，范围0.357596至0.970670。音频/视觉证据采用强制对齐起止秒，视觉关键帧为证据区间中点；元数据保存在 `alignment_metadata.json`。

文本证据只从PKL中的 `text_bert` input IDs，使用本地 `third_party/bert-base-uncased` tokenizer解码；共输出62段真实token词/短语，不根据原文补写或生成。

13号视觉输入全零，正式结果满足：`vision_available=false`、视觉Shapley=0、作用程度=0、视觉证据为空、视觉起止与关键帧为空、定位方法unavailable，并含 `VISION_ALL_ZERO` 标志。

正式产物：

- `artifacts/predictions/explanation_set_predictions_explanations.csv`
- `artifacts/explanations/explanation_set/details/01.json` 至 `20.json`
- `artifacts/explanations/explanation_set/regression_shapley.json`
- `artifacts/explanations/explanation_set/alignment_metadata.json`

## 5. 校验与测试

- `validate_explanations.py`：33列CSV通过；20个detail JSON逐一通过Draft-07及跨字段语义校验。
- 全项目 pytest：61 passed，14个既有特征提取依赖弃用警告，无失败。
- 可解释预测专项：16 passed，覆盖真实checkpoint格式加载、联盟置零且mask=false、原始observed mask交集、分类/回归固定目标分别计算、全零视觉贡献为0、checkpoint运行中篡改检测，以及既有schema/解释核心测试。
- 正式CSV为20行33列；忠实性CSV为1456行；所有正式输出只写相对工程引用或文件名+哈希，未写入绝对路径、连接信息或用户名。

## 6. 注意事项

1. 零张量基线和删除实验衡量模型对明确扰动的响应，不等同于因果效应。
2. 回归任务中主要模态删除大于次要模态删除的比例为63.05%，弱于分类；论文应如实报告，不宜只强调平均差值。
3. CTC模型加载时 transformers 提示 `masked_spec_embed` 未从checkpoint初始化；模型处于eval模式，SpecAugment关闭，该训练期参数不参与本次对齐前向。20条对齐均成功并保存逐条置信度，但时间定位仍应理解为自动强制对齐结果而非人工标注金标准。
4. 可解释性评测集无真值标签，因此只输出模型预测与解释，不报告可解释性评测集 Accuracy/F1/MAE/Pearson。
