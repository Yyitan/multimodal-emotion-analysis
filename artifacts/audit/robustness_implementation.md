# 模块二正式鲁棒性评估实现审计

## 实现范围

本次仅实现并修改以下职责文件：

- `src/evaluation/robustness.py`
- `scripts/evaluate_robustness.py`
- `scripts/summarize_experiments.py`
- `configs/robustness.yaml`
- `configs/robustness_smoke.yaml`
- `configs/robustness_comparisons.yaml`
- `tests/test_robustness.py`
- `artifacts/audit/robustness_implementation.md`

没有改动数据加载、模型、训练器或现有指标实现；评估仅调用 `MultimodalDataset`、`build_model`、`Trainer.predict` 和 `compute_metrics` 等现有 API。代码和配置没有写入 SSH 信息、账号、密码或服务器绝对路径。

## 数据和测试集隔离

- 正式评估入口只允许 `valid` 或 `test`，数据配置只指向对齐特征集统一特征文件。
- 路径安全检查显式拒绝包含两个评测集标识的输入；评估代码不会发现、加载或检索两个评测集，也不会用 ID、文本或文件名反查标签与重合关系。
- 加载后再次检查分类、回归标签是否完整，避免误把无标签评测集作为正式评估数据。
- `valid` 用于完整敏感性矩阵、规律分析和图表开发。
- `test` 只允许 `--mode full`，禁止样本/条件截断，并要求显式 `--confirm-frozen`。测试网格在 `configs/robustness.yaml` 预先固定，不能通过 CLI 临时改网格。
- 每次运行记录配置、数据和 checkpoint SHA-256、条件数、样本数、bootstrap 设置与冻结确认状态，便于复核。

## 正式评估矩阵

`configs/robustness.yaml` 中验证集矩阵为：

- 缺失模态：`T/A/V/TA/TV/AV/TAV`
- 缺失率：`0/0.1/0.2/0.3/0.5/0.7`
- 位置：`head/middle/tail/random`
- span 数：`1/3`

零缺失只生成一次 clean 条件；非零条件共 `7×5×4×2=280` 个，加 clean 共 281 个。测试集预注册网格为 clean 加所有模态组合与非零缺失率的单段 random 条件，共 36 个，作为冻结后的基础/预注册报告，不用于选择模型。

缺失点数按 `floor(valid_length × rate + 0.5)` 取整；非零比例至少遮一个有效点。`head/middle/tail` 为确定性放置，`random` 使用固定 NumPy Generator；多段之间至少保留一个有效点（可行时），各段长度尽量均衡。组合模态共享同一时间 span，便于公平比较。

`RobustnessDataset` 每次从原样本 clone 特征与 observed mask；只把有效位置的目标模态特征置零并同步把 observed 置假。padding 的特征值、observed mask 和 valid mask 均保持原样。每个样本的随机 span 由评估种子和样本索引经 `SeedSequence` 派生，不依赖 DataLoader 调用顺序。

## 统计与输出

每个 checkpoint、split 和条件都输出：

- Accuracy
- Macro-F1
- Weighted-F1
- MAE
- Pearson
- 样本数、分类有效样本数、回归有效样本数、独立 video group 数
- 每项指标的 95% percentile bootstrap CI

bootstrap 分组为样本 ID 第一个下划线前的 `video_id` 前缀；每次有放回抽取完整 video group，组内样本不拆分。默认正式重采样 2000 次，seed 固定且写入结果。

正式评估 CSV 写入 `artifacts/metrics`；clean 条件逐样本预测单独保存，以便从真实预测生成混淆矩阵。汇总脚本支持：

- checkpoint/baseline 同条件对比及指标差值；
- primary、baseline、ablation 汇总表；
- 同一 family 多训练种子均值和样本标准差；
- 缺失率性能曲线；
- 模态类型热图；
- 位置影响图；
- 每个 checkpoint 的 clean 混淆矩阵（仅当真实预测 CSV 存在）。

图同时输出 300 dpi PNG 和矢量 PDF。绘图前实际检查候选字体是否包含所需中文字形；本机未验证到满足条件的中文字体时自动使用英文标签，并在 `robustness_summary_manifest.json` 写入论文中文图注应说明的文字，避免缺字方框。

## 核心接口

- `normalize_modalities(value)`：标准化 T/A/V 组合。
- `MissingCondition`：不可变的单个缺失条件定义。
- `build_span_mask(valid, missing_rate, position, n_spans, seed)`：生成只覆盖有效区间的确定性连续 span mask。
- `apply_span_missingness(...)`：样本/批次级特征和 observed mask 遮蔽，保持 padding。
- `RobustnessDataset(base, condition)`：在现有数据集之上建立可重复条件视图。
- `video_group(id)`：取第一个下划线前缀。
- `iter_group_bootstrap_indices(...)`：整组 bootstrap 索引生成器。
- `compute_metrics_with_group_ci(prediction, ...)`：五项指标、样本数和 video-group 95% CI。
- `conditions_from_grid(grid, seed)`：展开并去重 clean 条件的预注册网格。

## 运行命令

轻量 smoke（隔离写入 `runs/`，不能作为论文结果）：

```bash
env/run.sh env/venv/bin/python scripts/evaluate_robustness.py \
  --config configs/robustness_smoke.yaml --split valid --mode smoke --device cpu
```

正式验证矩阵（冻结 checkpoint 就绪后已运行）：

```bash
env/run.sh env/venv/bin/python scripts/evaluate_robustness.py \
  --config configs/robustness.yaml --split valid --mode full --device auto
```

验证集结论和图表确认、方案冻结后，才可运行预注册测试矩阵：

```bash
env/run.sh env/venv/bin/python scripts/evaluate_robustness.py \
  --config configs/robustness.yaml --split test --mode full --device auto --confirm-frozen
```

生成正式统计表和论文图（以下以验证集为例）：

```bash
env/run.sh env/venv/bin/python scripts/summarize_experiments.py \
  --config configs/robustness.yaml \
  --metrics artifacts/metrics/robustness_valid.csv \
  --predictions artifacts/metrics/robustness_valid_clean_predictions.csv \
  --split valid
```

测试：

```bash
env/run.sh env/venv/bin/python -m pytest -q tests/test_robustness.py
env/run.sh env/venv/bin/python -m pytest -q
```

## 已执行验证和正式结果

- 冻结 checkpoint：`artifacts/models.pt`，SHA-256 为 `989b84b5e507a97186179e80e0f7f8c8e29714f3c84a84aa1d60b3a2ee5bb601`，与冻结模型卡一致。正式 valid/test 配置只加载此 checkpoint；没有加载或重命名其他 gated 候选。
- valid 正式矩阵：281/281 条件、728 个样本、235 个 video groups、每条件 2000 次整组 bootstrap。clean 指标为 Accuracy 0.620879、Macro-F1 0.609382、Weighted-F1 0.624585、MAE 0.585749、Pearson 0.661413。
- test 预注册矩阵：36/36 条件、727 个样本、378 个 video groups、每条件 2000 次整组 bootstrap；manifest 中 `frozen_test_confirmation=true`。clean 指标为 Accuracy 0.646492、Macro-F1 0.611130、Weighted-F1 0.651348、MAE 0.643800、Pearson 0.668963。test 仅在冻结后运行，未用于任何选择。
- baseline/消融只在 valid clean 条件做补充对比：concat Macro-F1 0.600280、MAE 0.615030；text-only Macro-F1 0.603254、MAE 0.606034；无增强和无一致性均为 Macro-F1 0.615506、MAE 0.593979。后两者按冻结模型卡政策仅作消融，不参与模型选择。
- 三训练种子均值±样本标准差从冻结模型卡已经记录的真实 valid 指标生成，不加载三个候选 checkpoint：Accuracy 0.625916±0.005551、Macro-F1 0.600513±0.009339、Weighted-F1 0.623587±0.004909、MAE 0.587110±0.001713、Pearson 0.653347±0.007239。
- 正式指标 CSV/manifest/表格均写入 `artifacts/metrics`，16 个正式图文件（valid/test 各 4 图×PNG/PDF）写入 `artifacts/figures`。混淆矩阵来自保存的逐样本 clean 真实预测。
- 字体检测没有确认到覆盖所需中文字形的中文字体，因此正式图均使用英文标签；summary manifest 已写入论文中文图注说明。
- 开发 smoke：对齐特征集 valid 前 8 个样本、4 个条件、每条件 30 次 group bootstrap，链路通过；smoke 产物与正式产物隔离。
- 专项测试覆盖 span 连续性、比例、位置、组合、padding 不变、随机种子可复现、dataset view 和整 video group bootstrap。
- 最终 `python -m pytest -q tests/test_robustness.py`：11 passed；最终全套 `python -m pytest -q`：61 passed、14 warnings、8.43s。warnings 均来自既有特征提取视觉测试加载 matplotlib 时的 pyparsing deprecation，不是本实现失败。
- 最终产物完整性检查：valid/test CI 分别 2810/360 个数值单元且无空值；混淆矩阵总数分别为 728/727，与对应 split 样本数一致；16 个 PNG/PDF 图文件均非空并人工检查主性能曲线可读；允许范围文件敏感连接信息扫描无匹配。
