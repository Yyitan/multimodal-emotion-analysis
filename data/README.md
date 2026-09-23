# 数据目录说明

本仓库不包含任何原始数据、视频、音频、PKL/XLSX特征文件或逐样本NPZ特征。

本项目使用公开数据集 CMU-MOSEI 的一个固定子集（100条视频）。请自行获取数据后，在本地建立以下目录，或通过环境变量/配置文件指向数据位置：

```text
data/
├── videos_raw/          # 原始100条视频与标注表
├── features_aligned/    # aligned_50.pkl / unaligned_50.pkl / 标签文件
├── missing_modality/    # 模态缺失评测集（对齐版本）
└── explanation_set/     # 可解释性评测集（对齐版本）
```

`data/` 下除本说明外的内容均被 `.gitignore` 排除。禁止提交数据文件和衍生逐样本NPZ。
