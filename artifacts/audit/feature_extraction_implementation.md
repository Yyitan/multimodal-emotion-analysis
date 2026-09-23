# 多模态特征提取流水线审计

## 1. 完成范围

本流水线已对原始视频集的100条视频全部实际处理并逐个校验，不修改原始数据。清单共100行，100个媒体文件均存在且同时含音频、视频流；总时长1120.144秒，单条时长范围2.648--34.567秒（中位数10.384秒）。

主要入口：

```bash
./scripts/feature_manifest.sh
./scripts/feature_smoke.sh
./scripts/feature_run.sh run --config configs/feature_extraction.yaml
./env/run.sh ./env/venv/bin/python scripts/feature_summarize.py
```

所有路径通过项目根目录相对解析；项目文件中未写入远程主机、用户名或密码。

## 2. 样本清单、媒体信息与音频

- `feature_manifest.csv`：100条标签、媒体路径及常用ffprobe字段。
- `feature_ffprobe.jsonl`：每条媒体的完整原始ffprobe JSON，便于复核。
- ffmpeg将每条视频提取为16 kHz、单声道、PCM signed 16-bit WAV；100条WAV合计24,875,336字节。
- 原始数据只读；所有派生产物均在工程根下。

## 3. 文本与词级对齐

固定文本模型为 `google-bert/bert-base-uncased`，revision `86b5e0934494bd15c9632b12f734a8a67f723594`，采用fast WordPiece tokenizer。每个词的subword hidden states均值形成768维词向量，再按最多50个连续位置池化。

词级时间优先使用公开CTC模型 `facebook/wav2vec2-base-960h`，revision `22aad52d435eb6dbaf354bdad9b0da84ce7d6156`，用blank交错状态Viterbi强制对齐。模型权重缓存文件SHA256为 `8aa76ab2243c81747a1f832954586bc566090c83a0ac167df6f31f0fa917d74a`。100条中88条为 `ctc_forced_alignment`。

明确fallback：遇到CTC字母表不支持的词，或平均对齐置信度低于0.02时，按每词去除非字母数字字符后的长度分配时长，最小权重为1，并写入 `alignment_method=approximate` 与异常原因。100条中12条为 `approximate`（9条含不支持字符，3条低置信度）。该近似结果未冒称精确。

## 4. 74维音频定义

帧参数为16 kHz、`n_fft=1024`、`hop_length=160`。严格连接：20 MFCC、20一阶delta、20二阶delta、12 chroma STFT、1 RMS、1 F0（优先pYIN，失败时YIN），合计74维。每个统一位置按对应词时间区间内音频帧均值池化。100条共1,917个有效音频位置。

## 5. 35维视觉替代特征

对齐特征集的35维视觉定义未公开，因此本实现**不声称复现对齐特征集视觉定义**。采用明确且可复现的替代：MediaPipe Face Landmarker公开blendshape中的固定35项子集，名称完整保存在每个NPZ的 `metadata_json.vision_definition` 和源码常量中。

模型为官方含blendshape head的 `face_landmarker_v2_with_blendshapes.task`：

- URL：`https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker_v2_with_blendshapes.task`
- SHA256：`41502960a7323080e5bbf46804f4fe6d0d83aa0aa82ba3209831286e94300083`
- 实测每次检测输出52个公开类别，从中按名称选择固定35项。
- 以约10 Hz采样视频帧，按位置对检测有效帧均值池化。
- `vision_mask` 与 `vision_detection_valid_mask` 保存统一50位置上的检测有效性。

全量共采样7,936帧，检测到人脸6,617帧；87条至少有一个有效视觉位置，13条未检测到人脸，特征保持零且掩码为false，并写入 `no_face_detected`，未伪造数值。两条最终smoke分别检测55/55与144/144帧，视觉特征绝对值和均大于零。

## 6. 统一NPZ模式

每条压缩NPZ至少包含：

- `text (50,768)`、`audio (50,74)`、`vision (50,35)`，均为float32；
- `text_mask`、`audio_mask`、`vision_mask`、`vision_detection_valid_mask`，均为 `(50,)`；
- `start_time`、`end_time`、`token`、`word`，均为 `(50,)`；
- `valid_length`、`alignment_method`、`alignment_confidence`；
- `anomaly_flags`、标签、原文与JSON元数据。

当词数超过50时，用 `numpy.array_split` 将连续词确定性分组；不超过50时每词一个位置。所有填充位置为零且mask为false。NPZ采用临时文件加原子rename落盘。

## 7. 实际全量结果与大小

- 状态为ok：100/100；NPZ：100；WAV：100。
- 对齐：CTC 88，明确近似12。
- 无视觉检测样本：13；含任一异常标记样本：22（包括上述近似对齐与无脸样本）。
- 100个NPZ实际总大小：6,981,612字节（约6.66 MiB）。
- 平均69,816字节；范围22,917--171,716字节。
- 因已处理全部100条，100条估算值即实际总大小6,981,612字节。

逐条事实见 `feature_sample_summary.csv`，聚合指标见 `feature_run_metrics.json`；不存在伪造行。

## 8. 版本与测试

运行版本：Python 3.10.18，NumPy 1.26.4，pandas 2.2.3，openpyxl 3.1.5，librosa 0.11.0，SoundFile 0.13.1，OpenCV 4.11.0，PyTorch 2.5.1+cu121，Transformers 4.57.1，MediaPipe 0.10.21，SciPy 1.15.3，ffmpeg/ffprobe 4.2.7。

最终测试：`8 passed`。测试覆盖74/35维定义、加权近似区间、50位置分组、掩码池化、CTC Viterbi、真实74维音频、官方模型SHA256及真实视频类别数/非零视觉检测、smoke产物视觉有效性。pytest另报告14条来自matplotlib旧API的弃用警告，不影响测试。

最终版本再次实际跑通2条端到端smoke：`processed=2 failures=0`。随后对全部100个NPZ逐个以 `allow_pickle=False` 读取，检查形状、键、有限数值、对齐方法及WAV存在性，并据此重建汇总CSV。
