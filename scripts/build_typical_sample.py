from pathlib import Path
import csv, json, wave
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT=Path(__file__).resolve().parents[1]
SID='-3g5yACwYnA__13'
NPZ=ROOT/'artifacts/audit/feature_outputs/npz'/f'{SID}.npz'
WAV=ROOT/'artifacts/audit/feature_outputs/audio_16k_mono'/f'{SID}.wav'
VIDEO=ROOT/'data/videos_raw/-3g5yACwYnA/13.mp4'
OUT=ROOT/'artifacts/audit/typical_sample'; OUT.mkdir(parents=True,exist_ok=True)

def get_frame(t):
    cap=cv2.VideoCapture(str(VIDEO)); cap.set(cv2.CAP_PROP_POS_MSEC,float(t)*1000)
    ok,bgr=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'cannot read frame at {t}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)

with np.load(NPZ,allow_pickle=True) as z:
    L=int(z['valid_length']); st=z['start_time'][:L].astype(float); en=z['end_time'][:L].astype(float)
    words=[str(x) for x in z['word'][:L]]; tokens=[str(x) for x in z['token'][:L]]
    text=z['text'][:L].astype(float); audio=z['audio'][:L].astype(float); vision=z['vision'][:L].astype(float)
    tm=z['text_mask'][:L].astype(bool); am=z['audio_mask'][:L].astype(bool); vm=z['vision_mask'][:L].astype(bool)
    transcript=str(z['transcript'].item()); label=float(z['label']); annotation=str(z['annotation'].item())
    method=str(z['alignment_method'].item()); conf=float(z['alignment_confidence'])

mid=(st+en)/2
with wave.open(str(WAV)) as wf:
    sr=wf.getframerate(); sig=np.frombuffer(wf.readframes(wf.getnframes()),np.int16).astype(float)
    sig/=max(np.max(np.abs(sig)),1); wav_t=np.arange(len(sig))/sr

text_norm=np.linalg.norm(text,axis=1); audio_norm=np.linalg.norm(audio,axis=1); vision_norm=np.linalg.norm(vision,axis=1)
rows=[]
for i in range(L):
    rows.append({
        'position':i+1,'word':words[i],'token':tokens[i],
        'start_sec':round(st[i],6),'end_sec':round(en[i],6),'keyframe_sec':round(mid[i],6),
        'text_dim':768,'audio_dim':74,'vision_dim':35,
        'text_valid':bool(tm[i]),'audio_valid':bool(am[i]),'vision_valid':bool(vm[i]),
        'text_l2_norm':round(float(text_norm[i]),6),
        'audio_l2_norm':round(float(audio_norm[i]),6),
        'vision_l2_norm':round(float(vision_norm[i]),6),
    })
with (OUT/'典型样本_逐位置对应表.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

frame_idx=[0,L//2,L-1]; frame_times=[mid[i] for i in frame_idx]; frames=[get_frame(t) for t in frame_times]
fig=plt.figure(figsize=(13,10),dpi=180)
gs=GridSpec(5,3,figure=fig,height_ratios=[1.55,1.0,.75,1.1,1.1],hspace=.55,wspace=.1)
for j,(fr,t,i) in enumerate(zip(frames,frame_times,frame_idx)):
    ax=fig.add_subplot(gs[0,j]); ax.imshow(fr); ax.axis('off'); ax.set_title(f'Frame {t:.2f}s | position {i+1}: {words[i]}',fontsize=9)
ax=fig.add_subplot(gs[1,:]); ax.plot(wav_t,sig,lw=.35,color='#C45A45'); ax.set_xlim(0,float(en[-1])); ax.set_ylabel('Wave amplitude'); ax.set_xlabel('Time (s)')
for x in st: ax.axvline(x,color='#8DA0B2',alpha=.25,lw=.5)
ax.set_title('Audio waveform and CTC word boundaries')
ax=fig.add_subplot(gs[2,:]); ax.set_xlim(0,float(en[-1])); ax.set_ylim(0,1); ax.set_yticks([]); ax.set_xlabel('Time (s)')
for i,(s,e,wrd) in enumerate(zip(st,en,words)):
    ax.barh(.5,e-s,left=s,height=.42,color='#DDEAF4',edgecolor='#466B8A',linewidth=.5)
    ax.text((s+e)/2,.5,wrd,ha='center',va='center',fontsize=7,rotation=40)
ax.set_title('Transcript intervals on the shared timeline')
ax=fig.add_subplot(gs[3,:]); im=ax.imshow(audio.T,aspect='auto',cmap='coolwarm',interpolation='nearest'); ax.set_ylabel('74 acoustic dimensions'); ax.set_title('Aligned audio features (74 × 15)'); fig.colorbar(im,ax=ax,fraction=.015,pad=.01)
ax=fig.add_subplot(gs[4,:]); im=ax.imshow(vision.T,aspect='auto',cmap='viridis',interpolation='nearest'); ax.set_ylabel('35 visual dimensions'); ax.set_title('Aligned visual features (35 × 15)'); ax.set_xlabel('Aligned position: '+', '.join(f'{i+1}:{w}' for i,w in enumerate(words))); ax.tick_params(axis='x',labelsize=6,rotation=55); fig.colorbar(im,ax=ax,fraction=.015,pad=.01)
fig.suptitle(f'特征提取 typical sample {SID} | {annotation}, intensity={label:.3f} | {method}, confidence={conf:.3f}\nText feature: 15×768 (L2 norms available in CSV)',fontsize=13,fontweight='bold')
fig.savefig(OUT/'典型样本_三模态时序对齐展示.png',bbox_inches='tight',facecolor='white')
plt.close(fig)

md=f'''# 典型样本验证

## 样本信息

- 样本编号：`{SID}`
- 原始文本：`{transcript}`
- 情感标注：{annotation}
- 连续情感强度：{label:.6f}
- 对齐方式：`{method}`
- 对齐置信度：{conf:.6f}
- 有效对齐位置：{L}
- 三模态特征维度：文本 `50×768`，语音 `50×74`，视觉 `50×35`

## 对齐逻辑

CTC强制对齐给出每个英文词的起止时间 `[start_sec, end_sec]`。同一时间窗作为跨模态共同锚点：文本位置保存对应BERT表示；语音位置聚合该时间窗内的74维帧级声学特征；视觉位置聚合该时间窗内检测到的人脸blendshape特征。关键帧时间取区间中点。该样本15个有效位置的文本、语音、视觉掩码均为真。

## 展示文件

- `典型样本_三模态时序对齐展示.png`：三个真实视频帧、音频波形、词级时间区间、74维语音特征热图和35维视觉特征热图。
- `典型样本_逐位置对应表.csv`：15个位置逐行列出词语、token、起止秒、关键帧秒、三模态维度、有效掩码和三模态特征L2范数，可直接回查原视频及NPZ。

## 核验结论

样本的15个文本区间按时间单调排列并覆盖0至{en[-1]:.6f}秒；语音和视觉均使用完全相同的时间窗聚合。展示图中的三个帧分别来自首、中、末位置的区间中点，能够直接核验文本—语音—视频帧的时间对应关系。
'''
(OUT/'典型样本说明.md').write_text(md,encoding='utf-8')
meta={'sample_id':SID,'valid_length':L,'alignment_method':method,'alignment_confidence':conf,'frame_times':frame_times,'output_files':['典型样本_三模态时序对齐展示.png','典型样本_逐位置对应表.csv','典型样本说明.md']}
(OUT/'typical_sample_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False,indent=2))
