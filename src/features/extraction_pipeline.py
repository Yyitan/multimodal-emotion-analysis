"""Reproducible multimodal feature extraction and word-level alignment.

The output schema is fixed to 50 positions with BERT text embeddings, the
specified 74-D acoustic descriptor, and a documented 35-D MediaPipe
blendshape subset. Identity with the original aligned 35-D visual features is
deliberately not claimed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_POSITIONS = 50
AUDIO_DIM = 74
VISION_DIM = 35
TEXT_DIM = 768
AUDIO_FEATURE_NAMES = (
    [f"mfcc_{i:02d}" for i in range(20)]
    + [f"mfcc_delta_{i:02d}" for i in range(20)]
    + [f"mfcc_delta2_{i:02d}" for i in range(20)]
    + [f"chroma_{i:02d}" for i in range(12)]
    + ["rms", "f0_hz"]
)
# Explicit public MediaPipe Face Blendshapes names: a reproducible substitute,
# not the original aligned 35-D visual feature definition.
VISION_FEATURE_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
]
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-tasks/face_landmarker/"
    "face_landmarker_v2_with_blendshapes.task"
)
FACE_MODEL_SHA256 = "41502960a7323080e5bbf46804f4fe6d0d83aa0aa82ba3209831286e94300083"
WORD_RE = re.compile(r"\S+")
CTC_WORD_RE = re.compile(r"[a-z]+")


@dataclass
class WordInterval:
    word: str
    start: float
    end: float


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_json(command: Sequence[str]) -> Dict[str, Any]:
    proc = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def ffprobe(path: Path) -> Dict[str, Any]:
    return run_json(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])


def probe_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    fmt = meta.get("format", {})
    streams = meta.get("streams", [])
    video = next((x for x in streams if x.get("codec_type") == "video"), {})
    audio = next((x for x in streams if x.get("codec_type") == "audio"), {})
    return {
        "duration_s": float(fmt.get("duration") or video.get("duration") or 0.0),
        "size_bytes": int(fmt.get("size") or 0),
        "video_codec": video.get("codec_name", ""), "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0), "video_fps": video.get("avg_frame_rate", ""),
        "audio_codec": audio.get("codec_name", ""),
        "audio_sample_rate": int(audio.get("sample_rate") or 0),
        "audio_channels": int(audio.get("channels") or 0),
        "has_video": bool(video), "has_audio": bool(audio),
    }


def build_manifest(data_dir: Path, manifest_csv: Path, ffprobe_jsonl: Path) -> List[Dict[str, Any]]:
    import pandas as pd
    labels = pd.read_excel(data_dir / "label-100.xlsx", sheet_name="label", dtype={"video_id": str, "clip_id": str})
    required = {"video_id", "clip_id", "text", "label", "annotation"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"label workbook missing columns: {sorted(missing)}")
    rows, raw_records = [], []
    for _, item in labels.iterrows():
        video_id, clip_id = str(item["video_id"]).strip(), str(item["clip_id"]).strip()
        if clip_id.endswith(".0"):
            clip_id = clip_id[:-2]
        media = data_dir / video_id / f"{clip_id}.mp4"
        sample_id = f"{video_id}__{clip_id}"
        row = {
            "sample_id": sample_id, "video_id": video_id, "clip_id": clip_id,
            "media_path": str(media), "text": "" if pd.isna(item["text"]) else str(item["text"]),
            "label": "" if pd.isna(item["label"]) else str(item["label"]),
            "annotation": "" if pd.isna(item["annotation"]) else str(item["annotation"]),
            "media_exists": media.is_file(),
        }
        if media.is_file():
            meta = ffprobe(media)
            row.update(probe_fields(meta))
            raw_records.append({"sample_id": sample_id, "media_path": str(media), "ffprobe": meta})
        else:
            row.update({"duration_s": 0.0, "size_bytes": 0, "video_codec": "", "width": 0,
                        "height": 0, "video_fps": "", "audio_codec": "", "audio_sample_rate": 0,
                        "audio_channels": 0, "has_video": False, "has_audio": False})
            raw_records.append({"sample_id": sample_id, "media_path": str(media), "error": "missing_media"})
        rows.append(row)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    with ffprobe_jsonl.open("w", encoding="utf-8") as f:
        for record in raw_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return rows


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def extract_audio(media_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    temp = wav_path.with_suffix(".tmp.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media_path),
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temp)], check=True)
    temp.replace(wav_path)


def split_words(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    matches = list(WORD_RE.finditer(text))
    return [m.group(0) for m in matches], [(m.start(), m.end()) for m in matches]


def approximate_word_intervals(words: Sequence[str], duration: float) -> List[WordInterval]:
    """Allocate duration by alphanumeric character count, with a unit floor."""
    if not words:
        return []
    weights = np.asarray([max(1, len(re.sub(r"[^A-Za-z0-9]", "", w))) for w in words], dtype=np.float64)
    edges = np.concatenate([[0.0], np.cumsum(weights) / weights.sum() * max(0.0, duration)])
    edges[-1] = max(0.0, duration)
    return [WordInterval(str(w), float(edges[i]), float(edges[i + 1])) for i, w in enumerate(words)]


def _ctc_viterbi(log_probs: np.ndarray, targets: Sequence[int], blank_id: int) -> Tuple[np.ndarray, float]:
    """Viterbi path over blank-interleaved CTC states."""
    t_count, _ = log_probs.shape
    if not targets:
        raise ValueError("empty CTC target")
    ext = np.empty(2 * len(targets) + 1, dtype=np.int64)
    ext[0::2], ext[1::2] = blank_id, np.asarray(targets, dtype=np.int64)
    s_count = len(ext)
    if t_count < len(targets):
        raise ValueError(f"CTC emissions too short: {t_count} < {len(targets)}")
    prev = np.full(s_count, -np.inf, dtype=np.float32)
    prev[0] = log_probs[0, blank_id]
    if s_count > 1:
        prev[1] = log_probs[0, ext[1]]
    back = np.full((t_count, s_count), -1, dtype=np.int16)
    for t in range(1, t_count):
        cur = np.full(s_count, -np.inf, dtype=np.float32)
        for s in range(s_count):
            choices = [(prev[s], s)]
            if s > 0:
                choices.append((prev[s - 1], s - 1))
            if s > 1 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                choices.append((prev[s - 2], s - 2))
            best, source = max(choices, key=lambda z: z[0])
            cur[s], back[t, s] = best + log_probs[t, ext[s]], source
        prev = cur
    candidates = [(prev[s_count - 1], s_count - 1)]
    if s_count > 1:
        candidates.append((prev[s_count - 2], s_count - 2))
    score, state = max(candidates, key=lambda z: z[0])
    if not np.isfinite(score):
        raise ValueError("no finite CTC alignment path")
    states = np.empty(t_count, dtype=np.int32); states[-1] = state
    for t in range(t_count - 1, 0, -1):
        state = int(back[t, state]); state = max(state, 0); states[t - 1] = state
    return states, float(score / t_count)


def ctc_forced_align(waveform: np.ndarray, sample_rate: int, words: Sequence[str], duration: float,
                     processor: Any, model: Any, device: str = "cpu") -> Tuple[List[WordInterval], float]:
    import torch
    clean = []
    for i, word in enumerate(words):
        letters = "".join(CTC_WORD_RE.findall(word.lower()))
        if letters:
            clean.append((i, letters))
    if not clean:
        raise ValueError("transcript has no CTC-supported letters")
    if len(clean) != len(words):
        raise ValueError("one or more words are unsupported by the alphabetic CTC vocabulary")
    vocab, delimiter = processor.tokenizer.get_vocab(), processor.tokenizer.word_delimiter_token
    target_ids, target_word = [], []
    for j, (source_i, word) in enumerate(clean):
        if j and delimiter in vocab:
            target_ids.append(int(vocab[delimiter])); target_word.append(-1)
        for ch in word.upper():
            if ch not in vocab:
                raise ValueError(f"character not in CTC vocabulary: {ch!r}")
            target_ids.append(int(vocab[ch])); target_word.append(source_i)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    attention_mask = getattr(inputs, "attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    with torch.inference_mode():
        logits = model(input_values, attention_mask=attention_mask).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    blank_id = int(processor.tokenizer.pad_token_id)
    states, path_score = _ctc_viterbi(log_probs, target_ids, blank_id)
    t_count, per_word, confidence = log_probs.shape[0], {}, []
    for frame, state in enumerate(states):
        if state % 2:
            target_index = (state - 1) // 2; word_index = target_word[target_index]
            if word_index >= 0:
                per_word.setdefault(word_index, []).append(frame)
                confidence.append(float(np.exp(log_probs[frame, target_ids[target_index]])))
    if len(per_word) != len(clean):
        raise ValueError(f"CTC missed words: aligned={len(per_word)} expected={len(clean)}")
    fallback = approximate_word_intervals(words, duration); result = []
    for i, word in enumerate(words):
        frames = per_word.get(i)
        if frames:
            result.append(WordInterval(word, min(frames) / t_count * duration, (max(frames) + 1) / t_count * duration))
        else:
            result.append(fallback[i])
    for i in range(len(result) - 1):
        boundary = 0.5 * (result[i].end + result[i + 1].start)
        result[i].end, result[i + 1].start = max(result[i].start, boundary), min(result[i + 1].end, boundary)
    result[0].start, result[-1].end = 0.0, duration
    return result, float(np.mean(confidence)) if confidence else float(math.exp(path_score))


def extract_audio_features(wav_path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Strict [20 MFCC, 20 delta, 20 delta2, 12 chroma, RMS, F0]."""
    import librosa
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    if y.size == 0:
        raise ValueError("empty audio")
    n_fft, hop = 1024, 160
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    delta = librosa.feature.delta(mfcc, order=1, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                                     sr=sr, frame_length=2048, hop_length=hop, fill_na=0.0)
        f0 = np.nan_to_num(f0, nan=0.0)[None, :]
        voiced_fraction = float(np.mean(voiced)) if voiced is not None else 0.0
    except Exception:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
                         sr=sr, frame_length=2048, hop_length=hop)[None, :]
        f0, voiced_fraction = np.nan_to_num(f0, nan=0.0), float("nan")
    n = min(x.shape[1] for x in (mfcc, delta, delta2, chroma, rms, f0))
    features = np.concatenate([mfcc[:, :n], delta[:, :n], delta2[:, :n], chroma[:, :n], rms[:, :n], f0[:, :n]], axis=0).T
    if features.shape[1] != AUDIO_DIM:
        raise AssertionError(f"audio dimension {features.shape[1]} != {AUDIO_DIM}")
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop, n_fft=n_fft)
    return times.astype(np.float32), np.nan_to_num(features).astype(np.float32), {
        "sample_rate": sr, "n_fft": n_fft, "hop_length": hop,
        "duration_s": float(len(y) / sr), "voiced_fraction": voiced_fraction,
    }


def ensure_face_model(path: Path, url: str = FACE_MODEL_URL) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temp = path.with_suffix(".download"); urllib.request.urlretrieve(url, temp); temp.replace(path)
    digest = sha256_file(path)
    if digest != FACE_MODEL_SHA256:
        raise ValueError(f"unexpected Face Landmarker SHA256: {digest}")
    return digest


def extract_visual_features(media_path: Path, model_path: Path, sample_hz: float = 10.0):
    import cv2
    import mediapipe as mp
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True, num_faces=1)
    cap = cv2.VideoCapture(str(media_path)); fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        cap.release(); raise ValueError("OpenCV could not open video or obtain FPS")
    step = max(1, int(round(fps / sample_hz)))
    times, values, valid, missing_names = [], [], [], set(); frame_index = sampled = 0
    category_count = 0
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok: break
            if frame_index % step == 0:
                timestamp_ms = int(round(frame_index / fps * 1000.0))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                sampled += 1; times.append(frame_index / fps)
                if result.face_blendshapes:
                    score_map = {x.category_name: float(x.score) for x in result.face_blendshapes[0]}
                    category_count = max(category_count, len(score_map))
                    missing = set(VISION_FEATURE_NAMES).difference(score_map); missing_names.update(missing)
                    values.append(np.asarray([score_map.get(x, 0.0) for x in VISION_FEATURE_NAMES], dtype=np.float32))
                    valid.append(not missing)
                else:
                    values.append(np.zeros(VISION_DIM, dtype=np.float32)); valid.append(False)
            frame_index += 1
    cap.release()
    if not times:
        return np.empty(0, np.float32), np.empty((0, VISION_DIM), np.float32), np.empty(0, bool), {
            "fps": fps, "sample_hz_actual": fps / step, "frames_sampled": 0, "frames_detected": 0,
            "missing_blendshape_names": [], "blendshape_category_count": 0}
    return np.asarray(times, np.float32), np.vstack(values), np.asarray(valid, bool), {
        "fps": fps, "sample_hz_actual": fps / step, "frames_sampled": sampled,
        "frames_detected": int(np.sum(valid)), "missing_blendshape_names": sorted(missing_names),
        "blendshape_category_count": category_count}


def position_groups(n_words: int, max_positions: int = N_POSITIONS) -> List[np.ndarray]:
    return [] if n_words <= 0 else [x.astype(np.int64) for x in np.array_split(np.arange(n_words), min(n_words, max_positions))]


def pool_frames(frame_times: np.ndarray, frame_features: np.ndarray, intervals: Sequence[WordInterval],
                groups: Sequence[np.ndarray], valid_frame_mask: Optional[np.ndarray] = None):
    output = np.zeros((N_POSITIONS, int(frame_features.shape[1])), dtype=np.float32)
    mask = np.zeros(N_POSITIONS, dtype=bool)
    if valid_frame_mask is None:
        valid_frame_mask = np.ones(len(frame_times), dtype=bool)
    for p, group in enumerate(groups):
        start, end = intervals[int(group[0])].start, intervals[int(group[-1])].end
        selected = (frame_times >= start) & (frame_times <= end) & valid_frame_mask
        if np.any(selected):
            output[p] = np.mean(frame_features[selected], axis=0); mask[p] = True
    return output, mask


def extract_text_features(words: Sequence[str], tokenizer: Any, model: Any, device: str = "cpu"):
    import torch
    output = np.zeros((len(words), TEXT_DIM), dtype=np.float32); valid = np.zeros(len(words), dtype=bool)
    word_tokens = [[] for _ in words]
    encoded = tokenizer(list(words), is_split_into_words=True, return_tensors="pt", truncation=True,
                        max_length=512, add_special_tokens=True)
    word_ids = encoded.word_ids(batch_index=0)
    with torch.inference_mode():
        hidden = model(**{k: v.to(device) for k, v in encoded.items()}).last_hidden_state[0].cpu().numpy()
    token_strings = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
    for token_index, word_index in enumerate(word_ids):
        if word_index is not None and word_index < len(words):
            output[word_index] += hidden[token_index]; valid[word_index] = True
            word_tokens[word_index].append(token_strings[token_index])
    for i in range(len(words)):
        if word_tokens[i]: output[i] /= len(word_tokens[i])
    return output, valid, [" ".join(x) for x in word_tokens]


def pool_words(word_features: np.ndarray, word_valid: np.ndarray, groups: Sequence[np.ndarray]):
    result = np.zeros((N_POSITIONS, int(word_features.shape[1])), dtype=np.float32)
    mask = np.zeros(N_POSITIONS, dtype=bool)
    for p, group in enumerate(groups):
        selected = group[word_valid[group]]
        if len(selected): result[p] = np.mean(word_features[selected], axis=0); mask[p] = True
    return result, mask


def load_transformers(config: Dict[str, Any], model_cache: Path):
    from transformers import AutoModel, AutoModelForCTC, AutoProcessor, AutoTokenizer
    os.environ.setdefault("HF_ENDPOINT", str(config.get("hf_endpoint", "https://hf-mirror.com")))
    model_cache.mkdir(parents=True, exist_ok=True)
    bert_id, bert_rev = str(config["bert_model_id"]), str(config["bert_revision"])
    ctc_id, ctc_rev = str(config["ctc_model_id"]), str(config["ctc_revision"])
    def local_then_network(cls: Any, model_id: str, revision: str, **kwargs: Any) -> Any:
        try:
            return cls.from_pretrained(model_id, revision=revision, cache_dir=str(model_cache), local_files_only=True, **kwargs)
        except Exception:
            return cls.from_pretrained(model_id, revision=revision, cache_dir=str(model_cache), local_files_only=False, **kwargs)
    tokenizer = local_then_network(AutoTokenizer, bert_id, bert_rev, use_fast=True)
    bert = local_then_network(AutoModel, bert_id, bert_rev)
    processor = local_then_network(AutoProcessor, ctc_id, ctc_rev)
    ctc = local_then_network(AutoModelForCTC, ctc_id, ctc_rev)
    bert.eval(); ctc.eval()
    return tokenizer, bert, processor, ctc


def load_config(path: Path) -> Dict[str, Any]:
    import yaml
    with path.open(encoding="utf-8") as f: return yaml.safe_load(f)


def process_sample(row: Dict[str, str], config: Dict[str, Any], output_root: Path,
                   tokenizer: Any, bert: Any, processor: Any, ctc: Any,
                   face_model_path: Path, face_model_sha256: str) -> Dict[str, Any]:
    import cv2, librosa, mediapipe, transformers
    sample_id, media_path = row["sample_id"], Path(row["media_path"])
    wav_path = output_root / "audio_16k_mono" / f"{sample_id}.wav"
    npz_path = output_root / "npz" / f"{sample_id}.npz"; npz_path.parent.mkdir(parents=True, exist_ok=True)
    anomalies = []; extract_audio(media_path, wav_path)
    audio_times, audio_frames, audio_meta = extract_audio_features(wav_path); duration = float(audio_meta["duration_s"])
    words, _ = split_words(row.get("text", ""))
    if not words: words = ["[EMPTY]"]; anomalies.append("empty_transcript")
    alignment_method, alignment_confidence = "ctc_forced_alignment", float("nan")
    try:
        intervals, alignment_confidence = ctc_forced_align(
            librosa.load(str(wav_path), sr=16000, mono=True)[0], 16000, words, duration, processor, ctc)
        if alignment_confidence < float(config.get("ctc_min_confidence", 0.02)):
            raise ValueError(f"CTC confidence below threshold: {alignment_confidence:.6f}")
    except Exception as exc:
        intervals = approximate_word_intervals(words, duration); alignment_method = "approximate"
        anomalies.append(f"ctc_fallback:{type(exc).__name__}:{str(exc)[:160]}")
    groups = position_groups(len(words))
    word_text, word_text_valid, word_tokens = extract_text_features(words, tokenizer, bert)
    text, text_mask = pool_words(word_text, word_text_valid, groups)
    audio, audio_mask = pool_frames(audio_times, audio_frames, intervals, groups)
    try:
        vision_times, vision_frames, vision_valid_frames, vision_meta = extract_visual_features(
            media_path, face_model_path, float(config.get("vision_sample_hz", 10.0)))
        vision, vision_mask = pool_frames(vision_times, vision_frames, intervals, groups, vision_valid_frames)
        if vision_meta["frames_detected"] == 0: anomalies.append("no_face_detected")
        if vision_meta["missing_blendshape_names"]:
            anomalies.append("missing_blendshape_names:" + "|".join(vision_meta["missing_blendshape_names"]))
    except Exception as exc:
        vision = np.zeros((N_POSITIONS, VISION_DIM), np.float32); vision_mask = np.zeros(N_POSITIONS, bool)
        vision_meta = {"error": f"{type(exc).__name__}:{str(exc)[:160]}", "frames_sampled": 0, "frames_detected": 0}
        anomalies.append(f"visual_error:{type(exc).__name__}:{str(exc)[:160]}")
    starts = np.zeros(N_POSITIONS, np.float32); ends = np.zeros(N_POSITIONS, np.float32)
    position_words = np.full(N_POSITIONS, "", dtype="<U1024"); position_tokens = np.full(N_POSITIONS, "", dtype="<U2048")
    for p, group in enumerate(groups):
        starts[p], ends[p] = intervals[int(group[0])].start, intervals[int(group[-1])].end
        position_words[p] = " ".join(words[int(i)] for i in group)
        position_tokens[p] = " | ".join(word_tokens[int(i)] for i in group)
    metadata = {
        "sample_id": sample_id, "source_media": str(media_path),
        "text_model": config["bert_model_id"], "text_revision": config["bert_revision"],
        "ctc_model": config["ctc_model_id"], "ctc_revision": config["ctc_revision"],
        "audio_definition": AUDIO_FEATURE_NAMES, "vision_definition": VISION_FEATURE_NAMES,
        "vision_definition_status": "MediaPipe 35-D reproducible substitute; not original-feature identity",
        "face_model_url": FACE_MODEL_URL, "face_model_sha256": face_model_sha256,
        "audio_meta": audio_meta, "vision_meta": vision_meta,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "librosa": librosa.__version__,
                     "opencv": cv2.__version__, "mediapipe": mediapipe.__version__, "transformers": transformers.__version__}}
    temp_npz = npz_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp_npz, text=text.astype(np.float32), audio=audio.astype(np.float32), vision=vision.astype(np.float32),
        text_mask=text_mask, audio_mask=audio_mask, vision_mask=vision_mask,
        vision_detection_valid_mask=vision_mask.copy(), start_time=starts, end_time=ends,
        token=position_tokens, word=position_words, valid_length=np.asarray(len(groups), np.int32),
        alignment_method=np.asarray(alignment_method), alignment_confidence=np.asarray(alignment_confidence, np.float32),
        anomaly_flags=np.asarray(anomalies, dtype="<U256"), sample_id=np.asarray(sample_id),
        transcript=np.asarray(row.get("text", "")), label=np.asarray(row.get("label", "")),
        annotation=np.asarray(row.get("annotation", "")), metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    temp_npz.replace(npz_path)
    return {"sample_id": sample_id, "status": "ok", "npz_path": str(npz_path), "wav_path": str(wav_path),
        "npz_bytes": npz_path.stat().st_size, "duration_s": duration, "word_count": len(words),
        "valid_length": len(groups), "alignment_method": alignment_method,
        "alignment_confidence": alignment_confidence, "text_valid_positions": int(text_mask.sum()),
        "audio_valid_positions": int(audio_mask.sum()), "vision_valid_positions": int(vision_mask.sum()),
        "visual_frames_sampled": int(vision_meta.get("frames_sampled", 0)),
        "visual_frames_detected": int(vision_meta.get("frames_detected", 0)), "anomaly_flags": "|".join(anomalies)}


def write_summary(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def validate_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        assert data["text"].shape == (N_POSITIONS, TEXT_DIM)
        assert data["audio"].shape == (N_POSITIONS, AUDIO_DIM)
        assert data["vision"].shape == (N_POSITIONS, VISION_DIM)
        for key in ("text_mask", "audio_mask", "vision_mask", "start_time", "end_time", "word", "token"):
            assert data[key].shape == (N_POSITIONS,), (key, data[key].shape)
        assert 0 <= int(data["valid_length"]) <= N_POSITIONS
        assert str(data["alignment_method"]) in {"ctc_forced_alignment", "approximate"}
        assert np.all(np.isfinite(data["text"])) and np.all(np.isfinite(data["audio"])) and np.all(np.isfinite(data["vision"]))


def run_pipeline(config_path: Path, limit: Optional[int] = None, sample_ids: Optional[Sequence[str]] = None):
    config = load_config(config_path); root = Path(config["project_root"]); output_root = root / config["output_dir"]
    manifest_path, ffprobe_path = root / config["manifest_csv"], root / config["ffprobe_jsonl"]
    if not manifest_path.is_file(): build_manifest(Path(config["data_dir"]), manifest_path, ffprobe_path)
    rows = read_manifest(manifest_path)
    if sample_ids: rows = [r for r in rows if r["sample_id"] in set(sample_ids)]
    if limit is not None: rows = rows[:limit]
    model_cache = root / config["model_cache_dir"]; face_path = model_cache / "face_landmarker_v2_with_blendshapes.task"
    face_sha = ensure_face_model(face_path)
    tokenizer, bert, processor, ctc = load_transformers(config, model_cache / "huggingface")
    summaries = []
    for index, row in enumerate(rows, 1):
        print(f"[{index}/{len(rows)}] {row['sample_id']}", flush=True)
        try:
            summary = process_sample(row, config, output_root, tokenizer, bert, processor, ctc, face_path, face_sha)
            validate_npz(Path(summary["npz_path"]))
        except Exception as exc:
            summary = {"sample_id": row["sample_id"], "status": "error", "error": f"{type(exc).__name__}:{str(exc)[:500]}"}
        summaries.append(summary); write_summary(summaries, root / config["summary_csv"])
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    manifest = sub.add_parser("manifest"); manifest.add_argument("--config", required=True, type=Path)
    run = sub.add_parser("run"); run.add_argument("--config", required=True, type=Path)
    run.add_argument("--limit", type=int); run.add_argument("--sample-id", action="append")
    validate = sub.add_parser("validate"); validate.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        cfg = load_config(args.config); root = Path(cfg["project_root"])
        rows = build_manifest(Path(cfg["data_dir"]), root / cfg["manifest_csv"], root / cfg["ffprobe_jsonl"])
        print(f"manifest rows: {len(rows)}")
    elif args.command == "run":
        summaries = run_pipeline(args.config, args.limit, args.sample_id)
        failures = sum(x.get("status") != "ok" for x in summaries)
        print(f"processed={len(summaries)} failures={failures}")
        if failures: raise SystemExit(1)
    else:
        for path in args.paths: validate_npz(path); print(f"valid: {path}")


if __name__ == "__main__":
    main()
