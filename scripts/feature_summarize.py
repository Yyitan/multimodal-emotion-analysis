#!/usr/bin/env python3
"""Validate all feature NPZ files and rebuild the truthful sample summary/metrics."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/audit/feature_manifest.csv"
NPZ_DIR = ROOT / "artifacts/audit/feature_outputs/npz"
WAV_DIR = ROOT / "artifacts/audit/feature_outputs/audio_16k_mono"
SUMMARY = ROOT / "artifacts/audit/feature_sample_summary.csv"
METRICS = ROOT / "artifacts/audit/feature_run_metrics.json"

manifest = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
by_id = {x["sample_id"]: x for x in manifest}
paths = sorted(NPZ_DIR.glob("*.npz"))
if len(manifest) != 100 or len(paths) != 100:
    raise SystemExit(f"expected 100 manifest rows and NPZs, got {len(manifest)} and {len(paths)}")
rows, sizes, confidence, no_vision, anomaly_samples = [], [], [], [], []
methods = Counter()
for path in paths:
    with np.load(path, allow_pickle=False) as data:
        sid = str(data["sample_id"])
        if sid not in by_id:
            raise ValueError(f"NPZ sample absent from manifest: {sid}")
        assert data["text"].shape == (50, 768)
        assert data["audio"].shape == (50, 74)
        assert data["vision"].shape == (50, 35)
        for key in ("text_mask", "audio_mask", "vision_mask", "vision_detection_valid_mask",
                    "start_time", "end_time", "token", "word"):
            assert data[key].shape == (50,), (sid, key, data[key].shape)
        for key in ("text", "audio", "vision"):
            assert np.isfinite(data[key]).all(), (sid, key)
        method = str(data["alignment_method"])
        assert method in {"ctc_forced_alignment", "approximate"}
        methods[method] += 1
        conf = float(data["alignment_confidence"])
        if np.isfinite(conf):
            confidence.append(conf)
        flags = data["anomaly_flags"].tolist()
        flags = [flags] if isinstance(flags, str) else flags
        if flags:
            anomaly_samples.append(sid)
        if not data["vision_mask"].any():
            no_vision.append(sid)
        metadata = json.loads(str(data["metadata_json"]))
        vision_meta = metadata["vision_meta"]
        wav_path = WAV_DIR / f"{sid}.wav"
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        rows.append({
            "sample_id": sid, "status": "ok", "npz_path": str(path.relative_to(ROOT)),
            "wav_path": str(wav_path.relative_to(ROOT)), "npz_bytes": path.stat().st_size,
            "duration_s": metadata["audio_meta"]["duration_s"],
            "word_count": len(str(data["transcript"]).split()),
            "valid_length": int(data["valid_length"]), "alignment_method": method,
            "alignment_confidence": conf, "text_valid_positions": int(data["text_mask"].sum()),
            "audio_valid_positions": int(data["audio_mask"].sum()),
            "vision_valid_positions": int(data["vision_mask"].sum()),
            "visual_frames_sampled": int(vision_meta.get("frames_sampled", 0)),
            "visual_frames_detected": int(vision_meta.get("frames_detected", 0)),
            "anomaly_flags": "|".join(flags),
        })
        sizes.append(path.stat().st_size)
order = {x["sample_id"]: i for i, x in enumerate(manifest)}
rows.sort(key=lambda x: order[x["sample_id"]])
with SUMMARY.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
metrics = {
    "manifest_count": len(manifest), "npz_count": len(paths),
    "wav_count": len(list(WAV_DIR.glob("*.wav"))), "status_ok_count": len(rows),
    "total_npz_bytes": sum(sizes), "mean_npz_bytes": sum(sizes) / len(sizes),
    "min_npz_bytes": min(sizes), "max_npz_bytes": max(sizes),
    "estimated_100_npz_bytes": round(sum(sizes) / len(sizes) * 100),
    "alignment_methods": dict(methods),
    "finite_ctc_confidence_mean": float(np.mean(confidence)),
    "no_vision_count": len(no_vision), "no_vision_samples": no_vision,
    "anomaly_sample_count": len(anomaly_samples),
    "wav_total_bytes": sum(p.stat().st_size for p in WAV_DIR.glob("*.wav")),
}
METRICS.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(metrics, indent=2, ensure_ascii=False))
