#!/usr/bin/env python3
"""Run real gated-checkpoint explanations for all explanation-set samples."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer

from src.data import MultimodalDataset
from src.explain.model_adapter import CLASS_NAMES, GatedCheckpointAdapter
from src.explain.pipeline import explain_sample
from src.explain.records import flatten_detail_record, write_detail_json, write_summary_csv
from src.explain.shapley import exact_modality_shapley
from src.features.extraction_pipeline import ctc_forced_align

MODALITIES = ("text", "audio", "vision")
WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def discover(path: Path) -> List[Path]:
    return [path] if path.is_file() else sorted(path.glob("*.pkl"))


def ffprobe_duration(path: Path) -> Optional[float]:
    try:
        value = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            text=True,
        ).strip()
        result = float(value)
        return result if np.isfinite(result) and result > 0 else None
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def decode_token_window(tokenizer: Any, token_ids: np.ndarray, start: int, end: int) -> Optional[str]:
    """Decode evidence only from supplied text_bert IDs; never synthesize text."""
    ids = [int(x) for x in token_ids[start:end]]
    special = set(int(x) for x in tokenizer.all_special_ids)
    ids = [x for x in ids if x not in special]
    if not ids:
        return None
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    return text or None


class ForcedAligner:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForCTC.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
        self.model_dir = model_dir

    def align(
        self,
        video: Path,
        raw_text: str,
        tokenizer: Any,
        supplied_ids: np.ndarray,
        attention_mask: np.ndarray,
        duration: float,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        words = [match.group(0) for match in WORD_RE.finditer(raw_text)]
        spans = [(match.start(), match.end()) for match in WORD_RE.finditer(raw_text)]
        metadata: Dict[str, Any] = {
            "method": "ctc_forced_alignment",
            "status": "failed",
            "word_count": len(words),
            "model": self.model_dir.name,
        }
        if not words:
            metadata["error"] = "transcript_has_no_supported_words"
            return None, metadata
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                wav_path = Path(handle.name)
            subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                 "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path)],
                check=True,
            )
            import librosa
            waveform = librosa.load(str(wav_path), sr=16000, mono=True)[0]
            intervals, confidence = ctc_forced_align(
                waveform, 16000, words, duration, self.processor, self.model, device=self.device
            )
            encoded = tokenizer(
                raw_text, truncation=True, max_length=len(supplied_ids), padding="max_length",
                return_offsets_mapping=True,
            )
            generated = np.asarray(encoded["input_ids"], dtype=np.int64)
            valid = np.asarray(attention_mask, dtype=bool)
            ids_match = bool(np.array_equal(generated[valid], supplied_ids[valid]))
            metadata.update({"ctc_confidence": float(confidence), "token_ids_match_raw_text": ids_match})
            if not ids_match:
                raise ValueError("local tokenizer/raw_text IDs do not match supplied text_bert IDs")
            offsets = encoded["offset_mapping"]
            result: List[Dict[str, Any]] = []
            last_end = 0.0
            valid_indices = np.flatnonzero(valid)
            last_valid = int(valid_indices[-1]) if len(valid_indices) else -1
            for position, ((start, end), is_valid) in enumerate(zip(offsets, valid)):
                if not is_valid:
                    result.append({"position": position, "start_sec": duration, "end_sec": duration, "confidence": "aligned"})
                    continue
                overlap = [i for i, (left, right) in enumerate(spans) if right > start and left < end]
                if overlap:
                    begin = min(intervals[i].start for i in overlap)
                    finish = max(intervals[i].end for i in overlap)
                    last_end = finish
                elif position == 0:
                    begin = finish = 0.0
                elif position == last_valid:
                    begin = finish = duration
                else:
                    begin = finish = last_end
                result.append({"position": position, "start_sec": float(begin), "end_sec": float(finish), "confidence": "aligned"})
            metadata["status"] = "ok"
            metadata["position_count"] = len(result)
            return result, metadata
        except Exception as exc:
            metadata["error"] = f"{type(exc).__name__}:{str(exc)[:240]}"
            return None, metadata
        finally:
            if wav_path is not None:
                wav_path.unlink(missing_ok=True)


def compact_faithfulness(value: Optional[Mapping[str, Any]], seed: int, trials: int) -> Dict[str, Any]:
    if not value:
        return {
            "metric": "AOPC cumulative deletion", "ranked_aopc": 0.0,
            "random_aopc_mean": 0.0, "random_aopc_std": 0.0,
            "aopc_gain_over_random": 0.0, "steps": 1, "random_trials": trials, "seed": seed,
        }
    return {
        "metric": "AOPC cumulative deletion",
        "ranked_aopc": float(value["ranked"]["aopc"]),
        "random_aopc_mean": float(value["random"]["aopc_mean"]),
        "random_aopc_std": float(value["random"]["aopc_std"]),
        "aopc_gain_over_random": float(value["aopc_gain_over_random"]),
        "steps": int(value["steps"]), "random_trials": int(value["random"]["trials"]),
        "seed": int(value["random"]["seed"]),
    }


def detail_record(
    sample_id: str,
    adapter: GatedCheckpointAdapter,
    class_target: Mapping[str, Any],
    result: Mapping[str, Any],
    feature_version: str,
    config_reference: str,
    seed: int,
    trials: int,
) -> Dict[str, Any]:
    prediction = class_target["prediction"]
    probabilities = np.asarray(prediction["probabilities"], dtype=float)
    shapley = result["shapley"]
    availability = result["quality"]["availability"]
    modalities = {
        modality: {
            "available": bool(availability[modality]),
            "shapley_signed": float(shapley["signed_contributions"][modality]),
            "degree": float(shapley["degrees"][modality]),
        }
        for modality in MODALITIES
    }
    notes = []
    for modality in MODALITIES:
        if not availability[modality]:
            notes.append(f"{modality} unavailable in model input; contribution and evidence forced to zero/empty")
    return {
        "schema_version": "1.0.0", "sample_id": sample_id,
        "prediction": {
            "polarity": CLASS_NAMES[int(class_target["class_index"])],
            "intensity": float(prediction["regression"]),
            "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
        },
        "explanation": {
            "target_output": f"predicted_class_probability:{class_target['class_name']}",
            "baseline_score": float(shapley["baseline_score"]), "full_score": float(shapley["full_score"]),
            "coalition_scores": {k: float(v) for k, v in shapley["coalition_scores"].items()},
            "modalities": modalities, "primary_modality": shapley["primary_modality"],
            "primary_tied_modalities": list(shapley["primary_tied_modalities"]),
            "primary_confidence": shapley["primary_confidence"],
            "efficiency_residual": float(shapley["efficiency_residual"]),
            "evidence": result["evidence"],
        },
        "faithfulness": compact_faithfulness(result["faithfulness"], seed, trials),
        "quality": {"flags": list(result["quality"]["quality_flags"]), "notes": notes},
        "provenance": {
            "feature_version": feature_version,
            "explainer": "exact_3modality_shapley_plus_local_occlusion",
            "explainer_config": config_reference,
            "model_reference": adapter.model_reference,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", help="Required for a formal frozen-checkpoint run")
    parser.add_argument("--allow-unfrozen", action="store_true", help="Smoke testing only")
    parser.add_argument("--input", required=True, help="explanation-set aligned directory or one PKL")
    parser.add_argument("--output-dir", default="artifacts/explanations/explanation_set")
    parser.add_argument("--summary-csv", default="artifacts/predictions/explanation_set_predictions_explanations.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokenizer", default="third_party/bert-base-uncased")
    parser.add_argument("--ctc-model-dir", default="src/features/model_cache/huggingface/models--facebook--wav2vec2-base-960h/snapshots/22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
    parser.add_argument("--ctc-device", default="cpu")
    parser.add_argument("--no-ctc", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--aopc-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if not args.expected_sha256 and not args.allow_unfrozen:
        parser.error("formal runs require --expected-sha256; use --allow-unfrozen only for smoke tests")

    adapter = GatedCheckpointAdapter(args.checkpoint, args.device, args.expected_sha256)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, use_fast=True)
    aligner = None if args.no_ctc else ForcedAligner(Path(args.ctc_model_dir), args.ctc_device)
    paths = discover(Path(args.input))
    if args.max_samples:
        paths = paths[: args.max_samples]
    if not paths:
        raise FileNotFoundError("no explanation-set PKL files found")
    out = Path(args.output_dir); details_dir = out / "details"; details_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_csv); summary_path.parent.mkdir(parents=True, exist_ok=True)
    rows, regression_records, alignment_records = [], [], []

    for path in paths:
        dataset = MultimodalDataset(path)
        if len(dataset) != 1:
            raise ValueError(f"explanation-set file must contain one sample: {path}")
        item = dataset[0]; sample = adapter.sample_from_item(item); sample_id = str(item["id"])
        full = adapter.predict(sample)
        class_index = int(np.argmax(full["probabilities"]))
        class_target = {
            "class_index": class_index, "class_name": CLASS_NAMES[class_index],
            "score": "probability", "selector": adapter.classification_selector(class_index),
            "prediction": full,
        }
        valid_length = int(np.asarray(sample["_valid"], dtype=bool).sum())
        valid_lengths = {m: valid_length for m in MODALITIES}
        video = path.parent / "videos" / f"{sample_id}.mp4"
        duration = ffprobe_duration(video)
        forced = None; align_meta: Dict[str, Any] = {"sample_id": sample_id, "video_found": video.is_file(), "duration_sec": duration}
        if aligner is not None and video.is_file() and duration is not None:
            forced, attempt = aligner.align(
                video, sample["raw_text"], tokenizer, sample["_text_bert"][0],
                sample["_text_bert"][1], duration,
            )
            align_meta.update(attempt)
        else:
            align_meta.update({"method": "uniform_by_video_duration", "status": "fallback", "error": "ctc_disabled_or_input_unavailable"})
        if forced is None:
            align_meta["fallback"] = "uniform_by_video_duration" if duration is not None else "unavailable"
        alignment_records.append(align_meta)
        forced_alignments = {"audio": forced, "vision": forced} if forced is not None else {}
        token_ids = sample["_text_bert"][0]
        result = explain_sample(
            adapter.predict, sample, score_selector=class_target["selector"],
            window_sizes={"text": 1, "audio": 3, "vision": 3},
            strides={m: 1 for m in MODALITIES}, valid_lengths=valid_lengths,
            top_k=5, max_merge_gap=0, duration_sec=duration,
            forced_alignments=forced_alignments,
            text_resolver=lambda start, end, ids=token_ids: decode_token_window(tokenizer, ids, start, end),
            aopc_steps=args.aopc_steps, random_trials=args.random_trials, seed=args.seed,
        )
        regression = exact_modality_shapley(
            adapter.predict, sample, score_selector=adapter.regression_selector,
            available=result["quality"]["availability"],
        )
        regression_records.append({
            "sample_id": sample_id, "target_output": "regression_value",
            "prediction": float(full["regression"]), "shapley": regression.to_dict(),
        })
        record = detail_record(
            sample_id, adapter, class_target, result, "aligned_50", "configs/explanation.yaml",
            args.seed, args.random_trials,
        )
        detail_path = details_dir / f"{sample_id}.json"; write_detail_json(record, detail_path)
        relative_detail = os.path.relpath(detail_path, summary_path.parent).replace(os.sep, "/")
        rows.append(flatten_detail_record(record, relative_detail))
        print(f"explained {sample_id}: class={class_target['class_name']} primary={record['explanation']['primary_modality']}")

    write_summary_csv(rows, summary_path)
    (out / "regression_shapley.json").write_text(json.dumps({
        "target_output": "regression_value", "model_reference": adapter.model_reference,
        "records": regression_records,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "alignment_metadata.json").write_text(json.dumps(alignment_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adapter.assert_checkpoint_unchanged()
    print(json.dumps({"rows": len(rows), "summary_csv": str(summary_path), "details_dir": str(details_dir),
                      "model_reference": adapter.model_reference}, ensure_ascii=False))


if __name__ == "__main__":
    main()
