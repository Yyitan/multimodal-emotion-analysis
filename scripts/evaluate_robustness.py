#!/usr/bin/env python3
"""Evaluate frozen checkpoints under a pre-specified continuous-span matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Support both ``python scripts/...`` and module-like execution from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MultimodalDataset
from src.evaluation.robustness import (
    METRIC_NAMES,
    MissingCondition,
    RobustnessDataset,
    compute_metrics_with_group_ci,
    conditions_from_grid,
    modality_code,
)
from src.models import build_model
from src.training.trainer import Trainer, set_seed


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Robustness config root must be a mapping")
    return config


def _validate_label_source(data_path: Path, split: str) -> None:
    """Guard the formal evaluator against accidental evaluation-set misuse."""
    if split not in {"valid", "test"}:
        raise ValueError("Formal robustness evaluation permits only aligned-features valid/test")
    # The file structure is checked by MultimodalDataset. This lexical guard catches
    # the most likely configuration error without searching any other data directory.
    lowered = str(data_path).lower()
    forbidden = ("missing_modality", "explanation_set")
    if any(token in lowered for token in forbidden):
        raise ValueError("Evaluation-set data is forbidden in robustness evaluation")


def _model_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("config.models must be a non-empty list of frozen checkpoints")
    names = [str(model.get("name", "")) for model in models]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Every model needs a unique non-empty name")
    return models


def _condition_metadata(condition: MissingCondition) -> dict[str, Any]:
    return {
        "condition": condition.code,
        "modalities": modality_code(condition.modalities),
        "missing_rate": condition.missing_rate,
        "position": "clean" if condition.missing_rate == 0 else condition.position,
        "n_spans": 0 if condition.missing_rate == 0 else condition.n_spans,
        "mask_seed": condition.seed,
    }


def evaluate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    config_path = Path(args.config)
    config = _load_config(config_path)
    split = args.split
    if args.mode == "full" and (args.max_samples is not None or args.max_conditions is not None):
        raise ValueError("full mode never permits sample or condition caps; use --mode smoke")
    if split == "test":
        if args.mode != "full":
            raise ValueError("test permits only the complete pre-registered full run")
        if not args.confirm_frozen:
            raise ValueError("test requires --confirm-frozen after validation decisions are final")
        grid_name = "test_grid"
    else:
        grid_name = "validation_grid"
    if grid_name not in config:
        raise KeyError(f"Missing pre-registered config section: {grid_name}")

    seed = int(config.get("seed", 20260923))
    set_seed(seed)
    data_path = _expand_path(config["data"]["aligned_path"])
    _validate_label_source(data_path, split)
    max_samples = args.max_samples
    if args.mode == "smoke":
        max_samples = max_samples or int(config.get("smoke", {}).get("max_samples", 16))
    dataset = MultimodalDataset(data_path, split=split, max_samples=max_samples)
    if len(dataset) == 0:
        raise ValueError("Selected split is empty")
    labels = np.asarray(dataset.records["classification"][: len(dataset)])
    regression = np.asarray(dataset.records["regression"][: len(dataset)])
    if np.any(labels < 0) or not np.isfinite(regression).all():
        raise ValueError("The selected aligned-features split must contain both official labels")

    conditions = conditions_from_grid(config[grid_name], seed=seed)
    if args.mode == "smoke":
        smoke_count = int(config.get("smoke", {}).get("max_conditions", 4))
        conditions = conditions[:smoke_count]
    if args.max_conditions is not None:
        conditions = conditions[: args.max_conditions]
    if not conditions or conditions[0].missing_rate != 0:
        raise ValueError("Every evaluation must start with the clean condition")

    statistics = config.get("statistics", {})
    resamples = int(statistics.get("bootstrap_resamples", 2000))
    if args.mode == "smoke":
        resamples = int(config.get("smoke", {}).get("bootstrap_resamples", 50))
    confidence = float(statistics.get("confidence", 0.95))
    batch_size = int(config.get("evaluation", {}).get("batch_size", 64))
    num_workers = int(config.get("evaluation", {}).get("num_workers", 0))
    device = _choose_device(args.device or config.get("device", "auto"))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.mode == "smoke":
        output_dir = Path("runs") / "robustness_smoke" / split
    else:
        output_dir = Path(config.get("output", {}).get("metrics_dir", "artifacts/metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"robustness_{split}.csv"
    predictions_path = output_dir / f"robustness_{split}_clean_predictions.csv"
    manifest_path = output_dir / f"robustness_{split}_manifest.json"

    metric_rows: list[dict[str, Any]] = []
    clean_prediction_rows: list[dict[str, Any]] = []
    checkpoint_manifest: list[dict[str, Any]] = []
    for model_entry in _model_entries(config):
        checkpoint_path = _expand_path(str(model_entry["checkpoint"]))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint_path}")
        checkpoint_hash = _sha256(checkpoint_path)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state" not in state or "config" not in state:
            raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")
        model = build_model(state["config"])
        model.load_state_dict(state["model_state"], strict=True)
        trainer = Trainer(model, state["config"], device=device)
        model_name = str(model_entry["name"])
        model_kind = str(state["config"].get("model", {}).get("type", "unknown"))
        model_seed = int(model_entry.get("seed", state["config"].get("seed", -1)))
        checkpoint_manifest.append(
            {
                "name": model_name,
                "family": str(model_entry.get("family", model_name)),
                "role": str(model_entry.get("role", "model")),
                "seed": model_seed,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
            }
        )
        for number, condition in enumerate(conditions, start=1):
            view = RobustnessDataset(dataset, condition)
            loader = DataLoader(
                view, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=False,
            )
            prediction = trainer.predict(loader)
            metrics = compute_metrics_with_group_ci(
                prediction,
                n_resamples=resamples,
                confidence=confidence,
                seed=int(np.random.SeedSequence([seed, number - 1]).generate_state(1)[0]),
            )
            row: dict[str, Any] = {
                "model": model_name,
                "family": str(model_entry.get("family", model_name)),
                "role": str(model_entry.get("role", "model")),
                "model_type": model_kind,
                "training_seed": model_seed,
                "split": split,
                "checkpoint_sha256": checkpoint_hash,
                **_condition_metadata(condition),
                **metrics,
            }
            metric_rows.append(row)
            if condition.missing_rate == 0:
                for i, sample_id in enumerate(prediction["ids"]):
                    clean_prediction_rows.append(
                        {
                            "model": model_name,
                            "family": row["family"],
                            "role": row["role"],
                            "training_seed": model_seed,
                            "split": split,
                            "id": sample_id,
                            "video_id": str(sample_id).split("_", 1)[0],
                            "true_class": prediction["true_class"][i],
                            "predicted_class": prediction["predicted_class"][i],
                            "true_regression": prediction["true_regression"][i],
                            "predicted_regression": prediction["predicted_regression"][i],
                        }
                    )
            print(
                f"[{model_name}] {number}/{len(conditions)} {condition.code} "
                f"macro_f1={metrics['macro_f1']:.6f} mae={metrics['mae']:.6f}",
                flush=True,
            )

    _write_csv(metrics_path, metric_rows)
    _write_csv(predictions_path, clean_prediction_rows)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "split": split,
        "frozen_test_confirmation": bool(args.confirm_frozen),
        "label_source_policy": "aligned_features_only; no evaluation_set_labels_or_overlap_lookup",
        "data_file": str(data_path),
        "data_sha256": _sha256(data_path),
        "config_file": str(config_path),
        "config_sha256": _sha256(config_path),
        "condition_grid": grid_name,
        "condition_count": len(conditions),
        "sample_count": len(dataset),
        "bootstrap_resamples": resamples,
        "confidence": confidence,
        "bootstrap_unit": "video_id_prefix_before_first_underscore",
        "device": device,
        "checkpoints": checkpoint_manifest,
        "outputs": {
            "metrics": str(metrics_path),
            "clean_predictions": str(predictions_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_path, predictions_path, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-samples", type=int, help="Validation smoke/debug only")
    parser.add_argument("--max-conditions", type=int, help="Validation smoke/debug only")
    parser.add_argument(
        "--confirm-frozen", action="store_true",
        help="Required for test: confirms model/checkpoint/grid were frozen using validation only",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = evaluate(args)
    print("outputs=" + ",".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
