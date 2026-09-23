"""CLI: python -m src.training.cli --config configs/model_gated.yaml"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MultimodalDataset
from src.models import build_model
from src.training.trainer import Trainer, compute_selection_score, set_seed


def load_config(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use env/run.sh for project GPU libraries")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test cap per split")
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 2026))
    set_seed(seed)
    data_cfg, train_cfg = config["data"], config.get("training", {})
    path = os.path.expandvars(data_cfg["aligned_path"])
    train_data = MultimodalDataset(path, split=data_cfg.get("train_split", "train"), max_samples=args.max_samples)
    valid_data = MultimodalDataset(path, split=data_cfg.get("valid_split", "valid"), max_samples=args.max_samples)
    loader_args = {
        "batch_size": int(train_cfg.get("batch_size", 32)),
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": False,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_args)
    device = choose_device(args.device or config.get("device", "auto"))
    model = build_model(config)
    trainer = Trainer(model, config, device=device)
    checkpoint = Path(os.path.expandvars(config.get("output", {}).get("checkpoint", "artifacts/models/model.pt")))
    metrics_path = Path(os.path.expandvars(config.get("output", {}).get("metrics", "artifacts/metrics/training.json")))
    if args.max_samples is not None:
        smoke_root = Path("runs") / f"smoke_{config.get('model', {}).get('type', 'model')}_{args.max_samples}"
        checkpoint = smoke_root / "artifacts/models/model.pt"
        metrics_path = smoke_root / "artifacts/metrics/training.json"
    selection_weights = train_cfg.get(
        "selection_weights",
        {"macro_f1": 0.35, "accuracy": 0.15, "pearson": 0.25, "mae": 0.25},
    )
    best = float("-inf")
    history = []
    for epoch in range(1, int(train_cfg.get("epochs", 10)) + 1):
        train_metrics = trainer.train_epoch(train_loader)
        valid_metrics = trainer.evaluate(valid_loader)
        score = compute_selection_score(valid_metrics, selection_weights)
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "selection_score": score,
        }
        history.append(row)
        if score > best:
            best = score
            trainer.save(checkpoint, epoch, valid_metrics)
        print(json.dumps(row, ensure_ascii=False, allow_nan=True))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(history, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    print(f"checkpoint={checkpoint} metrics={metrics_path} device={device}")


if __name__ == "__main__":
    main()
