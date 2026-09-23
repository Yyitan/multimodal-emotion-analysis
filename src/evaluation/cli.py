"""Evaluate a checkpoint on a labeled aligned split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import MultimodalDataset
from src.models import build_model
from src.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"])
    trainer = Trainer(model, checkpoint["config"], args.device)
    dataset = MultimodalDataset(args.data, split=args.split, max_samples=args.max_samples)
    metrics = trainer.evaluate(DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
    text = json.dumps(metrics, indent=2, allow_nan=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
