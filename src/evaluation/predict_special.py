"""Label-safe prediction CLI for evaluation-set aligned feature files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import BertTextProvider, MultimodalDataset
from src.models import build_model
from src.training.trainer import Trainer


def discover(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("*.pkl") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="A .pkl or directory of aligned .pkl files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bert-device", default="cpu")
    parser.add_argument("--bert-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(state["config"])
    model.load_state_dict(state["model_state"])
    trainer = Trainer(model, state["config"], args.device)
    provider = BertTextProvider(
        model_name=args.bert_model,
        cache_dir=args.cache_dir,
        device=args.bert_device,
        local_files_only=args.local_files_only,
    )
    rows = []
    for path in discover(Path(args.input)):
        dataset = MultimodalDataset(path, text_provider=provider)
        prediction = trainer.predict(DataLoader(dataset, batch_size=32, shuffle=False))
        text_source = dataset.records["text_source"]
        for i, sample_id in enumerate(prediction["ids"]):
            row = {
                "source_file": path.name,
                "id": sample_id,
                "class_index": prediction["predicted_class"][i],
                "sentiment": prediction["predicted_regression"][i],
                "text_source": text_source,
            }
            if prediction["gates"]:
                for name, weight in zip(("text", "audio", "vision"), prediction["gates"][i]):
                    row[f"gate_{name}"] = weight
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No .pkl files found under {args.input}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} label-free predictions to {output}")


if __name__ == "__main__":
    main()
