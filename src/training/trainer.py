"""Deterministic training/evaluation loop for lightweight multi-task models."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from src.evaluation.metrics import compute_metrics
from .augment import continuous_span_mask
from .losses import consistency_loss, supervised_multitask_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def compute_selection_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Validation-only joint score with every task metric normalized to [0, 1]."""
    expected = ("macro_f1", "accuracy", "pearson", "mae")
    if set(weights) != set(expected):
        raise ValueError(f"selection weights must contain exactly {sorted(expected)}")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError("selection weights must be non-negative")
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ValueError("selection weights must sum to 1.0")

    def finite_or_worst(name: str, worst: float) -> float:
        value = float(metrics.get(name, worst))
        return value if np.isfinite(value) else worst

    normalized = {
        "macro_f1": float(np.clip(finite_or_worst("macro_f1", 0.0), 0.0, 1.0)),
        "accuracy": float(np.clip(finite_or_worst("accuracy", 0.0), 0.0, 1.0)),
        "pearson": float(np.clip((finite_or_worst("pearson", -1.0) + 1.0) / 2.0, 0.0, 1.0)),
        "mae": float(1.0 - np.clip(finite_or_worst("mae", 6.0), 0.0, 6.0) / 6.0),
    }
    return float(sum(float(weights[name]) * normalized[name] for name in expected))


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    result = dict(batch)
    result["features"] = {k: v.to(device) for k, v in batch["features"].items()}
    result["observed"] = {k: v.to(device) for k, v in batch["observed"].items()}
    for key in ("valid", "classification", "regression", "text_bert"):
        result[key] = batch[key].to(device)
    return result


class Trainer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any], device: str = "cpu") -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        train_cfg = config.get("training", {})
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.augmentation = train_cfg.get("augmentation", {})
        self.consistency_weight = float(train_cfg.get("consistency_weight", 0.2))
        self.classification_weight = float(train_cfg.get("classification_weight", 1.0))
        self.regression_weight = float(train_cfg.get("regression_weight", 1.0))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 5.0))

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "classification_loss": 0.0, "regression_loss": 0.0, "consistency_loss": 0.0}
        count = 0
        for original in loader:
            batch = _move(original, self.device)
            clean = self.model(batch["features"], batch["observed"], batch["valid"])
            losses = supervised_multitask_loss(
                clean,
                batch["classification"],
                batch["regression"],
                self.classification_weight,
                self.regression_weight,
            )
            consistency = clean["logits"].sum() * 0.0
            if self.consistency_weight > 0 and self.augmentation.get("enabled", True):
                augmented_features, augmented_observed = continuous_span_mask(
                    batch["features"],
                    batch["observed"],
                    batch["valid"],
                    probability=float(self.augmentation.get("probability", 0.5)),
                    min_fraction=float(self.augmentation.get("min_fraction", 0.1)),
                    max_fraction=float(self.augmentation.get("max_fraction", 0.4)),
                    modalities=self.augmentation.get("modalities", ["text", "audio", "vision"]),
                )
                augmented = self.model(augmented_features, augmented_observed, batch["valid"])
                consistency = consistency_loss(clean, augmented)
            total = losses["loss"] + self.consistency_weight * consistency
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            batch_size = batch["classification"].shape[0]
            totals["loss"] += float(total.detach()) * batch_size
            totals["classification_loss"] += float(losses["classification_loss"].detach()) * batch_size
            totals["regression_loss"] += float(losses["regression_loss"].detach()) * batch_size
            totals["consistency_loss"] += float(consistency.detach()) * batch_size
            count += batch_size
        return {k: v / max(count, 1) for k, v in totals.items()}

    @torch.inference_mode()
    def predict(self, loader) -> Dict[str, Any]:
        self.model.eval()
        result = {"ids": [], "true_class": [], "predicted_class": [], "true_regression": [], "predicted_regression": [], "gates": []}
        for original in loader:
            batch = _move(original, self.device)
            output = self.model(batch["features"], batch["observed"], batch["valid"])
            result["ids"].extend(original["id"])
            result["true_class"].extend(batch["classification"].cpu().tolist())
            result["predicted_class"].extend(output["logits"].argmax(-1).cpu().tolist())
            result["true_regression"].extend(batch["regression"].cpu().tolist())
            result["predicted_regression"].extend(output["regression"].clamp(-3, 3).cpu().tolist())
            if "gates" in output:
                result["gates"].extend(output["gates"].cpu().tolist())
        return result

    def evaluate(self, loader) -> Dict[str, float]:
        prediction = self.predict(loader)
        return compute_metrics(
            prediction["true_class"], prediction["predicted_class"],
            prediction["true_regression"], prediction["predicted_regression"],
        )

    def save(self, path: str | Path, epoch: int, metrics: Dict[str, float]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state": self.model.state_dict(), "config": self.config, "epoch": epoch, "metrics": metrics},
            path,
        )
