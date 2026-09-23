"""Real-checkpoint adapter for mask-correct three-modality explanations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np
import torch

from src.models import build_model

MODALITIES = ("text", "audio", "vision")
CLASS_NAMES = ("Negative", "Neutral", "Positive")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class GatedCheckpointAdapter:
    """Expose a trained gated model through the explainer's mapping API.

    Each call derives ``observed`` again from the perturbed tensors and intersects
    it with the sample's original observed mask. Consequently a modality omitted
    from a coalition is both zero-valued *and* unobserved. This is important for
    the gated model because merely replacing a tensor by zero while retaining its
    availability bit would describe a different intervention.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        expected_sha256: Optional[str] = None,
        zero_atol: float = 1e-12,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.checkpoint_sha256 = sha256_file(self.checkpoint)
        if expected_sha256 and self.checkpoint_sha256.lower() != expected_sha256.lower():
            raise RuntimeError("checkpoint SHA-256 does not match the frozen model reference")
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(state, Mapping) or "model_state" not in state or "config" not in state:
            raise ValueError("checkpoint must contain model_state and config")
        config = dict(state["config"])
        model_cfg = config.get("model", {})
        if model_cfg.get("type", "gated") != "gated":
            raise ValueError("Explanation integration requires a gated checkpoint")
        modalities = tuple(model_cfg.get("modalities", MODALITIES))
        if modalities != MODALITIES:
            raise ValueError("Explanation integration requires text/audio/vision in canonical order")
        self.device = torch.device(device)
        self.model = build_model(config).to(self.device)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.eval()
        self.config = config
        self.epoch = int(state.get("epoch", -1))
        self.validation_metrics = dict(state.get("metrics", {}))
        self.zero_atol = float(zero_atol)

    @staticmethod
    def sample_from_item(item: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one ``MultimodalDataset`` item to an explainer sample."""
        valid = np.asarray(item["valid"].detach().cpu(), dtype=bool)
        observed: Dict[str, np.ndarray] = {}
        sample: Dict[str, Any] = {}
        for modality in MODALITIES:
            values = np.asarray(item["features"][modality].detach().cpu(), dtype=np.float32).copy()
            mask = np.asarray(item["observed"][modality].detach().cpu(), dtype=bool) & valid
            values[~mask] = 0.0
            sample[modality] = values
            observed[modality] = mask
        sample["_observed"] = observed
        sample["_valid"] = valid
        if "text_bert" in item:
            sample["_text_bert"] = np.asarray(item["text_bert"].detach().cpu(), dtype=np.int64)
        sample["id"] = str(item.get("id", ""))
        sample["raw_text"] = str(item.get("raw_text", ""))
        return sample

    def _tensors(self, sample: Mapping[str, Any]) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        first = np.asarray(sample[MODALITIES[0]])
        if first.ndim != 2:
            raise ValueError("each modality must be a [positions, features] array")
        length = first.shape[0]
        valid = np.asarray(sample.get("_valid", np.ones(length, dtype=bool)), dtype=bool)
        if valid.shape != (length,):
            raise ValueError("_valid must match the sequence length")
        original_observed = sample.get("_observed", {})
        features: Dict[str, torch.Tensor] = {}
        observed: Dict[str, torch.Tensor] = {}
        for modality in MODALITIES:
            values = np.asarray(sample[modality], dtype=np.float32)
            if values.ndim != 2 or values.shape[0] != length:
                raise ValueError(f"{modality} must share the sequence length")
            finite_rows = np.isfinite(values).all(axis=-1)
            nonzero_rows = np.max(np.abs(np.nan_to_num(values)), axis=-1) > self.zero_atol
            base = np.asarray(original_observed.get(modality, valid), dtype=bool)
            if base.shape != (length,):
                raise ValueError(f"_observed[{modality!r}] must match the sequence length")
            mask = valid & base & finite_rows & nonzero_rows
            clean = np.nan_to_num(values, copy=True)
            clean[~mask] = 0.0
            features[modality] = torch.as_tensor(clean[None], device=self.device)
            observed[modality] = torch.as_tensor(mask[None], device=self.device)
        return features, observed, torch.as_tensor(valid[None], device=self.device)

    @torch.inference_mode()
    def predict_many(self, samples: list[Mapping[str, Any]], batch_size: int = 512) -> list[Dict[str, Any]]:
        """Predict many perturbations in batches without changing mask semantics."""
        results: list[Dict[str, Any]] = []
        for start in range(0, len(samples), batch_size):
            part = samples[start : start + batch_size]
            prepared = [self._tensors(sample) for sample in part]
            features = {
                modality: torch.cat([item[0][modality] for item in prepared], dim=0)
                for modality in MODALITIES
            }
            observed = {
                modality: torch.cat([item[1][modality] for item in prepared], dim=0)
                for modality in MODALITIES
            }
            valid = torch.cat([item[2] for item in prepared], dim=0)
            output = self.model(features, observed, valid)
            logits = output["logits"].float().cpu().numpy()
            probabilities = torch.softmax(output["logits"].float(), dim=-1).cpu().numpy()
            regression = output["regression"].float().cpu().numpy()
            gates = output.get("gates")
            gates_array = gates.float().cpu().numpy() if gates is not None else None
            for i in range(len(part)):
                result: Dict[str, Any] = {
                    "logits": logits[i], "probabilities": probabilities[i],
                    "regression": float(regression[i]),
                    "observed_counts": {
                        modality: int(observed[modality][i].sum().cpu()) for modality in MODALITIES
                    },
                }
                if gates_array is not None:
                    result["gates"] = gates_array[i]
                results.append(result)
        return results

    def predict(self, sample: Mapping[str, Any]) -> Dict[str, Any]:
        return self.predict_many([sample], batch_size=1)[0]

    def fixed_class_target(self, sample: Mapping[str, Any], score: str = "probability") -> Dict[str, Any]:
        prediction = self.predict(sample)
        class_index = int(np.argmax(prediction["probabilities"]))
        return {
            "class_index": class_index,
            "class_name": CLASS_NAMES[class_index],
            "score": score,
            "selector": self.classification_selector(class_index, score),
            "prediction": prediction,
        }

    @staticmethod
    def classification_selector(class_index: int, score: str = "probability") -> Callable[[Mapping[str, Any]], float]:
        class_index = int(class_index)
        if class_index not in range(len(CLASS_NAMES)):
            raise ValueError("class_index must be 0, 1, or 2")
        key = "probabilities" if score == "probability" else "logits" if score == "logit" else None
        if key is None:
            raise ValueError("classification score must be 'probability' or 'logit'")
        return lambda output: float(np.asarray(output[key])[class_index])

    @staticmethod
    def regression_selector(output: Mapping[str, Any]) -> float:
        return float(output["regression"])

    def assert_checkpoint_unchanged(self) -> None:
        """Fail if the model file changed during a supposedly frozen run."""
        if sha256_file(self.checkpoint) != self.checkpoint_sha256:
            raise RuntimeError("checkpoint changed during explanation inference")

    @property
    def model_reference(self) -> str:
        return f"{self.checkpoint.name}@sha256:{self.checkpoint_sha256}"
