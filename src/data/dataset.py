"""Unified, label-safe adapters for aligned the three aligned data sources feature files."""
from __future__ import annotations

import hashlib
import importlib
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

MODALITY_DIMS = {"text": 768, "audio": 74, "vision": 35}


def _load_pickle(path: Path) -> Any:
    """Load trusted competition pickles, including NumPy-2 files under NumPy 1.x."""
    # NumPy 2 renamed numpy.core to numpy._core. The explanation-set pickle uses the new path,
    # while the prepared runtime intentionally retains NumPy 1.x for compatibility.
    core = importlib.import_module("numpy.core")
    sys.modules.setdefault("numpy._core", core)
    for child in ("numeric", "multiarray", "_multiarray_umath"):
        try:
            sys.modules.setdefault(
                f"numpy._core.{child}", importlib.import_module(f"numpy.core.{child}")
            )
        except ModuleNotFoundError:
            pass
    with path.open("rb") as handle:
        return pickle.load(handle)


def _as_batch(value: Any, ndim: int) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == ndim - 1:
        value = value[None, ...]
    if value.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, got shape {value.shape}")
    return value


def validate_text_bert(value: Any) -> np.ndarray:
    """Return BERT token triplets as int64, accepting float storage only if integral."""
    tokens = _as_batch(value, 3)
    if not np.isfinite(tokens).all():
        raise ValueError("text_bert contains NaN or infinity")
    if np.issubdtype(tokens.dtype, np.floating) and not np.equal(tokens, np.rint(tokens)).all():
        raise ValueError("Floating text_bert contains non-integral token values")
    tokens = tokens.astype(np.int64, copy=False)
    if tokens.shape[1] != 3:
        raise ValueError(f"text_bert must have [input_ids, attention_mask, token_type_ids], got {tokens.shape}")
    if not np.isin(tokens[:, 1], (0, 1)).all():
        raise ValueError("text_bert attention_mask must be binary")
    return tokens


class BertTextProvider:
    """Reproduce supplied 768-D text features from their BERT token triplets.

    The model is loaded lazily. Results are cached under the project cache and no
    labels are read or used. `local_files_only` can enforce offline operation.
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        cache_dir: Optional[str | Path] = None,
        device: str = "cpu",
        batch_size: int = 16,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.local_files_only = local_files_only
        self._model = None

    def _cache_path(self, tokens: np.ndarray) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(self.model_name.encode() + tokens.tobytes()).hexdigest()
        return self.cache_dir / "bert_text" / f"{digest}.npy"

    def encode(self, text_bert: Any) -> np.ndarray:
        tokens = validate_text_bert(text_bert)
        cache_path = self._cache_path(tokens)
        if cache_path and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if cached.shape == (len(tokens), tokens.shape[-1], 768):
                return cached.astype(np.float32, copy=False)
        if self._model is None:
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(
                self.model_name, local_files_only=self.local_files_only
            ).to(self.device).eval()
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(tokens), self.batch_size):
                part = torch.as_tensor(tokens[start : start + self.batch_size], device=self.device)
                hidden = self._model(
                    input_ids=part[:, 0],
                    attention_mask=part[:, 1],
                    token_type_ids=part[:, 2],
                ).last_hidden_state
                outputs.append(hidden.float().cpu().numpy())
        result = np.concatenate(outputs).astype(np.float32, copy=False)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp.npy")
            np.save(tmp, result, allow_pickle=False)
            tmp.replace(cache_path)
        return result


def _normalise_mapping(mapping: Mapping[str, Any], source: Path, text_provider: Optional[BertTextProvider]) -> Dict[str, Any]:
    required = {"text_bert", "audio", "vision"}
    missing = required.difference(mapping)
    if missing:
        raise KeyError(f"{source}: missing required fields {sorted(missing)}")

    tokens = validate_text_bert(mapping["text_bert"])
    audio = _as_batch(mapping["audio"], 3)
    vision = _as_batch(mapping["vision"], 3)
    n, time = tokens.shape[0], tokens.shape[-1]
    if audio.shape[:2] != (n, time) or vision.shape[:2] != (n, time):
        raise ValueError(
            f"{source}: inconsistent aligned shapes: tokens={tokens.shape}, audio={audio.shape}, vision={vision.shape}"
        )
    if audio.shape[-1] != 74 or vision.shape[-1] != 35:
        raise ValueError(f"{source}: expected audio/vision dims 74/35")

    if "text" in mapping:
        text = _as_batch(mapping["text"], 3).astype(np.float32, copy=False)
        if text.shape != (n, time, 768):
            raise ValueError(f"{source}: unexpected text shape {text.shape}")
        text_source = "supplied_bert_last_hidden_state"
    else:
        if text_provider is None:
            raise ValueError(
                f"{source}: text is absent; pass BertTextProvider to reconstruct it from text_bert"
            )
        text = text_provider.encode(tokens)
        text_source = "reconstructed_bert_last_hidden_state"

    valid = tokens[:, 1].astype(bool)
    features = {
        "text": text,
        "audio": audio.astype(np.float32, copy=False),
        "vision": vision.astype(np.float32, copy=False),
    }
    observed = {"text": valid.copy()}
    for name in ("audio", "vision"):
        x = features[name]
        finite = np.isfinite(x).all(axis=-1)
        nonzero = np.max(np.abs(np.nan_to_num(x, copy=False)), axis=-1) > 1e-8
        # Crucially, valid comes from token attention, not from a zero suffix heuristic.
        observed[name] = valid & finite & nonzero
        features[name] = np.nan_to_num(x, copy=False)

    raw_ids = mapping.get("id", [source.stem] * n)
    if isinstance(raw_ids, (str, np.str_)):
        raw_ids = [str(raw_ids)]
    ids = [str(x) for x in raw_ids]
    if len(ids) != n:
        raise ValueError(f"{source}: id count {len(ids)} differs from sample count {n}")

    raw_text = mapping.get("raw_text", [""] * n)
    if isinstance(raw_text, (str, np.str_)):
        raw_text = [str(raw_text)]
    raw_text = [str(x) for x in raw_text]

    # Only canonical training label fields are accepted. IDs/text are never used to infer labels.
    classification = np.asarray(mapping.get("classification_labels", np.full(n, -100)), dtype=np.int64).reshape(-1)
    regression = np.asarray(mapping.get("regression_labels", np.full(n, np.nan)), dtype=np.float32).reshape(-1)
    if len(classification) != n or len(regression) != n:
        raise ValueError(f"{source}: label count differs from sample count")

    return {
        "features": features,
        "observed": observed,
        "valid": valid,
        "text_bert": tokens,
        "ids": ids,
        "raw_text": raw_text,
        "classification": classification,
        "regression": regression,
        "text_source": text_source,
        "source": str(source),
    }


def load_records(path: str | Path, split: Optional[str] = None, text_provider: Optional[BertTextProvider] = None) -> Dict[str, Any]:
    """Load aligned-features split, missing-modality wrapped test, or explanation-set direct dict."""
    path = Path(path)
    obj = _load_pickle(path)
    if not isinstance(obj, Mapping):
        raise TypeError(f"{path}: pickle root must be a mapping")
    if split is not None:
        if split not in obj:
            raise KeyError(f"{path}: split {split!r} not found; choices={list(obj)}")
        obj = obj[split]
    elif "test" in obj and isinstance(obj["test"], Mapping) and "text_bert" in obj["test"]:
        obj = obj["test"]
    elif "text_bert" not in obj:
        choices = [k for k, v in obj.items() if isinstance(v, Mapping)]
        raise ValueError(f"{path}: choose a split from {choices}")
    return _normalise_mapping(obj, path, text_provider)


class MultimodalDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        split: Optional[str] = None,
        text_provider: Optional[BertTextProvider] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.records = load_records(path, split=split, text_provider=text_provider)
        n = len(self.records["ids"])
        self.length = min(n, max_samples) if max_samples else n

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        r = self.records
        return {
            "features": {k: torch.from_numpy(v[index]).float() for k, v in r["features"].items()},
            "observed": {k: torch.from_numpy(v[index]).bool() for k, v in r["observed"].items()},
            "valid": torch.from_numpy(r["valid"][index]).bool(),
            "text_bert": torch.from_numpy(r["text_bert"][index]).long(),
            "classification": torch.tensor(r["classification"][index], dtype=torch.long),
            "regression": torch.tensor(r["regression"][index], dtype=torch.float32),
            "id": r["ids"][index],
            "raw_text": r["raw_text"][index] if index < len(r["raw_text"]) else "",
        }
