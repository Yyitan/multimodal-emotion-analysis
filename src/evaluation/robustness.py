"""Formal robustness evaluation primitives for the robust prediction module.

This module is deliberately label-safe: it only transforms already-loaded feature
sequences and computes statistics from supplied predictions. It never searches
other data splits, IDs, or text for labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.evaluation.metrics import compute_metrics

MODALITIES: Tuple[str, ...] = ("text", "audio", "vision")
MODALITY_CODES = {"T": "text", "A": "audio", "V": "vision"}
POSITIONS: Tuple[str, ...] = ("head", "middle", "tail", "random")
METRIC_NAMES: Tuple[str, ...] = (
    "accuracy", "macro_f1", "weighted_f1", "mae", "pearson"
)


def normalize_modalities(value: str | Iterable[str]) -> Tuple[str, ...]:
    """Convert T/A/V combinations or long names to a canonical tuple."""
    if isinstance(value, str):
        compact = value.strip().upper().replace("+", "").replace(",", "")
        if compact in {"", "NONE", "CLEAN"}:
            return ()
        if set(compact).issubset(MODALITY_CODES):
            requested = [MODALITY_CODES[c] for c in "TAV" if c in compact]
        else:
            requested = [part.strip().lower() for part in value.split(",")]
    else:
        requested = [str(part).strip().lower() for part in value]
    unknown = set(requested).difference(MODALITIES)
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    return tuple(name for name in MODALITIES if name in requested)


def modality_code(modalities: str | Iterable[str]) -> str:
    names = normalize_modalities(modalities)
    reverse = {name: code for code, name in MODALITY_CODES.items()}
    return "".join(reverse[name] for name in names) or "NONE"


@dataclass(frozen=True)
class MissingCondition:
    """One pre-specified missingness condition."""

    modalities: Tuple[str, ...]
    missing_rate: float
    position: str
    n_spans: int
    seed: int = 20260923

    def __post_init__(self) -> None:
        object.__setattr__(self, "modalities", normalize_modalities(self.modalities))
        if not 0.0 <= float(self.missing_rate) <= 1.0:
            raise ValueError("missing_rate must be in [0, 1]")
        if self.position not in POSITIONS and float(self.missing_rate) > 0:
            raise ValueError(f"position must be one of {POSITIONS}")
        if int(self.n_spans) < 1 and float(self.missing_rate) > 0:
            raise ValueError("n_spans must be positive")

    @property
    def code(self) -> str:
        if self.missing_rate == 0 or not self.modalities:
            return "clean_r0"
        rate = f"{self.missing_rate:.6g}".replace(".", "p")
        return f"{modality_code(self.modalities)}_r{rate}_{self.position}_s{self.n_spans}_z{self.seed}"


def _balanced_lengths(total: int, parts: int) -> np.ndarray:
    result = np.full(parts, total // parts, dtype=np.int64)
    result[: total % parts] += 1
    return result


def _rank_spans(
    n_valid: int,
    n_missing: int,
    position: str,
    n_spans: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """Create disjoint half-open spans in valid-position rank coordinates."""
    if n_missing <= 0:
        return []
    parts = min(max(1, int(n_spans)), n_missing, n_valid - n_missing + 1)
    lengths = _balanced_lengths(n_missing, parts)
    minimum_internal = max(parts - 1, 0)
    free = n_valid - n_missing - minimum_internal
    gaps = np.zeros(parts + 1, dtype=np.int64)
    if parts > 1:
        gaps[1:parts] = 1

    if position == "head":
        gaps[-1] += free
    elif position == "tail":
        gaps[0] += free
    elif position == "middle":
        gaps[0] += free // 2
        gaps[-1] += free - free // 2
    elif position == "random":
        if free:
            gaps += rng.multinomial(free, np.full(parts + 1, 1.0 / (parts + 1)))
    else:
        raise ValueError(f"Unknown position: {position}")

    spans: list[tuple[int, int]] = []
    cursor = int(gaps[0])
    for i, length in enumerate(lengths):
        end = cursor + int(length)
        spans.append((cursor, end))
        cursor = end + int(gaps[i + 1])
    return spans


def build_span_mask(
    valid: Sequence[bool] | np.ndarray | torch.Tensor,
    missing_rate: float,
    position: str = "random",
    n_spans: int = 1,
    seed: int = 20260923,
) -> np.ndarray:
    """Return a boolean missing mask, changing valid positions only.

    The requested count is round-half-up(n_valid * missing_rate). For a positive
    rate on a non-empty sequence at least one point is selected. Spans are
    contiguous in valid-position rank coordinates; padding is never selected.
    """
    if isinstance(valid, torch.Tensor):
        valid_array = valid.detach().cpu().numpy().astype(bool, copy=False)
    else:
        valid_array = np.asarray(valid, dtype=bool)
    if valid_array.ndim != 1:
        raise ValueError(f"valid must be one-dimensional, got {valid_array.shape}")
    if not 0.0 <= float(missing_rate) <= 1.0:
        raise ValueError("missing_rate must be in [0, 1]")
    mask = np.zeros(valid_array.shape, dtype=bool)
    positions = np.flatnonzero(valid_array)
    if len(positions) == 0 or missing_rate == 0:
        return mask
    n_missing = min(len(positions), max(1, int(np.floor(len(positions) * missing_rate + 0.5))))
    rng = np.random.default_rng(int(seed))
    for start, end in _rank_spans(len(positions), n_missing, position, n_spans, rng):
        mask[positions[start:end]] = True
    if np.any(mask & ~valid_array):
        raise AssertionError("internal error: missing mask touched padding")
    return mask


def mask_intervals(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """Return physical half-open intervals, useful for audits and tests."""
    values = np.asarray(mask, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def apply_span_missingness(
    features: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor],
    valid: torch.Tensor,
    modalities: str | Iterable[str],
    missing_rate: float,
    position: str = "random",
    n_spans: int = 1,
    seed: int = 20260923,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
    """Clone and mask a sample or batch while preserving every padding value."""
    names = normalize_modalities(modalities)
    squeeze = valid.ndim == 1
    batch_valid = valid.unsqueeze(0) if squeeze else valid
    if batch_valid.ndim != 2:
        raise ValueError("valid must have shape [time] or [batch, time]")
    result_features = {name: value.clone() for name, value in features.items()}
    result_observed = {name: value.clone() for name, value in observed.items()}
    masks = []
    for index in range(batch_valid.shape[0]):
        sample_seed = int(np.random.SeedSequence([int(seed), index]).generate_state(1)[0])
        masks.append(build_span_mask(batch_valid[index], missing_rate, position, n_spans, sample_seed))
    missing = torch.as_tensor(np.stack(masks), dtype=torch.bool, device=batch_valid.device)
    for name in names:
        if name not in result_features or name not in result_observed:
            raise KeyError(f"Modality {name!r} is unavailable")
        feature_mask = missing[0] if squeeze else missing
        result_features[name][feature_mask] = 0
        result_observed[name][feature_mask] = False
    return result_features, result_observed, missing[0] if squeeze else missing


class RobustnessDataset(Dataset):
    """Deterministic view applying one condition to an existing labeled dataset."""

    def __init__(self, base: Dataset, condition: MissingCondition) -> None:
        self.base = base
        self.condition = condition

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        original = self.base[index]
        item = dict(original)
        item["features"] = {name: value.clone() for name, value in original["features"].items()}
        item["observed"] = {name: value.clone() for name, value in original["observed"].items()}
        sample_seed = int(
            np.random.SeedSequence([int(self.condition.seed), int(index)]).generate_state(1)[0]
        )
        missing = build_span_mask(
            original["valid"], self.condition.missing_rate, self.condition.position,
            self.condition.n_spans, sample_seed,
        )
        missing_tensor = torch.as_tensor(missing, dtype=torch.bool)
        for name in self.condition.modalities:
            item["features"][name][missing_tensor] = 0
            item["observed"][name][missing_tensor] = False
        item["robustness_mask"] = missing_tensor
        return item


def video_group(sample_id: Any) -> str:
    """Competition grouping unit: prefix before the first underscore."""
    return str(sample_id).split("_", 1)[0]


def iter_group_bootstrap_indices(
    groups: Sequence[Any], n_resamples: int = 2000, seed: int = 20260923
) -> Iterator[np.ndarray]:
    """Yield row indices after sampling whole groups with replacement."""
    group_array = np.asarray(groups).astype(str)
    if group_array.ndim != 1 or len(group_array) == 0:
        raise ValueError("groups must be a non-empty one-dimensional sequence")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    unique = np.asarray(list(dict.fromkeys(group_array.tolist())), dtype=str)
    if len(unique) < 2:
        raise ValueError("at least two video groups are required for a confidence interval")
    members = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_resamples)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        yield np.concatenate([members[group] for group in sampled])


def compute_metrics_with_group_ci(
    prediction: Mapping[str, Sequence[Any]],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260923,
) -> Dict[str, float | int | str]:
    """Compute required metrics and percentile CIs by video-level bootstrap."""
    required = (
        "ids", "true_class", "predicted_class", "true_regression", "predicted_regression"
    )
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError(f"Prediction is missing fields: {missing}")
    arrays = {key: np.asarray(prediction[key]) for key in required}
    n = len(arrays["ids"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("All prediction fields must have equal length")
    point = compute_metrics(
        arrays["true_class"], arrays["predicted_class"],
        arrays["true_regression"], arrays["predicted_regression"],
    )
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if int(n_resamples) < 1:
        raise ValueError("n_resamples must be positive")
    groups = np.asarray([video_group(value) for value in arrays["ids"]])
    unique = np.asarray(list(dict.fromkeys(groups.tolist())), dtype=str)
    if len(unique) < 2:
        raise ValueError("at least two video groups are required for a confidence interval")
    group_index = np.asarray([np.flatnonzero(unique == group)[0] for group in groups])
    rng = np.random.default_rng(int(seed))
    # Sampling G groups with replacement is exactly multinomial group counts.
    # Sufficient statistics then compute all bootstrap metrics at once, rather
    # than making thousands of repeated sklearn calls per matrix condition.
    weights = rng.multinomial(
        len(unique), np.full(len(unique), 1.0 / len(unique)), size=int(n_resamples)
    ).astype(float)
    samples: Dict[str, np.ndarray] = {}

    true_class = arrays["true_class"].astype(int)
    predicted_class = arrays["predicted_class"].astype(int)
    class_mask = true_class >= 0
    if class_mask.any():
        labels = np.unique(np.r_[true_class[class_mask], predicted_class[class_mask]])
        lookup = {int(label): i for i, label in enumerate(labels)}
        confusion = np.zeros((len(unique), len(labels), len(labels)), dtype=float)
        for row in np.flatnonzero(class_mask):
            confusion[group_index[row], lookup[int(true_class[row])], lookup[int(predicted_class[row])]] += 1
        boot_confusion = np.einsum("bg,gij->bij", weights, confusion, optimize=True)
        true_support = boot_confusion.sum(axis=2)
        predicted_support = boot_confusion.sum(axis=1)
        true_positive = np.diagonal(boot_confusion, axis1=1, axis2=2)
        denominator = true_support + predicted_support
        per_class_f1 = np.divide(
            2.0 * true_positive, denominator,
            out=np.zeros_like(true_positive), where=denominator > 0,
        )
        active = denominator > 0
        samples["accuracy"] = np.divide(
            true_positive.sum(axis=1), true_support.sum(axis=1),
            out=np.full(len(weights), np.nan), where=true_support.sum(axis=1) > 0,
        )
        samples["macro_f1"] = np.divide(
            (per_class_f1 * active).sum(axis=1), active.sum(axis=1),
            out=np.full(len(weights), np.nan), where=active.sum(axis=1) > 0,
        )
        samples["weighted_f1"] = np.divide(
            (per_class_f1 * true_support).sum(axis=1), true_support.sum(axis=1),
            out=np.full(len(weights), np.nan), where=true_support.sum(axis=1) > 0,
        )
    else:
        for name in ("accuracy", "macro_f1", "weighted_f1"):
            samples[name] = np.full(len(weights), np.nan)

    true_regression = arrays["true_regression"].astype(float)
    predicted_regression = arrays["predicted_regression"].astype(float)
    regression_mask = np.isfinite(true_regression) & np.isfinite(predicted_regression)
    if regression_mask.any():
        # Per-group n, absolute error, x, y, x^2, y^2, and xy suffice for MAE/Pearson.
        stats = np.zeros((len(unique), 7), dtype=float)
        for row in np.flatnonzero(regression_mask):
            x, y = true_regression[row], predicted_regression[row]
            stats[group_index[row]] += (1.0, abs(x - y), x, y, x * x, y * y, x * y)
        boot = weights @ stats
        count, absolute, sx, sy, sxx, syy, sxy = boot.T
        samples["mae"] = absolute / count
        covariance = sxy - sx * sy / count
        variance_x = sxx - sx * sx / count
        variance_y = syy - sy * sy / count
        denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
        samples["pearson"] = np.divide(
            covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
        )
    else:
        samples["mae"] = np.full(len(weights), np.nan)
        samples["pearson"] = np.full(len(weights), np.nan)

    alpha = (1.0 - float(confidence)) / 2.0
    result: Dict[str, float | int | str] = dict(point)
    for name in METRIC_NAMES:
        finite = np.asarray(samples[name], dtype=float)
        finite = finite[np.isfinite(finite)]
        result[f"{name}_ci_lower"] = float(np.quantile(finite, alpha)) if len(finite) else float("nan")
        result[f"{name}_ci_upper"] = float(np.quantile(finite, 1.0 - alpha)) if len(finite) else float("nan")
    class_values = arrays["true_class"].astype(float)
    regression_values = arrays["true_regression"].astype(float)
    result.update(
        {
            "n_samples": int(n),
            "n_classification": int(np.sum(class_values >= 0)),
            "n_regression": int(np.sum(np.isfinite(regression_values))),
            "n_video_groups": int(len(np.unique(groups))),
            "bootstrap_resamples": int(n_resamples),
            "confidence": float(confidence),
            "bootstrap_unit": "video_id_prefix",
        }
    )
    return result


def conditions_from_grid(grid: Mapping[str, Any], seed: int = 20260923) -> list[MissingCondition]:
    """Expand a YAML grid, retaining a single non-duplicated clean condition."""
    modalities = grid.get("modalities", ["T", "A", "V", "TA", "TV", "AV", "TAV"])
    rates = [float(value) for value in grid.get("missing_rates", [0, .1, .2, .3, .5, .7])]
    positions = grid.get("positions", list(POSITIONS))
    span_counts = [int(value) for value in grid.get("span_counts", [1, 3])]
    result: list[MissingCondition] = []
    if any(rate == 0 for rate in rates):
        result.append(MissingCondition((), 0.0, "head", 1, int(seed)))
    for names in modalities:
        for rate in rates:
            if rate == 0:
                continue
            for position in positions:
                for count in span_counts:
                    result.append(
                        MissingCondition(normalize_modalities(names), rate, str(position), count, int(seed))
                    )
    return result
