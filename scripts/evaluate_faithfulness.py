#!/usr/bin/env python3
"""Evaluate real explanation faithfulness on aligned-feature validation data."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from math import factorial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.data import MultimodalDataset
from src.explain.model_adapter import GatedCheckpointAdapter
from src.explain.shapley import MODALITIES, normalize_contributions

WINDOW_SIZES = {"text": 1, "audio": 3, "vision": 3}


def clone_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(sample)
    for modality in MODALITIES:
        result[modality] = np.asarray(sample[modality]).copy()
    result["_observed"] = {m: np.asarray(sample["_observed"][m]).copy() for m in MODALITIES}
    result["_valid"] = np.asarray(sample["_valid"]).copy()
    return result


def availability(sample: Mapping[str, Any]) -> Dict[str, bool]:
    return {m: bool(np.asarray(sample["_observed"][m], dtype=bool).any()) for m in MODALITIES}


def delete_windows(sample: Mapping[str, Any], modality: str, windows: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    result = clone_sample(sample)
    for start, end in windows:
        result[modality][start:end] = 0.0
        result["_observed"][modality][start:end] = False
    return result


def delete_modality(sample: Mapping[str, Any], modality: str) -> Dict[str, Any]:
    result = clone_sample(sample)
    result[modality].fill(0.0)
    result["_observed"][modality].fill(False)
    return result


def windows(length: int, size: int) -> List[Tuple[int, int]]:
    size = min(size, length)
    return [(i, i + size) for i in range(length - size + 1)]


def scalar_predictions(
    adapter: GatedCheckpointAdapter,
    samples: Sequence[Mapping[str, Any]],
    selector: Callable[[Mapping[str, Any]], float],
) -> np.ndarray:
    return np.asarray([selector(x) for x in adapter.predict_many(list(samples))], dtype=float)


def exact_shapley_batched(
    adapter: GatedCheckpointAdapter,
    sample: Mapping[str, Any],
    selector: Callable[[Mapping[str, Any]], float],
    available: Mapping[str, bool],
) -> Dict[str, Any]:
    coalitions = [frozenset(chosen) for size in range(4) for chosen in combinations(MODALITIES, size)]
    perturbed = []
    for coalition in coalitions:
        value = clone_sample(sample)
        for modality in MODALITIES:
            if modality not in coalition or not available[modality]:
                value[modality].fill(0.0)
                value["_observed"][modality].fill(False)
        perturbed.append(value)
    scores = scalar_predictions(adapter, perturbed, selector)
    values = dict(zip(coalitions, scores))
    phi: Dict[str, float] = {}
    for modality in MODALITIES:
        others = [m for m in MODALITIES if m != modality]
        contribution = 0.0
        for size in range(3):
            for chosen in combinations(others, size):
                coalition = frozenset(chosen)
                weight = factorial(size) * factorial(2 - size) / factorial(3)
                contribution += weight * (values[coalition | {modality}] - values[coalition])
        phi[modality] = float(contribution) if available[modality] else 0.0
    normalized = normalize_contributions(phi, available)
    residual = float(values[frozenset(MODALITIES)] - values[frozenset()] - sum(phi.values()))
    return {"signed_contributions": phi, "primary_modality": normalized["primary_modality"],
            "efficiency_residual": residual}


def score_local_windows(
    adapter: GatedCheckpointAdapter,
    sample: Mapping[str, Any],
    modality: str,
    length: int,
    selector: Callable[[Mapping[str, Any]], float],
    reference: float,
) -> List[Dict[str, Any]]:
    pairs = windows(length, WINDOW_SIZES[modality])
    values = scalar_predictions(adapter, [delete_windows(sample, modality, [pair]) for pair in pairs], selector)
    return [
        {"start": pair[0], "end": pair[1], "drop": float(reference - value),
         "importance": float(max(reference - value, 0.0))}
        for pair, value in zip(pairs, values)
    ]


def deletion_aopc(
    adapter: GatedCheckpointAdapter,
    sample: Mapping[str, Any],
    modality: str,
    order: Sequence[Tuple[int, int]],
    selector: Callable[[Mapping[str, Any]], float],
    reference: float,
) -> Tuple[float, List[float]]:
    cumulative = [order[:i] for i in range(1, len(order) + 1)]
    values = scalar_predictions(adapter, [delete_windows(sample, modality, x) for x in cumulative], selector)
    drops = (reference - values).tolist()
    return float(np.mean(drops)) if drops else 0.0, [float(x) for x in drops]


def mild_perturbation(sample: Mapping[str, Any], rng: np.random.RandomState, relative_scale: float) -> Dict[str, Any]:
    result = clone_sample(sample)
    for modality in MODALITIES:
        mask = np.asarray(result["_observed"][modality], dtype=bool)
        values = result[modality]
        observed_values = values[mask]
        scale = float(np.std(observed_values)) if observed_values.size else 0.0
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.max(np.abs(observed_values))) if observed_values.size else 0.0
        noise = rng.normal(0.0, relative_scale * max(scale, 1e-6), size=values.shape).astype(np.float32)
        values[mask] += noise[mask]
        values[~mask] = 0.0
    return result


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-15:
        return 1.0 if np.linalg.norm(x - y) <= 1e-15 else 0.0
    return float(np.dot(x, y) / denominator)


def jaccard_windows(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> float:
    left = {position for start, end in a for position in range(start, end)}
    right = {position for start, end in b for position in range(start, end)}
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def evaluate_target(
    adapter: GatedCheckpointAdapter,
    sample: Mapping[str, Any],
    selector: Callable[[Mapping[str, Any]], float],
    target_name: str,
    rng: np.random.RandomState,
    top_k: int,
    random_trials: int,
    noise_scale: float,
) -> Dict[str, Any]:
    available = availability(sample)
    full_prediction = adapter.predict(sample)
    reference = float(selector(full_prediction))
    shapley = exact_shapley_batched(adapter, sample, selector, available)
    ranked_modalities = sorted(
        [m for m in MODALITIES if available[m]],
        key=lambda m: (-abs(shapley["signed_contributions"][m]), MODALITIES.index(m)),
    )
    primary = ranked_modalities[0] if ranked_modalities else "none"
    secondary = ranked_modalities[1] if len(ranked_modalities) > 1 else "none"
    length = int(np.asarray(sample["_valid"], dtype=bool).sum())
    local: Dict[str, List[Dict[str, Any]]] = {}
    all_importance = []
    for modality in ranked_modalities:
        local[modality] = score_local_windows(adapter, sample, modality, length, selector, reference)
        all_importance.extend(x["importance"] for x in local[modality])
    primary_ranked = sorted(local.get(primary, []), key=lambda x: (-x["importance"], x["start"]))
    chosen = [(x["start"], x["end"]) for x in primary_ranked[: min(top_k, len(primary_ranked))]]
    ranked_aopc, ranked_curve = deletion_aopc(adapter, sample, primary, chosen, selector, reference) if chosen else (0.0, [])
    candidate_windows = [(x["start"], x["end"]) for x in local.get(primary, [])]
    random_orders = []
    for _ in range(random_trials):
        order = list(candidate_windows); rng.shuffle(order)
        random_orders.append(order[: len(chosen)])
    random_samples = [
        delete_windows(sample, primary, order[:step])
        for order in random_orders for step in range(1, len(order) + 1)
    ]
    random_values = scalar_predictions(adapter, random_samples, selector) if random_samples else np.empty(0)
    random_aopcs = (
        np.mean((reference - random_values).reshape(len(random_orders), len(chosen)), axis=1).tolist()
        if random_orders and chosen else []
    )
    random_mean = float(np.mean(random_aopcs)) if random_aopcs else 0.0

    deleted = [delete_modality(sample, m) for m in ranked_modalities[:2]]
    deletion_scores = scalar_predictions(adapter, deleted, selector) if deleted else np.empty(0)
    primary_drop = float(reference - deletion_scores[0]) if len(deletion_scores) else 0.0
    secondary_drop = float(reference - deletion_scores[1]) if len(deletion_scores) > 1 else 0.0

    positive = np.asarray(all_importance, dtype=float)
    positive_count = int(np.count_nonzero(positive > 0))
    top_mass = float(np.sort(positive)[::-1][:top_k].sum()) if positive.size else 0.0
    total_mass = float(positive.sum())
    selected_positions = {position for pair in chosen for position in range(*pair)}
    sparsity = {
        "positive_window_fraction": float(positive_count / len(positive)) if positive.size else 0.0,
        "top_k_importance_mass_fraction": float(top_mass / total_mass) if total_mass > 0 else 0.0,
        "primary_top_k_position_fraction": float(len(selected_positions) / length) if length else 0.0,
    }

    noisy = mild_perturbation(sample, rng, noise_scale)
    noisy_shapley = exact_shapley_batched(adapter, noisy, selector, available)
    original_vector = [shapley["signed_contributions"][m] for m in MODALITIES]
    noisy_vector = [noisy_shapley["signed_contributions"][m] for m in MODALITIES]
    noisy_reference = float(selector(adapter.predict(noisy)))
    noisy_local = score_local_windows(adapter, noisy, primary, length, selector, noisy_reference) if primary != "none" else []
    noisy_ranked = sorted(noisy_local, key=lambda x: (-x["importance"], x["start"]))
    noisy_chosen = [(x["start"], x["end"]) for x in noisy_ranked[: len(chosen)]]
    stability = {
        "noise_relative_scale": noise_scale,
        "shapley_cosine_similarity": cosine(original_vector, noisy_vector),
        "primary_modality_unchanged": primary == noisy_shapley["primary_modality"],
        "top_evidence_position_jaccard": jaccard_windows(chosen, noisy_chosen),
    }
    return {
        "target": target_name, "reference_score": reference,
        "primary_modality": primary, "secondary_modality": secondary,
        "shapley_text": float(shapley["signed_contributions"]["text"]),
        "shapley_audio": float(shapley["signed_contributions"]["audio"]),
        "shapley_vision": float(shapley["signed_contributions"]["vision"]),
        "efficiency_residual": float(shapley["efficiency_residual"]),
        "top_evidence_ranked_aopc": ranked_aopc,
        "random_deletion_aopc_mean": random_mean,
        "aopc_gain_over_random": float(ranked_aopc - random_mean),
        "primary_modality_deletion_drop": primary_drop,
        "secondary_modality_deletion_drop": secondary_drop,
        "primary_minus_secondary_drop": float(primary_drop - secondary_drop),
        "ranked_deletion_curve": ranked_curve,
        "random_trials": random_trials, "top_k": len(chosen),
        **sparsity, **stability,
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"sample_target_rows": len(rows), "targets": {}}
    numeric = [
        "top_evidence_ranked_aopc", "random_deletion_aopc_mean", "aopc_gain_over_random",
        "primary_modality_deletion_drop", "secondary_modality_deletion_drop", "primary_minus_secondary_drop",
        "positive_window_fraction", "top_k_importance_mass_fraction", "primary_top_k_position_fraction",
        "shapley_cosine_similarity", "top_evidence_position_jaccard", "efficiency_residual",
    ]
    for target in sorted({str(row["target"]) for row in rows}):
        subset = [row for row in rows if row["target"] == target]
        summary = {"n": len(subset)}
        for key in numeric:
            values = np.asarray([float(row[key]) for row in subset], dtype=float)
            summary[key + "_mean"] = float(np.mean(values))
            summary[key + "_std"] = float(np.std(values))
        summary["primary_modality_unchanged_rate"] = float(np.mean([bool(x["primary_modality_unchanged"]) for x in subset]))
        summary["ranked_aopc_beats_random_rate"] = float(np.mean([x["aopc_gain_over_random"] > 0 for x in subset]))
        summary["primary_drop_exceeds_secondary_rate"] = float(np.mean([x["primary_minus_secondary_drop"] > 0 for x in subset]))
        result["targets"][target] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", help="Required for a formal frozen-checkpoint run")
    parser.add_argument("--allow-unfrozen", action="store_true", help="Smoke testing only")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="valid")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", default="artifacts/metrics/faithfulness_valid.json")
    parser.add_argument("--output-csv", default="artifacts/metrics/faithfulness_valid.csv")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--random-trials", type=int, default=50)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    if not args.expected_sha256 and not args.allow_unfrozen:
        parser.error("formal runs require --expected-sha256; use --allow-unfrozen only for smoke tests")
    adapter = GatedCheckpointAdapter(args.checkpoint, args.device, args.expected_sha256)
    dataset = MultimodalDataset(args.data, split=args.split, max_samples=args.max_samples)
    rng = np.random.RandomState(args.seed)
    rows: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        item = dataset[index]; sample = adapter.sample_from_item(item); prediction = adapter.predict(sample)
        class_index = int(np.argmax(prediction["probabilities"]))
        targets = [
            ("classification_predicted_probability", adapter.classification_selector(class_index, "probability")),
            ("regression_value", adapter.regression_selector),
        ]
        for target_name, selector in targets:
            row = evaluate_target(adapter, sample, selector, target_name, rng, args.top_k, args.random_trials, args.noise_scale)
            row.update({"sample_id": str(item["id"]), "sample_index": index,
                        "predicted_class_index": class_index if target_name.startswith("classification") else ""})
            rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(dataset):
            print(f"evaluated {index + 1}/{len(dataset)} validation samples")
    output_csv = Path(args.output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_rows = [{k: v for k, v in row.items() if k != "ranked_deletion_curve"} for row in rows]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    payload = {
        "definition": {
            "classification_target": "full-input predicted-class probability fixed across all perturbations",
            "regression_target": "model regression value",
            "aopc": "mean_k(score(full)-score(cumulative top-k evidence deletion))",
            "random_control": "same candidate windows and deletion count in seeded random order",
            "sparsity": "positive-window fraction, top-k positive-importance mass, selected-position fraction",
            "stability": f"Gaussian perturbation at {args.noise_scale:.4g} times observed-feature standard deviation",
        },
        "split": args.split, "seed": args.seed, "model_reference": adapter.model_reference,
        "summary": aggregate(rows), "records": rows,
    }
    output_json = Path(args.output_json); output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adapter.assert_checkpoint_unchanged()
    print(json.dumps(payload["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
