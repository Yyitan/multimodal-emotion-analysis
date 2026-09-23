"""Local window occlusion, evidence merging, and deletion faithfulness."""
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from .shapley import PredictCallable, ScoreSelector, select_score


@dataclass(frozen=True)
class EvidenceWindow:
    start_index: int
    end_index: int
    score_drop: float
    importance: float
    peak_importance: float
    member_windows: Tuple[Tuple[int, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"start_index": self.start_index, "end_index": self.end_index,
                "score_drop": self.score_drop, "importance": self.importance,
                "peak_importance": self.peak_importance,
                "member_windows": [list(x) for x in self.member_windows]}


def make_windows(length: int, window_size: int, stride: int = 1) -> List[Tuple[int, int]]:
    if length <= 0 or window_size <= 0 or stride <= 0:
        raise ValueError("length, window_size, and stride must be positive")
    window_size = min(window_size, length)
    starts = list(range(0, length - window_size + 1, stride))
    if starts[-1] != length - window_size:
        starts.append(length - window_size)
    return [(s, s + window_size) for s in starts]


def _validate_windows(windows: Iterable[Tuple[int, int]], length: int) -> List[Tuple[int, int]]:
    result = []
    for start, end in windows:
        start, end = int(start), int(end)
        if not (0 <= start < end <= length):
            raise ValueError("invalid half-open window [%d, %d)" % (start, end))
        result.append((start, end))
    if not result:
        raise ValueError("at least one window is required")
    return result


def _baseline_array(value: np.ndarray, baseline: Optional[Any]) -> np.ndarray:
    if baseline is None:
        return np.zeros_like(value)
    base = np.asarray(baseline)
    if base.ndim == 0:
        return np.full_like(value, base.item())
    if base.shape != value.shape:
        raise ValueError("baseline must be scalar or match modality shape")
    return base.astype(value.dtype, copy=False)


def _apply_windows(value: np.ndarray, baseline: np.ndarray,
                   windows: Sequence[Tuple[int, int]]) -> np.ndarray:
    masked = value.copy()
    for start, end in windows:
        masked[start:end] = baseline[start:end]
    return masked


def local_window_occlusion(predict: PredictCallable, sample: Mapping[str, Any], modality: str,
                           window_size: int = 1, stride: int = 1,
                           valid_length: Optional[int] = None,
                           windows: Optional[Sequence[Tuple[int, int]]] = None,
                           baseline: Optional[Any] = None,
                           score_selector: Optional[ScoreSelector] = None) -> List[Dict[str, Any]]:
    """Mask one local window at a time; positive score drop is supporting evidence."""
    if modality not in sample:
        raise KeyError("sample lacks modality %s" % modality)
    value = np.asarray(sample[modality])
    if value.ndim < 1:
        raise ValueError("modality must have a sequence axis")
    length = value.shape[0] if valid_length is None else int(valid_length)
    if not (1 <= length <= value.shape[0]):
        raise ValueError("valid_length is outside modality sequence")
    chosen = make_windows(length, window_size, stride) if windows is None else _validate_windows(windows, length)
    baseline_value = _baseline_array(value, baseline)
    reference = select_score(predict(dict(sample)), score_selector)
    result = []
    for start, end in chosen:
        perturbed = dict(sample)
        perturbed[modality] = _apply_windows(value, baseline_value, [(start, end)])
        masked_score = select_score(predict(perturbed), score_selector)
        drop = reference - masked_score
        result.append({"start_index": start, "end_index": end,
                       "reference_score": reference, "occluded_score": masked_score,
                       "score_drop": float(drop), "importance": float(max(drop, 0.0))})
    return result


def merge_adjacent_evidence(scored_windows: Sequence[Mapping[str, Any]], top_k: Optional[int] = None,
                            min_importance: float = 0.0, max_gap: int = 0) -> List[EvidenceWindow]:
    """Select by importance, then merge overlapping/adjacent position windows."""
    if max_gap < 0:
        raise ValueError("max_gap must be non-negative")
    candidates = [dict(x) for x in scored_windows
                  if float(x.get("importance", max(float(x.get("score_drop", 0.0)), 0.0))) > min_importance]
    candidates.sort(key=lambda x: (-float(x.get("importance", x.get("score_drop", 0.0))),
                                   int(x["start_index"]), int(x["end_index"])))
    if top_k is not None:
        if top_k <= 0:
            return []
        candidates = candidates[:top_k]
    candidates.sort(key=lambda x: (int(x["start_index"]), int(x["end_index"])))
    groups: List[List[Mapping[str, Any]]] = []
    for item in candidates:
        start = int(item["start_index"])
        if not groups or start > max(int(x["end_index"]) for x in groups[-1]) + max_gap:
            groups.append([item])
        else:
            groups[-1].append(item)
    merged = []
    for group in groups:
        starts = [int(x["start_index"]) for x in group]
        ends = [int(x["end_index"]) for x in group]
        drops = [float(x.get("score_drop", 0.0)) for x in group]
        imps = [float(x.get("importance", max(float(x.get("score_drop", 0.0)), 0.0))) for x in group]
        merged.append(EvidenceWindow(min(starts), max(ends), float(sum(drops)), float(sum(imps)),
                                     float(max(imps)), tuple((int(x["start_index"]), int(x["end_index"]))
                                                              for x in group)))
    return sorted(merged, key=lambda x: (-x.importance, x.start_index))


def _deletion_curve(predict: PredictCallable, sample: Mapping[str, Any], modality: str,
                    ordered_windows: Sequence[Tuple[int, int]], baseline: Optional[Any],
                    score_selector: Optional[ScoreSelector]) -> Dict[str, Any]:
    value = np.asarray(sample[modality]); baseline_value = _baseline_array(value, baseline)
    reference = select_score(predict(dict(sample)), score_selector)
    scores, drops, removed, cumulative = [], [], [], []
    for window in ordered_windows:
        cumulative.append(window)
        perturbed = dict(sample)
        perturbed[modality] = _apply_windows(value, baseline_value, cumulative)
        score = select_score(predict(perturbed), score_selector)
        scores.append(float(score)); drops.append(float(reference - score)); removed.append(list(window))
    return {"reference_score": reference, "scores": scores, "drops": drops,
            "removed_windows": removed, "aopc": float(np.mean(drops)) if drops else 0.0}


def aopc_random_deletion(predict: PredictCallable, sample: Mapping[str, Any], modality: str,
                         scored_windows: Sequence[Mapping[str, Any]], baseline: Optional[Any] = None,
                         score_selector: Optional[ScoreSelector] = None,
                         max_steps: Optional[int] = None, random_trials: int = 100,
                         seed: int = 0) -> Dict[str, Any]:
    """AOPC of ranked cumulative deletion versus seeded random-order controls."""
    if random_trials <= 0:
        raise ValueError("random_trials must be positive")
    ranked = sorted(scored_windows,
                    key=lambda x: (-float(x.get("importance", x.get("score_drop", 0.0))),
                                   int(x["start_index"])))
    windows = [(int(x["start_index"]), int(x["end_index"])) for x in ranked]
    if not windows:
        raise ValueError("scored_windows cannot be empty")
    k = len(windows) if max_steps is None else min(int(max_steps), len(windows))
    if k <= 0:
        raise ValueError("max_steps must be positive")
    ranked_curve = _deletion_curve(predict, sample, modality, windows[:k], baseline, score_selector)
    rng = np.random.RandomState(seed); random_aopc, random_drops = [], []
    for _ in range(random_trials):
        order = list(windows); rng.shuffle(order)
        curve = _deletion_curve(predict, sample, modality, order[:k], baseline, score_selector)
        random_aopc.append(curve["aopc"]); random_drops.append(curve["drops"])
    random_array = np.asarray(random_drops, dtype=float); random_mean = float(np.mean(random_aopc))
    return {"definition": "mean_k(f(x)-f(x_with_top_1_to_k_windows_deleted))", "steps": k,
            "ranked": ranked_curve,
            "random": {"trials": random_trials, "seed": seed, "aopc_mean": random_mean,
                       "aopc_std": float(np.std(random_aopc)),
                       "mean_drop_by_step": np.mean(random_array, axis=0).tolist()},
            "aopc_gain_over_random": float(ranked_curve["aopc"] - random_mean)}
