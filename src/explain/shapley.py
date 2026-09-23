"""Exact, model-agnostic Shapley attribution for three modalities."""
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np

MODALITIES: Tuple[str, str, str] = ("text", "audio", "vision")
ScoreSelector = Callable[[Any], float]
PredictCallable = Callable[[Mapping[str, Any]], Any]


def select_score(prediction: Any, score_selector: Optional[ScoreSelector] = None) -> float:
    """Select one fixed scalar target; vector/mapping outputs require a selector."""
    if score_selector is not None:
        prediction = score_selector(prediction)
    elif isinstance(prediction, Mapping):
        raise ValueError("mapping prediction requires score_selector")
    arr = np.asarray(prediction)
    if arr.size != 1:
        raise ValueError("predict output is not scalar; provide score_selector")
    value = float(arr.reshape(-1)[0])
    if not np.isfinite(value):
        raise ValueError("selected prediction score must be finite")
    return value


def _zero_baseline(value: Any) -> Any:
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError("default baseline requires numeric modality values")
    return np.zeros_like(arr)


def _resolve_baseline(name: str, value: Any, baselines: Optional[Any]) -> Any:
    if baselines is None:
        return _zero_baseline(value)
    if callable(baselines):
        return baselines(name, value)
    if name not in baselines:
        raise KeyError("missing baseline for modality %s" % name)
    return baselines[name]


def normalize_contributions(contributions: Mapping[str, float],
                            available: Optional[Mapping[str, bool]] = None,
                            tie_tolerance: float = 1e-12) -> Dict[str, Any]:
    """Return absolute contribution shares and an auditable primary modality."""
    available = available or {m: True for m in MODALITIES}
    magnitudes = {m: abs(float(contributions[m])) if available.get(m, True) else 0.0
                  for m in MODALITIES}
    total = sum(magnitudes.values())
    if total <= tie_tolerance:
        return {"degrees": {m: 0.0 for m in MODALITIES}, "primary_modality": "none",
                "primary_tied_modalities": [], "primary_confidence": "insufficient_signal"}
    degrees = {m: magnitudes[m] / total for m in MODALITIES}
    maximum = max(degrees.values())
    tied = [m for m in MODALITIES if available.get(m, True)
            and abs(degrees[m] - maximum) <= tie_tolerance]
    return {"degrees": degrees, "primary_modality": tied[0],
            "primary_tied_modalities": tied,
            "primary_confidence": "unique" if len(tied) == 1 else "tie"}


@dataclass(frozen=True)
class ShapleyResult:
    baseline_score: float
    full_score: float
    coalition_scores: Mapping[str, float]
    signed_contributions: Mapping[str, float]
    degrees: Mapping[str, float]
    primary_modality: str
    primary_tied_modalities: Sequence[str]
    primary_confidence: str
    efficiency_residual: float
    model_calls: int

    def to_dict(self) -> Dict[str, Any]:
        return {"baseline_score": self.baseline_score, "full_score": self.full_score,
                "coalition_scores": dict(self.coalition_scores),
                "signed_contributions": dict(self.signed_contributions),
                "degrees": dict(self.degrees), "primary_modality": self.primary_modality,
                "primary_tied_modalities": list(self.primary_tied_modalities),
                "primary_confidence": self.primary_confidence,
                "efficiency_residual": self.efficiency_residual, "model_calls": self.model_calls}


def _coalition_name(coalition: Sequence[str]) -> str:
    return "+".join(m for m in MODALITIES if m in coalition) if coalition else "empty"


def exact_modality_shapley(predict: PredictCallable, sample: Mapping[str, Any],
                           baselines: Optional[Any] = None,
                           score_selector: Optional[ScoreSelector] = None,
                           available: Optional[Mapping[str, bool]] = None) -> ShapleyResult:
    """Compute Shapley values with exactly all 2^3=8 coalition calls.

    ``sample`` may include metadata. Absent coalition modalities are replaced by
    explicit baselines. An unavailable modality is baseline in every coalition,
    so its attribution is zero and it cannot become the primary modality.
    """
    missing = [m for m in MODALITIES if m not in sample]
    if missing:
        raise KeyError("sample lacks modalities: %s" % ", ".join(missing))
    availability = {m: True for m in MODALITIES}
    if available is not None:
        availability.update({m: bool(available.get(m, True)) for m in MODALITIES})
    baseline_values = {m: _resolve_baseline(m, sample[m], baselines) for m in MODALITIES}
    values: Dict[frozenset, float] = {}
    named: Dict[str, float] = {}
    for size in range(4):
        for chosen in combinations(MODALITIES, size):
            coalition = frozenset(chosen)
            model_input = dict(sample)
            for modality in MODALITIES:
                model_input[modality] = (sample[modality] if modality in coalition
                                         and availability[modality] else baseline_values[modality])
            score = select_score(predict(model_input), score_selector)
            values[coalition] = score
            named[_coalition_name(chosen)] = score
    phi: Dict[str, float] = {}
    for modality in MODALITIES:
        others = [m for m in MODALITIES if m != modality]
        contribution = 0.0
        for size in range(3):
            for chosen in combinations(others, size):
                coalition = frozenset(chosen)
                weight = factorial(size) * factorial(2 - size) / factorial(3)
                contribution += weight * (values[coalition | {modality}] - values[coalition])
        phi[modality] = 0.0 if not availability[modality] and abs(contribution) < 1e-12 else float(contribution)
    normalized = normalize_contributions(phi, availability)
    residual = (values[frozenset(MODALITIES)] - values[frozenset()]) - sum(phi.values())
    return ShapleyResult(values[frozenset()], values[frozenset(MODALITIES)], named, phi,
                         normalized["degrees"], normalized["primary_modality"],
                         normalized["primary_tied_modalities"], normalized["primary_confidence"],
                         float(residual), len(values))
