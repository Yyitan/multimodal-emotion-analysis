"""End-to-end composition of the model-agnostic multimodal explainers."""
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .occlusion import aopc_random_deletion, local_window_occlusion, merge_adjacent_evidence
from .quality import inspect_modalities
from .shapley import MODALITIES, PredictCallable, ScoreSelector, exact_modality_shapley
from .time_mapping import map_position_window

TextResolver = Callable[[int, int], str]


def explain_sample(
    predict: PredictCallable,
    sample: Mapping[str, Any],
    score_selector: Optional[ScoreSelector] = None,
    baselines: Optional[Any] = None,
    window_sizes: Optional[Mapping[str, int]] = None,
    strides: Optional[Mapping[str, int]] = None,
    valid_lengths: Optional[Mapping[str, int]] = None,
    top_k: int = 5,
    max_merge_gap: int = 0,
    duration_sec: Optional[float] = None,
    forced_alignments: Optional[Mapping[str, Any]] = None,
    text_resolver: Optional[TextResolver] = None,
    aopc_steps: int = 5,
    random_trials: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """Explain one sample through any callable model.

    ``score_selector`` must select one fixed output (for example the complete
    input's predicted-class probability) for all perturbations. ``text_resolver``
    is intentionally required to emit textual quotations: without an auditable
    position-to-source-text mapping no text string is invented, and the returned
    text evidence contains positions only plus ``text=None``.
    """
    window_sizes = dict(window_sizes or {"text": 1, "audio": 3, "vision": 3})
    strides = dict(strides or {m: 1 for m in MODALITIES})
    valid_lengths = dict(valid_lengths or {})
    forced_alignments = dict(forced_alignments or {})
    quality = inspect_modalities(sample)
    shapley = exact_modality_shapley(
        predict, sample, baselines=baselines, score_selector=score_selector,
        available=quality["availability"],
    )
    evidence: Dict[str, Any] = {m: [] for m in MODALITIES}
    raw_scores: Dict[str, Any] = {}
    for modality in MODALITIES:
        if not quality["availability"][modality]:
            continue
        modality_baseline = None
        if isinstance(baselines, Mapping):
            modality_baseline = baselines.get(modality)
        elif callable(baselines):
            modality_baseline = baselines(modality, sample[modality])
        scores = local_window_occlusion(
            predict, sample, modality,
            window_size=int(window_sizes.get(modality, 1)),
            stride=int(strides.get(modality, 1)),
            valid_length=valid_lengths.get(modality),
            baseline=modality_baseline,
            score_selector=score_selector,
        )
        raw_scores[modality] = scores
        merged = merge_adjacent_evidence(scores, top_k=top_k, max_gap=max_merge_gap)
        for window in merged:
            item = window.to_dict()
            if modality == "text":
                text = text_resolver(window.start_index, window.end_index) if text_resolver else None
                if text is None:
                    continue
                item.update({
                    "text": text,
                    "location_method": "token_alignment",
                    "location_confidence": "exact",
                })
            else:
                total_positions = int(valid_lengths.get(modality, np.asarray(sample[modality]).shape[0]))
                mapping = map_position_window(
                    window.start_index, window.end_index,
                    total_positions=total_positions,
                    duration_sec=duration_sec,
                    forced_alignment=forced_alignments.get(modality),
                    include_keyframe=(modality == "vision"),
                )
                item["time"] = mapping.to_dict()
                if modality == "vision":
                    item["keyframe_sec"] = mapping.keyframe_sec
            evidence[modality].append(item)

    primary = shapley.primary_modality
    faithfulness = None
    if primary in raw_scores and raw_scores[primary]:
        primary_baseline = baselines.get(primary) if isinstance(baselines, Mapping) else None
        faithfulness = aopc_random_deletion(
            predict, sample, primary, raw_scores[primary], baseline=primary_baseline,
            score_selector=score_selector, max_steps=aopc_steps,
            random_trials=random_trials, seed=seed,
        )
    return {
        "shapley": shapley.to_dict(),
        "evidence": evidence,
        "faithfulness": faithfulness,
        "quality": quality,
    }
