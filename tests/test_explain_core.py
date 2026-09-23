import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.explain import (
    aopc_random_deletion,
    exact_modality_shapley,
    explain_sample,
    inspect_modalities,
    local_window_occlusion,
    map_position_window,
    merge_adjacent_evidence,
)


def additive_toy(sample):
    return (2.0 * float(np.sum(sample["text"]))
            - 1.0 * float(np.sum(sample["audio"]))
            + 0.5 * float(np.sum(sample["vision"])))


def unit_sample():
    return {"text": np.ones(1), "audio": np.ones(1), "vision": np.ones(1), "id": "toy"}


def test_exact_shapley_recovers_additive_coefficients_and_efficiency():
    result = exact_modality_shapley(additive_toy, unit_sample())
    assert result.model_calls == 8
    assert set(result.coalition_scores) == {
        "empty", "text", "audio", "vision", "text+audio", "text+vision",
        "audio+vision", "text+audio+vision",
    }
    assert result.signed_contributions == pytest.approx(
        {"text": 2.0, "audio": -1.0, "vision": 0.5}, abs=1e-12
    )
    assert sum(result.signed_contributions.values()) == pytest.approx(
        result.full_score - result.baseline_score, abs=1e-12
    )
    assert result.efficiency_residual == pytest.approx(0.0, abs=1e-12)
    assert result.degrees == pytest.approx(
        {"text": 2 / 3.5, "audio": 1 / 3.5, "vision": 0.5 / 3.5}, abs=1e-12
    )
    assert result.primary_modality == "text"


def test_vector_prediction_requires_and_uses_fixed_selector():
    def predict(sample):
        score = additive_toy(sample)
        return {"probabilities": [1.0 - score / 10.0, score / 10.0]}

    with pytest.raises(ValueError):
        exact_modality_shapley(predict, unit_sample())
    selected = exact_modality_shapley(
        predict, unit_sample(), score_selector=lambda output: output["probabilities"][1]
    )
    assert selected.signed_contributions == pytest.approx(
        {"text": 0.2, "audio": -0.1, "vision": 0.05}, abs=1e-12
    )


def test_unavailable_all_zero_modality_has_zero_contribution():
    sample = unit_sample(); sample["vision"] = np.zeros((50, 35))
    quality = inspect_modalities(sample)
    assert quality["availability"]["vision"] is False
    assert "VISION_ALL_ZERO" in quality["quality_flags"]
    result = exact_modality_shapley(
        additive_toy, sample, available=quality["availability"]
    )
    assert result.signed_contributions["vision"] == 0.0
    assert result.degrees["vision"] == 0.0
    assert result.primary_modality != "vision"


def sequence_toy(sample):
    return float(np.sum(sample["text"]))


def test_local_occlusion_preserves_support_and_counter_evidence_direction():
    sample = {"text": np.asarray([[3.0], [-2.0], [1.0]]),
              "audio": np.zeros((1, 1)), "vision": np.zeros((1, 1))}
    scores = local_window_occlusion(sequence_toy, sample, "text", window_size=1)
    assert [x["score_drop"] for x in scores] == pytest.approx([3.0, -2.0, 1.0])
    assert [x["importance"] for x in scores] == pytest.approx([3.0, 0.0, 1.0])


def test_adjacent_evidence_merges_and_aopc_beats_seeded_random_control():
    sample = {"text": np.asarray([[4.0], [3.0], [2.0], [1.0]]),
              "audio": np.zeros((1, 1)), "vision": np.zeros((1, 1))}
    scores = local_window_occlusion(sequence_toy, sample, "text", window_size=1)
    merged = merge_adjacent_evidence(scores, top_k=2)
    assert len(merged) == 1
    assert (merged[0].start_index, merged[0].end_index) == (0, 2)
    check = aopc_random_deletion(sequence_toy, sample, "text", scores,
                                 max_steps=2, random_trials=200, seed=7)
    assert check["ranked"]["aopc"] == pytest.approx((4.0 + 7.0) / 2.0)
    assert check["aopc_gain_over_random"] > 0.0


def test_time_mapping_forced_alignment_is_exact_and_uniform_is_approximate():
    alignment = [
        {"position": 0, "start_sec": 0.0, "end_sec": 0.3, "confidence": "exact"},
        {"position": 1, "start_sec": 0.3, "end_sec": 0.8, "confidence": "exact"},
        {"position": 2, "start_sec": 0.8, "end_sec": 1.0, "confidence": "exact"},
    ]
    exact = map_position_window(0, 2, total_positions=3, duration_sec=3.0,
                                forced_alignment=alignment, include_keyframe=True)
    assert exact.method == "forced_alignment"
    assert exact.confidence == "exact"
    assert exact.approximate is False
    assert (exact.start_sec, exact.end_sec, exact.keyframe_sec) == pytest.approx((0.0, 0.8, 0.4))

    approximate = map_position_window(10, 20, total_positions=50, duration_sec=10.0)
    assert approximate.method == "uniform_50_position"
    assert approximate.confidence == "approximate"
    assert approximate.approximate is True
    assert (approximate.start_sec, approximate.end_sec) == pytest.approx((2.0, 4.0))

    unavailable = map_position_window(10, 20, total_positions=50)
    assert unavailable.method == "unavailable"
    assert unavailable.start_sec is None and unavailable.end_sec is None


def test_incomplete_forced_alignment_falls_back_without_inventing_exactness():
    incomplete = [{"position": 0, "start_sec": 0.0, "end_sec": 0.2}]
    result = map_position_window(0, 2, total_positions=50, duration_sec=5.0,
                                 forced_alignment=incomplete)
    assert result.method == "uniform_50_position"
    assert result.approximate is True


def test_end_to_end_pipeline_accepts_predict_callable_and_marks_uniform_times():
    sample = {
        "text": np.asarray([[3.0], [1.0], [0.5]]),
        "audio": np.asarray([[-1.0], [-0.5], [-0.25]]),
        "vision": np.asarray([[0.5], [0.25], [0.125]]),
    }
    result = explain_sample(
        additive_toy, sample,
        window_sizes={"text": 1, "audio": 1, "vision": 1},
        top_k=1, duration_sec=3.0,
        text_resolver=lambda start, end: "token-%d-%d" % (start, end),
        aopc_steps=1, random_trials=5, seed=3,
    )
    assert result["shapley"]["model_calls"] == 8
    assert result["shapley"]["primary_modality"] == "text"
    assert result["evidence"]["text"][0]["text"].startswith("token-")
    assert result["evidence"]["audio"][0]["time"]["approximate"] is True
    assert result["evidence"]["vision"][0]["time"]["method"] == "uniform_position"
    assert result["faithfulness"]["steps"] == 1


def test_detail_json_schema_itself_is_valid():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "configs/explanation_set_detail.schema.json").read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("draft-07/schema#")
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"schema_version", "sample_id", "prediction", "explanation", "faithfulness", "quality", "provenance"}
