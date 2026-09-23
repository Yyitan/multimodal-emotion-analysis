"""Integration tests for the real explanation checkpoint boundary."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.explain.model_adapter import GatedCheckpointAdapter, sha256_file
from src.explain.shapley import exact_modality_shapley
from src.models import build_model


def make_checkpoint(path: Path) -> Path:
    config = {
        "seed": 7,
        "model": {"type": "gated", "modalities": ["text", "audio", "vision"], "hidden_dim": 8, "dropout": 0.0},
        "training": {},
    }
    torch.manual_seed(7)
    model = build_model(config)
    torch.save({"model_state": model.state_dict(), "config": config, "epoch": 3,
                "metrics": {"macro_f1": 0.5}}, path)
    return path


def sample(vision_zero: bool = False):
    rng = np.random.RandomState(3)
    length = 5
    values = {
        "text": rng.normal(size=(length, 768)).astype(np.float32),
        "audio": rng.normal(size=(length, 74)).astype(np.float32),
        "vision": rng.normal(size=(length, 35)).astype(np.float32),
    }
    if vision_zero:
        values["vision"].fill(0.0)
    values["_valid"] = np.asarray([True, True, True, False, False])
    values["_observed"] = {m: values["_valid"].copy() for m in ("text", "audio", "vision")}
    values["text"][3:] = 0.0; values["audio"][3:] = 0.0; values["vision"][3:] = 0.0
    return values


def test_adapter_zeroes_coalition_and_marks_it_unobserved(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    adapter = GatedCheckpointAdapter(checkpoint, expected_sha256=sha256_file(checkpoint))
    full = sample()
    removed = dict(full); removed["audio"] = np.zeros_like(full["audio"])
    prediction = adapter.predict(removed)
    assert prediction["observed_counts"] == {"text": 3, "audio": 0, "vision": 3}
    assert np.isfinite(prediction["probabilities"]).all()
    assert prediction["probabilities"].sum() == pytest.approx(1.0)


def test_adapter_intersects_original_observed_mask(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    adapter = GatedCheckpointAdapter(checkpoint)
    value = sample(); value["_observed"]["text"][1] = False
    # A nonzero vector at an explicitly unobserved position must remain unobserved.
    assert adapter.predict(value)["observed_counts"]["text"] == 2


def test_classification_and_regression_shapley_are_named_separate_targets(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    adapter = GatedCheckpointAdapter(checkpoint)
    value = sample(vision_zero=True); available = {"text": True, "audio": True, "vision": False}
    full = adapter.predict(value); fixed_class = int(np.argmax(full["probabilities"]))
    classification = exact_modality_shapley(
        adapter.predict, value, score_selector=adapter.classification_selector(fixed_class), available=available
    )
    regression = exact_modality_shapley(
        adapter.predict, value, score_selector=adapter.regression_selector, available=available
    )
    assert classification.model_calls == regression.model_calls == 8
    assert classification.signed_contributions["vision"] == pytest.approx(0.0, abs=1e-12)
    assert regression.signed_contributions["vision"] == pytest.approx(0.0, abs=1e-12)
    assert classification.full_score == pytest.approx(full["probabilities"][fixed_class])
    assert regression.full_score == pytest.approx(full["regression"])
    assert abs(classification.efficiency_residual) < 1e-6
    assert abs(regression.efficiency_residual) < 1e-6


def test_adapter_detects_checkpoint_mutation(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "model.pt")
    adapter = GatedCheckpointAdapter(checkpoint)
    with checkpoint.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        adapter.assert_checkpoint_unchanged()
