import numpy as np
import pytest
import torch

from src.evaluation.metrics import compute_metrics
from src.training.augment import continuous_span_mask
from src.training.losses import consistency_loss, supervised_multitask_loss
from src.training.trainer import compute_selection_score


def test_continuous_span_augmentation_masks_all_modalities_and_preserves_padding():
    valid = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    features = {
        "text": torch.ones(1, 6, 2),
        "audio": torch.ones(1, 6, 2),
        "vision": torch.ones(1, 6, 2),
    }
    observed = {k: valid.clone() for k in features}
    augmented, masks = continuous_span_mask(
        features, observed, valid, probability=1.0, min_fraction=0.5, max_fraction=0.5,
        generator=torch.Generator().manual_seed(3),
    )
    for modality in features:
        removed = (~masks[modality][0, :4]).nonzero().flatten()
        assert removed.numel() == 2
        assert torch.equal(removed, torch.arange(removed[0], removed[0] + 2))
        assert augmented[modality][0, removed].eq(0).all()
        assert not masks[modality][0, 4:].any()
        assert augmented[modality][0, 4:].eq(1).all()  # padding values untouched


def test_losses_and_required_metrics():
    clean = {"logits": torch.randn(4, 3, requires_grad=True), "regression": torch.tanh(torch.randn(4, requires_grad=True)) * 3}
    augmented = {"logits": torch.randn(4, 3, requires_grad=True), "regression": torch.tanh(torch.randn(4, requires_grad=True)) * 3}
    losses = supervised_multitask_loss(clean, torch.tensor([0, 1, 2, -100]), torch.tensor([-2.0, 0.0, 2.0, float("nan")]))
    total = losses["loss"] + consistency_loss(clean, augmented)
    total.backward()
    assert torch.isfinite(total)
    metrics = compute_metrics([0, 1, 2], [0, 1, 1], [-1, 0, 1], [-1, 0.2, 0.8])
    assert set(metrics) == {"accuracy", "macro_f1", "weighted_f1", "mae", "pearson"}
    assert np.isfinite(list(metrics.values())).all()


def test_selection_score_is_normalized_weighted_joint_metric():
    weights = {"macro_f1": 0.35, "accuracy": 0.15, "pearson": 0.25, "mae": 0.25}
    metrics = {"macro_f1": 0.8, "accuracy": 0.6, "pearson": 0.2, "mae": 1.2}
    expected = 0.35 * 0.8 + 0.15 * 0.6 + 0.25 * 0.6 + 0.25 * 0.8
    assert compute_selection_score(metrics, weights) == pytest.approx(expected)
    assert compute_selection_score({"macro_f1": 2, "accuracy": -1, "pearson": 4, "mae": 99}, weights) == pytest.approx(0.60)
    assert compute_selection_score({key: float("nan") for key in weights}, weights) == 0.0


def test_selection_weights_must_be_fixed_convex_combination():
    with pytest.raises(ValueError, match="sum to 1.0"):
        compute_selection_score(
            {"macro_f1": 1, "accuracy": 1, "pearson": 1, "mae": 0},
            {"macro_f1": 1, "accuracy": 1, "pearson": 1, "mae": 1},
        )
