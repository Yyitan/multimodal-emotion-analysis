import pytest
import torch

from src.models.multitask import ConcatBaseline, GatedMultimodalMultiTask, build_model


def batch(batch_size=3, time=8):
    features = {
        "text": torch.randn(batch_size, time, 768),
        "audio": torch.randn(batch_size, time, 74),
        "vision": torch.randn(batch_size, time, 35),
    }
    valid = torch.zeros(batch_size, time, dtype=torch.bool)
    valid[:, :6] = True
    observed = {k: valid.clone() for k in features}
    observed["audio"][0, 2:4] = False
    observed["vision"][1] = False
    return features, observed, valid


def test_single_and_concat_shapes_and_range():
    features, observed, valid = batch()
    for modalities in (("text",), ("text", "audio", "vision")):
        model = ConcatBaseline(modalities=modalities, hidden_dim=16)
        output = model(features, observed, valid)
        assert output["logits"].shape == (3, 3)
        assert output["regression"].shape == (3,)
        assert torch.all(output["regression"].abs() <= 3)


@pytest.mark.parametrize(
    "modalities",
    [
        ["text"],
        ["audio"],
        ["vision"],
        ["text", "audio"],
        ["text", "vision"],
        ["audio", "vision"],
        ["text", "audio", "vision"],
    ],
)
def test_gated_supports_single_dual_and_trimodal_combinations(modalities):
    features, observed, valid = batch()
    model = build_model(
        {"model": {"type": "gated", "modalities": modalities, "hidden_dim": 16}}
    )
    output = model(features, observed, valid)
    assert output["logits"].shape == (3, 3)
    assert output["gates"].shape == (3, len(modalities))
    assert torch.allclose(output["gates"].sum(-1), torch.ones(3), atol=1e-6)


def test_gated_model_masks_unavailable_modality():
    features, observed, valid = batch()
    model = GatedMultimodalMultiTask(hidden_dim=16).eval()
    output = model(features, observed, valid)
    assert output["gates"].shape == (3, 3)
    assert torch.allclose(output["gates"].sum(-1), torch.ones(3), atol=1e-6)
    assert output["gates"][1, 2].item() < 1e-6
    assert torch.isfinite(output["logits"]).all()
