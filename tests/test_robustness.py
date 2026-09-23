import numpy as np
import pytest
import torch

from src.evaluation.robustness import (
    MissingCondition,
    RobustnessDataset,
    apply_span_missingness,
    build_span_mask,
    compute_metrics_with_group_ci,
    conditions_from_grid,
    iter_group_bootstrap_indices,
    mask_intervals,
    normalize_modalities,
    video_group,
)


def test_single_span_is_contiguous_and_has_requested_rounded_proportion():
    valid = np.array([1] * 17 + [0] * 5, dtype=bool)
    mask = build_span_mask(valid, 0.3, position="random", n_spans=1, seed=11)
    assert mask.sum() == 5  # floor(17 * .3 + .5)
    assert len(mask_intervals(mask)) == 1
    assert not mask[~valid].any()


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("head", np.r_[np.ones(4), np.zeros(6)]),
        ("middle", np.r_[np.zeros(3), np.ones(4), np.zeros(3)]),
        ("tail", np.r_[np.zeros(6), np.ones(4)]),
    ],
)
def test_fixed_positions(position, expected):
    mask = build_span_mask(np.ones(10, dtype=bool), 0.4, position=position, n_spans=1)
    assert np.array_equal(mask, expected.astype(bool))


def test_multi_span_count_continuity_and_exact_total():
    mask = build_span_mask(np.ones(30, dtype=bool), 0.4, "random", n_spans=3, seed=7)
    intervals = mask_intervals(mask)
    assert mask.sum() == 12
    assert len(intervals) == 3
    assert all(end > start for start, end in intervals)


def test_padding_values_and_masks_are_unchanged_for_all_combinations():
    valid = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    features = {
        "text": torch.arange(18, dtype=torch.float32).reshape(9, 2) + 1,
        "audio": torch.arange(27, dtype=torch.float32).reshape(9, 3) + 101,
        "vision": torch.arange(36, dtype=torch.float32).reshape(9, 4) + 201,
    }
    observed = {name: valid.clone() for name in features}
    for code in ("T", "A", "V", "TA", "TV", "AV", "TAV"):
        augmented, masks, missing = apply_span_missingness(
            features, observed, valid, code, 0.5, "middle", 1, seed=9
        )
        assert missing.sum() == 3
        assert not missing[~valid].any()
        changed = set(normalize_modalities(code))
        for name in features:
            assert torch.equal(augmented[name][~valid], features[name][~valid])
            assert torch.equal(masks[name][~valid], observed[name][~valid])
            if name in changed:
                assert augmented[name][missing].eq(0).all()
                assert not masks[name][missing].any()
            else:
                assert torch.equal(augmented[name], features[name])
                assert torch.equal(masks[name], observed[name])
        assert all(torch.equal(features[name], {
            "text": torch.arange(18, dtype=torch.float32).reshape(9, 2) + 1,
            "audio": torch.arange(27, dtype=torch.float32).reshape(9, 3) + 101,
            "vision": torch.arange(36, dtype=torch.float32).reshape(9, 4) + 201,
        }[name]) for name in features)


def test_random_mask_seed_is_reproducible_and_can_change():
    valid = np.ones(40, dtype=bool)
    first = build_span_mask(valid, 0.25, "random", 3, seed=123)
    second = build_span_mask(valid, 0.25, "random", 3, seed=123)
    third = build_span_mask(valid, 0.25, "random", 3, seed=124)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


class _TinyDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        valid = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
        return {
            "features": {name: torch.ones(5, size) for name, size in (("text", 2), ("audio", 3), ("vision", 4))},
            "observed": {name: valid.clone() for name in ("text", "audio", "vision")},
            "valid": valid,
            "classification": torch.tensor(index),
            "regression": torch.tensor(float(index)),
            "id": f"v{index}_clip",
        }


def test_dataset_view_is_index_reproducible_and_preserves_base():
    base = _TinyDataset()
    view = RobustnessDataset(base, MissingCondition(("text", "audio"), 0.5, "random", 2, 77))
    first, second = view[1], view[1]
    assert torch.equal(first["robustness_mask"], second["robustness_mask"])
    assert first["features"]["text"].eq(second["features"]["text"]).all()
    assert base[1]["features"]["text"].eq(1).all()


def test_full_grid_has_one_clean_and_every_required_combination():
    grid = {
        "modalities": ["T", "A", "V", "TA", "TV", "AV", "TAV"],
        "missing_rates": [0, .1, .2, .3, .5, .7],
        "positions": ["head", "middle", "tail", "random"],
        "span_counts": [1, 3],
    }
    conditions = conditions_from_grid(grid)
    assert len(conditions) == 1 + 7 * 5 * 4 * 2
    assert sum(item.missing_rate == 0 for item in conditions) == 1
    assert {(item.position, item.n_spans) for item in conditions if item.missing_rate} == {
        (position, count) for position in grid["positions"] for count in grid["span_counts"]
    }


def test_video_prefix_and_group_bootstrap_keep_whole_groups():
    ids = ["video1_0", "video1_1", "video2_0", "video2_1", "video2_2", "video3_0"]
    groups = [video_group(value) for value in ids]
    assert groups == ["video1", "video1", "video2", "video2", "video2", "video3"]
    draws = list(iter_group_bootstrap_indices(groups, n_resamples=10, seed=6))
    draws_again = list(iter_group_bootstrap_indices(groups, n_resamples=10, seed=6))
    assert all(np.array_equal(a, b) for a, b in zip(draws, draws_again))
    source_sizes = {"video1": 2, "video2": 3, "video3": 1}
    for indices in draws:
        counts = {group: groups_array.tolist().count(group) for group in source_sizes for groups_array in [np.asarray(groups)[indices]]}
        assert all(counts[group] % source_sizes[group] == 0 for group in source_sizes)


def test_grouped_metric_ci_reports_counts_and_is_reproducible():
    prediction = {
        "ids": ["a_0", "a_1", "b_0", "b_1", "c_0", "c_1"],
        "true_class": [0, 0, 1, 1, 2, 2],
        "predicted_class": [0, 1, 1, 1, 2, 0],
        "true_regression": [-2, -1, 0, 1, 2, 3],
        "predicted_regression": [-1.5, -1, .2, .7, 1.8, 2.5],
    }
    first = compute_metrics_with_group_ci(prediction, n_resamples=60, seed=19)
    second = compute_metrics_with_group_ci(prediction, n_resamples=60, seed=19)
    assert first == second
    assert first["n_samples"] == 6
    assert first["n_classification"] == 6
    assert first["n_regression"] == 6
    assert first["n_video_groups"] == 3
    assert first["bootstrap_unit"] == "video_id_prefix"
    for metric in ("accuracy", "macro_f1", "weighted_f1", "mae", "pearson"):
        assert np.isfinite(first[metric])
        assert first[f"{metric}_ci_lower"] <= first[f"{metric}_ci_upper"]
