import pickle

import numpy as np
import pytest

from src.data.dataset import MultimodalDataset, load_records, validate_text_bert


def synthetic_mapping(n=2, time=6):
    tokens = np.zeros((n, 3, time), dtype=np.int64)
    tokens[:, 0, :4] = np.array([101, 2001, 2002, 102])
    tokens[:, 1, :4] = 1
    return {
        "text_bert": tokens,
        "text": np.random.default_rng(1).normal(size=(n, time, 768)).astype(np.float32),
        "audio": np.ones((n, time, 74), dtype=np.float32),
        "vision": np.ones((n, time, 35), dtype=np.float32),
        "id": ["a", "b"],
        "classification_labels": np.array([0, 2]),
        "regression_labels": np.array([-1.0, 2.0]),
    }


def test_float_integral_text_bert_is_cast_but_fraction_is_rejected():
    value = synthetic_mapping()["text_bert"].astype(np.float32)
    assert validate_text_bert(value).dtype == np.int64
    value[0, 0, 1] = 1.5
    with pytest.raises(ValueError, match="non-integral"):
        validate_text_bert(value)


def test_masks_distinguish_padding_and_internal_missing(tmp_path):
    data = synthetic_mapping()
    data["audio"][0, 2] = 0  # internal missing inside attention-valid region
    data["vision"][0, 4] = 9  # nonzero padding must still be excluded
    path = tmp_path / "sample.pkl"
    with path.open("wb") as handle:
        pickle.dump({"train": data}, handle)
    records = load_records(path, split="train")
    assert records["valid"][0].tolist() == [True, True, True, True, False, False]
    assert not records["observed"]["audio"][0, 2]
    assert not records["observed"]["vision"][0, 4]
    assert records["classification"].tolist() == [0, 2]
    assert len(MultimodalDataset(path, split="train")) == 2


def test_explanation_set_direct_mapping_gets_unknown_labels(tmp_path):
    data = synthetic_mapping(n=2)
    data.pop("classification_labels")
    data.pop("regression_labels")
    path = tmp_path / "04.pkl"
    with path.open("wb") as handle:
        pickle.dump(data, handle)
    records = load_records(path)
    assert records["classification"].tolist() == [-100, -100]
    assert np.isnan(records["regression"]).all()


def test_missing_text_requires_explicit_provider(tmp_path):
    data = synthetic_mapping()
    data.pop("text")
    path = tmp_path / "03.pkl"
    with path.open("wb") as handle:
        pickle.dump({"test": data}, handle)
    with pytest.raises(ValueError, match="BertTextProvider"):
        load_records(path)
