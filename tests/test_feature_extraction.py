import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.extraction_pipeline import (
    AUDIO_DIM, AUDIO_FEATURE_NAMES, N_POSITIONS, VISION_DIM,
    VISION_FEATURE_NAMES, WordInterval, _ctc_viterbi,
    approximate_word_intervals, extract_audio_features, pool_frames,
    position_groups, split_words,
)


def test_feature_definitions_are_exact():
    assert len(AUDIO_FEATURE_NAMES) == AUDIO_DIM == 74
    assert AUDIO_FEATURE_NAMES[:20] == [f"mfcc_{i:02d}" for i in range(20)]
    assert AUDIO_FEATURE_NAMES[60:72] == [f"chroma_{i:02d}" for i in range(12)]
    assert AUDIO_FEATURE_NAMES[-2:] == ["rms", "f0_hz"]
    assert len(VISION_FEATURE_NAMES) == VISION_DIM == 35
    assert len(set(VISION_FEATURE_NAMES)) == VISION_DIM


def test_approximate_alignment_is_weighted_and_bounded():
    intervals = approximate_word_intervals(["a", "longer", "!!"], 8.0)
    assert intervals[0].start == 0.0
    assert intervals[-1].end == 8.0
    durations = [x.end - x.start for x in intervals]
    assert durations[1] > durations[0]
    assert durations[2] == durations[0]
    assert all(a.end <= b.start + 1e-10 for a, b in zip(intervals, intervals[1:]))


def test_split_and_position_grouping_caps_at_50():
    words, spans = split_words("one  two\tthree")
    assert words == ["one", "two", "three"]
    assert ["one  two\tthree"[a:b] for a, b in spans] == words
    groups = position_groups(73)
    assert len(groups) == N_POSITIONS
    assert np.array_equal(np.concatenate(groups), np.arange(73))


def test_pool_frames_respects_detection_mask():
    times = np.asarray([0.1, 0.4, 0.6, 0.9], np.float32)
    features = np.asarray([[1, 1], [3, 3], [10, 10], [20, 20]], np.float32)
    intervals = [WordInterval("one", 0.0, 0.5), WordInterval("two", 0.5, 1.0)]
    values, mask = pool_frames(times, features, intervals, position_groups(2), np.asarray([True, False, False, True]))
    assert np.allclose(values[0], [1, 1])
    assert np.allclose(values[1], [20, 20])
    assert mask[:2].all() and not mask[2:].any()


def test_ctc_viterbi_finds_ordered_targets():
    # vocab: blank=0, A=1, B=2; dominant sequence blank,A,A,blank,B,B
    probs = np.asarray([
        [.90, .05, .05], [.05, .90, .05], [.05, .90, .05],
        [.90, .05, .05], [.05, .05, .90], [.05, .05, .90]], np.float32)
    states, score = _ctc_viterbi(np.log(probs), [1, 2], 0)
    assert states.shape == (6,)
    assert any(s == 1 for s in states) and any(s == 3 for s in states)
    assert score < 0


def test_audio_extractor_returns_strict_74d(tmp_path):
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    y = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr); f.writeframes(y.tobytes())
    times, features, meta = extract_audio_features(wav)
    assert features.ndim == 2 and features.shape[1] == AUDIO_DIM
    assert len(times) == len(features)
    assert np.isfinite(features).all()
    assert meta["sample_rate"] == 16000
