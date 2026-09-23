import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features.extraction_pipeline import extract_visual_features, sha256_file

MODEL = ROOT / "src/features/model_cache/face_landmarker_v2_with_blendshapes.task"
VIDEO = ROOT / "data/videos_raw/-3g5yACwYnA/13.mp4"
EXPECTED_SHA256 = "41502960a7323080e5bbf46804f4fe6d0d83aa0aa82ba3209831286e94300083"


def test_official_model_initializes_and_emits_at_least_35_categories():
    assert MODEL.is_file()
    assert sha256_file(MODEL) == EXPECTED_SHA256
    times, features, valid, meta = extract_visual_features(VIDEO, MODEL, sample_hz=1.0)
    assert len(times) > 0
    assert features.shape == (len(times), 35)
    assert meta["blendshape_category_count"] >= 35
    assert meta["frames_detected"] > 0
    assert valid.any()
    assert float(np.abs(features[valid]).sum()) > 0.0


def test_smoke_outputs_have_real_visual_detections():
    outputs = sorted((ROOT / "artifacts/audit/feature_outputs/npz").glob("-3g5yACwYnA__*.npz"))
    assert len(outputs) >= 2
    for path in outputs[:2]:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata_json"]))
            assert meta["vision_meta"]["blendshape_category_count"] >= 35
            assert meta["vision_meta"]["frames_detected"] > 0
            assert data["vision_mask"].any()
            assert float(np.abs(data["vision"]).sum()) > 0.0
