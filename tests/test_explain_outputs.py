import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.explain.records import CSV_COLUMNS, flatten_detail_record, write_detail_json, write_summary_csv


def sample13_record():
    zero_modal = {"available": False, "shapley_signed": 0.0, "degree": 0.0}
    return {
        "schema_version": "1.0.0",
        "sample_id": "13",
        "prediction": {
            "polarity": "Neutral", "intensity": 0.0,
            "probabilities": {"Negative": 0.2, "Neutral": 0.6, "Positive": 0.2},
        },
        "explanation": {
            "target_output": "predicted_class_probability:Neutral",
            "baseline_score": 0.3, "full_score": 0.6,
            "coalition_scores": {
                "empty": 0.3, "text": 0.5, "audio": 0.4, "vision": 0.3,
                "text+audio": 0.6, "text+vision": 0.5, "audio+vision": 0.4,
                "text+audio+vision": 0.6,
            },
            "modalities": {
                "text": {"available": True, "shapley_signed": 0.2, "degree": 2 / 3},
                "audio": {"available": True, "shapley_signed": 0.1, "degree": 1 / 3},
                "vision": zero_modal,
            },
            "primary_modality": "text", "primary_tied_modalities": ["text"],
            "primary_confidence": "unique", "efficiency_residual": 0.0,
            "evidence": {
                "text": [{"start_index": 1, "end_index": 2, "score_drop": 0.2,
                          "importance": 0.2, "text": "example",
                          "location_method": "token_alignment", "location_confidence": "exact"}],
                "audio": [{"start_index": 3, "end_index": 5, "score_drop": 0.1,
                           "importance": 0.1,
                           "time": {"start_sec": 0.5208, "end_sec": 0.868,
                                    "method": "uniform_50_position", "confidence": "approximate",
                                    "approximate": True, "note": "uniform fallback"}}],
                "vision": [],
            },
        },
        "faithfulness": {
            "metric": "AOPC cumulative deletion", "ranked_aopc": 0.2,
            "random_aopc_mean": 0.1, "random_aopc_std": 0.02,
            "aopc_gain_over_random": 0.1, "steps": 2, "random_trials": 100, "seed": 7,
        },
        "quality": {"flags": ["VISION_ALL_ZERO"],
                    "notes": ["aligned visual tensor is all zero; no visual timestamp emitted"]},
        "provenance": {
            "feature_version": "aligned_50",
            "explainer": "exact_3modality_shapley_plus_local_occlusion",
            "explainer_config": "configs/explanation.yaml",
            "model_reference": "toy-test-only",
        },
    }


def run_validator(root, path, schema, kind):
    return subprocess.run(
        [sys.executable, str(root / "scripts/validate_explanations.py"), str(path),
         "--schema", str(schema), "--kind", kind],
        text=True, capture_output=True,
    )


def test_sample13_detailed_json_and_flat_csv_validate(tmp_path):
    root = Path(__file__).resolve().parents[1]; record = sample13_record()
    detail = tmp_path / "13.json"; write_detail_json(record, detail)
    detail_result = run_validator(root, detail,
                                  root / "configs/explanation_set_detail.schema.json", "json")
    assert detail_result.returncode == 0, detail_result.stdout + detail_result.stderr

    row = flatten_detail_record(record, "details/13.json")
    summary = tmp_path / "explanation_set.csv"; write_summary_csv([row], summary)
    csv_result = run_validator(root, summary,
                               root / "configs/explanation_set_csv.schema.json", "csv")
    assert csv_result.returncode == 0, csv_result.stdout + csv_result.stderr
    with open(summary, encoding="utf-8-sig", newline="") as handle:
        parsed = next(csv.DictReader(handle))
    assert tuple(parsed) == CSV_COLUMNS
    assert parsed["vision_available"] == "false"
    assert parsed["vision_start_sec"] == ""
    assert parsed["vision_keyframe_sec"] == ""
    assert parsed["vision_location_method"] == "unavailable"
    assert "VISION_ALL_ZERO" in json.loads(parsed["quality_flags"])


def test_validator_rejects_fabricated_sample13_visual_evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]; record = sample13_record()
    record["explanation"]["modalities"]["vision"] = {
        "available": True, "shapley_signed": 0.1, "degree": 0.1,
    }
    path = tmp_path / "bad13.json"; write_detail_json(record, path)
    result = run_validator(root, path, root / "configs/explanation_set_detail.schema.json", "json")
    assert result.returncode == 1
    assert "aligned sample 13" in result.stdout


def test_csv_schema_header_matches_writer_contract():
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "configs/explanation_set_csv.schema.json").read_text(encoding="utf-8"))
    assert tuple(column["name"] for column in schema["columns"]) == CSV_COLUMNS
    assert schema["schema_version"] == "1.0.0"
