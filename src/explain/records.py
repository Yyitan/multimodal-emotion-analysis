"""Build and flatten explanation-set explanation records."""
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

MODALITIES = ("text", "audio", "vision")
CSV_COLUMNS = (
    "schema_version", "feature_version", "sample_id", "polarity", "intensity", "prob_negative", "prob_neutral", "prob_positive",
    "text_shapley_signed", "audio_shapley_signed", "vision_shapley_signed",
    "text_degree", "audio_degree", "vision_degree", "primary_modality", "primary_confidence",
    "text_evidence", "audio_start_sec", "audio_end_sec", "vision_start_sec", "vision_end_sec",
    "vision_keyframe_sec", "text_location_method", "audio_location_method", "vision_location_method",
    "text_location_confidence", "audio_location_confidence", "vision_location_confidence",
    "text_available", "audio_available", "vision_available", "quality_flags", "explanation_json"
)


def _first(sequence: Sequence[Any], default: Any = None) -> Any:
    return sequence[0] if sequence else default


def _time_fields(evidence: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    item = _first(evidence)
    if item is None:
        return {"start": "", "end": "", "method": "unavailable", "confidence": "unavailable"}
    time = item["time"]
    return {"start": "" if time["start_sec"] is None else time["start_sec"],
            "end": "" if time["end_sec"] is None else time["end_sec"],
            "method": time["method"], "confidence": time["confidence"]}


def flatten_detail_record(record: Mapping[str, Any], explanation_json: str) -> Dict[str, Any]:
    """Flatten one validated detailed record into the required CSV row."""
    prediction = record["prediction"]; explanation = record["explanation"]
    modalities = explanation["modalities"]; evidence = explanation["evidence"]
    audio = _time_fields(evidence["audio"]); vision = _time_fields(evidence["vision"])
    visual_first = _first(evidence["vision"])
    text_first = _first(evidence["text"])
    row = {
        "schema_version": record["schema_version"],
        "feature_version": record["provenance"]["feature_version"],
        "sample_id": record["sample_id"], "polarity": prediction["polarity"],
        "intensity": prediction["intensity"],
        "prob_negative": prediction["probabilities"]["Negative"],
        "prob_neutral": prediction["probabilities"]["Neutral"],
        "prob_positive": prediction["probabilities"]["Positive"],
        "primary_modality": explanation["primary_modality"],
        "primary_confidence": explanation["primary_confidence"],
        "text_evidence": json.dumps([x["text"] for x in evidence["text"]], ensure_ascii=False),
        "audio_start_sec": audio["start"], "audio_end_sec": audio["end"],
        "vision_start_sec": vision["start"], "vision_end_sec": vision["end"],
        "vision_keyframe_sec": "" if visual_first is None else visual_first["keyframe_sec"],
        "text_location_method": "unavailable" if text_first is None else text_first["location_method"],
        "audio_location_method": audio["method"], "vision_location_method": vision["method"],
        "text_location_confidence": "unavailable" if text_first is None else text_first["location_confidence"],
        "audio_location_confidence": audio["confidence"], "vision_location_confidence": vision["confidence"],
        "quality_flags": json.dumps(record["quality"]["flags"], ensure_ascii=False),
        "explanation_json": explanation_json,
    }
    for modality in MODALITIES:
        row[modality + "_shapley_signed"] = modalities[modality]["shapley_signed"]
        row[modality + "_degree"] = modalities[modality]["degree"]
        row[modality + "_available"] = str(bool(modalities[modality]["available"])).lower()
    return {column: row[column] for column in CSV_COLUMNS}


def write_detail_json(record: Mapping[str, Any], path: Any) -> None:
    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary_csv(rows: Iterable[Mapping[str, Any]], path: Any) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
