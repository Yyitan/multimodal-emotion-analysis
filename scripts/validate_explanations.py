#!/usr/bin/env python3
"""Validate explanation-set detailed JSON and flattened CSV without pandas."""
import argparse
import csv
import json
import math
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except ImportError:  # Semantic checks below remain dependency-free.
    Draft7Validator = None


def close(a, b, tol=1e-6):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def validate_detail(data, schema):
    errors = []
    if Draft7Validator is not None:
        errors.extend(
            "%s: %s" % ("/".join(map(str, error.path)), error.message)
            for error in sorted(Draft7Validator(schema).iter_errors(data), key=lambda item: list(item.path))
        )
    if not isinstance(data, dict):
        return ["root must be an object"]
    for field in schema.get("required", []):
        if field not in data:
            errors.append(field + ": required field is missing")
    if data.get("schema_version") != "1.0.0":
        errors.append("schema_version must be 1.0.0")
    probs = data.get("prediction", {}).get("probabilities", {})
    if probs and (set(probs) != {"Negative", "Neutral", "Positive"} or not close(sum(probs.values()), 1.0)):
        errors.append("prediction/probabilities: labels must be complete and values sum to 1")
    exp = data.get("explanation", {}); modalities = exp.get("modalities", {})
    expected_coalitions = {"empty", "text", "audio", "vision", "text+audio", "text+vision", "audio+vision", "text+audio+vision"}
    if set(exp.get("coalition_scores", {})) != expected_coalitions:
        errors.append("explanation/coalition_scores: all eight coalitions are required")
    degrees = [float(modalities.get(m, {}).get("degree", 0.0)) for m in ("text", "audio", "vision")]
    if exp.get("primary_modality") != "none" and not close(sum(degrees), 1.0):
        errors.append("explanation/modalities: degrees must sum to 1")
    for modality in ("text", "audio", "vision"):
        info = modalities.get(modality, {})
        if info.get("available") is False:
            if not close(info.get("shapley_signed", 1), 0) or not close(info.get("degree", 1), 0):
                errors.append("%s unavailable contribution must be zero" % modality)
            if exp.get("evidence", {}).get(modality):
                errors.append("%s unavailable evidence must be empty" % modality)
    for modality in ("audio", "vision"):
        for i, item in enumerate(exp.get("evidence", {}).get(modality, [])):
            time = item["time"]; start, end = time["start_sec"], time["end_sec"]
            if start is not None and end is not None and end < start:
                errors.append("%s evidence %d has end before start" % (modality, i))
            if time["method"].startswith("uniform") and (not time["approximate"] or time["confidence"] != "approximate"):
                errors.append("%s evidence %d uniform mapping must be approximate" % (modality, i))
            if modality == "vision" and item["keyframe_sec"] is not None and start is not None and not (start <= item["keyframe_sec"] <= end):
                errors.append("vision evidence %d keyframe outside interval" % i)
    if data.get("sample_id") == "13" and data.get("provenance", {}).get("feature_version") == "aligned_50":
        vision = modalities.get("vision", {})
        if vision.get("available") is not False or "VISION_ALL_ZERO" not in data.get("quality", {}).get("flags", []):
            errors.append("aligned sample 13 must flag unavailable all-zero vision")
    return errors

def parse_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def parse_number(value, field, errors):
    try:
        number = float(value)
        if not math.isfinite(number): raise ValueError
        return number
    except Exception:
        errors.append("%s must be a finite number" % field); return 0.0


def validate_csv(path, schema):
    errors = []
    columns = schema["columns"]; expected = [x["name"] for x in columns]
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected:
            errors.append("CSV header mismatch")
        for line, row in enumerate(reader, 2):
            prefix = "line %d: " % line
            if None in row:
                errors.append(prefix + "unexpected extra columns"); continue
            for spec in columns:
                name, value = spec["name"], row.get(spec["name"], "")
                if spec.get("required") and value == "": errors.append(prefix + name + " is required")
                typ = spec["type"]
                if value and (typ == "number" or typ == "number_or_null"):
                    number = parse_number(value, name, errors)
                    if "minimum" in spec and number < spec["minimum"]: errors.append(prefix + name + " below minimum")
                    if "maximum" in spec and number > spec["maximum"]: errors.append(prefix + name + " above maximum")
                if value and typ == "enum" and value not in spec["values"]: errors.append(prefix + name + " invalid enum")
                if value and typ == "boolean" and str(value).strip().lower() not in ("true", "false"):
                    errors.append(prefix + name + " must be true or false")
                if value and typ.startswith("json_array"):
                    try:
                        if not isinstance(json.loads(value), list): raise ValueError
                    except Exception: errors.append(prefix + name + " must be a JSON array")
            probs = [parse_number(row.get("prob_" + x, ""), "prob_" + x, errors)
                     for x in ("negative", "neutral", "positive")]
            if not close(sum(probs), 1): errors.append(prefix + "probabilities must sum to 1")
            degrees = [parse_number(row.get(x + "_degree", ""), x + "_degree", errors)
                       for x in ("text", "audio", "vision")]
            if row.get("primary_modality") != "none" and not close(sum(degrees), 1):
                errors.append(prefix + "degrees must sum to 1")
            for modality in ("text", "audio", "vision"):
                available = parse_bool(row.get(modality + "_available", ""))
                if not available:
                    if not close(row.get(modality + "_shapley_signed", 1), 0) or not close(row.get(modality + "_degree", 1), 0):
                        errors.append(prefix + modality + " unavailable contribution must be zero")
                    if row.get("primary_modality") == modality:
                        errors.append(prefix + "unavailable modality cannot be primary")
                    if modality == "text":
                        try: text_evidence = json.loads(row.get("text_evidence", "[]"))
                        except Exception: text_evidence = ["invalid"]
                        if text_evidence or row.get("text_location_method") != "unavailable":
                            errors.append(prefix + "unavailable text must have no evidence")
                    else:
                        time_fields = [modality + "_start_sec", modality + "_end_sec"]
                        if modality == "vision": time_fields.append("vision_keyframe_sec")
                        if any(row.get(field, "") != "" for field in time_fields) or row.get(modality + "_location_method") != "unavailable":
                            errors.append(prefix + "unavailable %s must have null time" % modality)
            if row.get("sample_id") == "13" and row.get("feature_version") == "aligned_50":
                try: flags = json.loads(row.get("quality_flags", "[]"))
                except Exception: flags = []
                if parse_bool(row.get("vision_available")) or "VISION_ALL_ZERO" not in flags:
                    errors.append(prefix + "aligned sample 13 must flag unavailable all-zero vision")
    return errors


def main():
    p = argparse.ArgumentParser(); p.add_argument("path"); p.add_argument("--schema", required=True)
    p.add_argument("--kind", choices=("json", "csv"), required=True); args = p.parse_args()
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    if args.kind == "json":
        errors = validate_detail(json.loads(Path(args.path).read_text(encoding="utf-8")), schema)
    else: errors = validate_csv(args.path, schema)
    if errors:
        for error in errors: print("ERROR", error)
        raise SystemExit(1)
    print("OK", args.path)


if __name__ == "__main__": main()
