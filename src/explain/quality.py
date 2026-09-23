"""Input quality inspection and unavailable-modality policy."""
from typing import Any, Dict, Mapping
import numpy as np
from .shapley import MODALITIES


def inspect_modalities(sample: Mapping[str, Any], zero_atol: float = 0.0) -> Dict[str, Any]:
    availability, flags, details = {}, [], {}
    for modality in MODALITIES:
        if modality not in sample:
            availability[modality] = False; flags.append(modality.upper() + "_MISSING")
            details[modality] = {"present": False, "all_zero": None, "finite": None}
            continue
        arr = np.asarray(sample[modality])
        finite = bool(np.all(np.isfinite(arr))) if np.issubdtype(arr.dtype, np.number) else False
        all_zero = bool(np.all(np.abs(arr) <= zero_atol)) if finite else False
        available = finite and arr.size > 0 and not all_zero
        availability[modality] = available
        if all_zero:
            flags.append(modality.upper() + "_ALL_ZERO")
        if not finite:
            flags.append(modality.upper() + "_NONFINITE")
        details[modality] = {"present": True, "shape": list(arr.shape),
                             "all_zero": all_zero, "finite": finite}
    return {"availability": availability, "quality_flags": flags, "details": details}


def apply_unavailable_modality_policy(explanation: Mapping[str, Any],
                                      quality: Mapping[str, Any]) -> Dict[str, Any]:
    """Clear unavailable contributions/evidence and prevent false primary labels.

    This is the explicit handling for explanation-set sample 13's aligned visual
    all-zero anomaly, and applies the same conservative rule to any sample.
    """
    result = dict(explanation); availability = dict(quality["availability"])
    signed = dict(result.get("signed_contributions", {})); degrees = dict(result.get("degrees", {}))
    evidence = dict(result.get("evidence", {}))
    for modality, is_available in availability.items():
        if not is_available:
            signed[modality] = 0.0; degrees[modality] = 0.0; evidence[modality] = []
    total = sum(max(float(degrees.get(m, 0.0)), 0.0) for m in MODALITIES
                if availability.get(m, False))
    if total > 0:
        degrees = {m: (max(float(degrees.get(m, 0.0)), 0.0) / total
                       if availability.get(m, False) else 0.0) for m in MODALITIES}
        primary = max((m for m in MODALITIES if availability.get(m, False)), key=lambda m: degrees[m])
    else:
        degrees = {m: 0.0 for m in MODALITIES}; primary = "none"
    result.update({"signed_contributions": signed, "degrees": degrees, "evidence": evidence,
                   "primary_modality": primary, "modality_availability": availability,
                   "quality_flags": list(quality.get("quality_flags", []))})
    return result
