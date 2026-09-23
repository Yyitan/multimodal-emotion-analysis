"""Required classification and regression metrics with degenerate-case guards."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def compute_metrics(
    true_class, predicted_class, true_regression, predicted_regression
) -> Dict[str, float]:
    true_class = np.asarray(true_class)
    predicted_class = np.asarray(predicted_class)
    true_regression = np.asarray(true_regression, dtype=float)
    predicted_regression = np.asarray(predicted_regression, dtype=float)
    result: Dict[str, float] = {}
    class_mask = true_class >= 0
    if class_mask.any():
        result.update(
            {
                "accuracy": float(accuracy_score(true_class[class_mask], predicted_class[class_mask])),
                "macro_f1": float(f1_score(true_class[class_mask], predicted_class[class_mask], average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(true_class[class_mask], predicted_class[class_mask], average="weighted", zero_division=0)),
            }
        )
    else:
        result.update({"accuracy": float("nan"), "macro_f1": float("nan"), "weighted_f1": float("nan")})
    reg_mask = np.isfinite(true_regression) & np.isfinite(predicted_regression)
    if reg_mask.any():
        yt, yp = true_regression[reg_mask], predicted_regression[reg_mask]
        result["mae"] = float(mean_absolute_error(yt, yp))
        if len(yt) > 1 and np.std(yt) > 0 and np.std(yp) > 0:
            result["pearson"] = float(np.corrcoef(yt, yp)[0, 1])
        else:
            result["pearson"] = 0.0
    else:
        result.update({"mae": float("nan"), "pearson": float("nan")})
    return result
