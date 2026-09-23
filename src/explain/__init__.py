"""Model-agnostic explanation utilities for multimodal sentiment prediction."""
from .shapley import MODALITIES, ShapleyResult, exact_modality_shapley, normalize_contributions
from .occlusion import EvidenceWindow, aopc_random_deletion, local_window_occlusion, merge_adjacent_evidence
from .time_mapping import TimeMapping, map_position_window
from .quality import inspect_modalities, apply_unavailable_modality_policy
from .pipeline import explain_sample

__all__ = [
    "MODALITIES", "ShapleyResult", "exact_modality_shapley", "normalize_contributions",
    "EvidenceWindow", "aopc_random_deletion", "local_window_occlusion",
    "merge_adjacent_evidence", "TimeMapping", "map_position_window",
    "inspect_modalities", "apply_unavailable_modality_policy", "explain_sample",
]
