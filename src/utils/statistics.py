from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np

@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    sampling_unit: str

def grouped_bootstrap_ci(
    values: Sequence,
    groups: Sequence,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260923,
) -> ConfidenceInterval:
    values=np.asarray(values)
    groups=np.asarray(groups)
    if len(values)!=len(groups) or len(values)==0:
        raise ValueError('values and groups must have the same non-zero length')
    unique=np.unique(groups)
    if len(unique)<2:
        raise ValueError('at least two independent groups are required')
    by_group={g:np.flatnonzero(groups==g) for g in unique}
    rng=np.random.default_rng(seed); samples=np.empty(n_resamples,dtype=float)
    for i in range(n_resamples):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([by_group[g] for g in sampled])
        samples[i]=statistic(values[idx])
    alpha=(1-confidence)/2
    return ConfidenceInterval(
        estimate=float(statistic(values)),
        lower=float(np.quantile(samples,alpha)),
        upper=float(np.quantile(samples,1-alpha)),
        confidence=confidence,
        resamples=n_resamples,
        sampling_unit='group',
    )

def paired_grouped_bootstrap_difference(
    first: Sequence,
    second: Sequence,
    groups: Sequence,
    statistic: Callable[[np.ndarray], float],
    **kwargs,
) -> ConfidenceInterval:
    first=np.asarray(first); second=np.asarray(second)
    if first.shape!=second.shape:
        raise ValueError('paired arrays must have identical shapes')
    return grouped_bootstrap_ci(first-second,groups,statistic,**kwargs)
