from __future__ import annotations
import os, random
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic: bool

def set_global_seed(seed: int, deterministic: bool = True) -> SeedState:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    return SeedState(seed=seed, deterministic=deterministic)
