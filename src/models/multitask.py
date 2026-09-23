"""Lightweight baselines and a mask-aware gated multimodal multi-task model."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
from torch import nn

DEFAULT_DIMS = {"text": 768, "audio": 74, "vision": 35}


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(x.dtype).unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return masked_mean(h, observed)


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "logits": self.classifier(h),
            "regression": 3.0 * torch.tanh(self.regressor(h).squeeze(-1)),
        }


class ConcatBaseline(nn.Module):
    """Single-modal or concatenation baseline, selected through `modalities`."""

    def __init__(
        self,
        modalities: Iterable[str] = ("text", "audio", "vision"),
        input_dims: Mapping[str, int] = DEFAULT_DIMS,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        if not self.modalities or not set(self.modalities).issubset(input_dims):
            raise ValueError(f"Invalid modalities {self.modalities}")
        self.encoders = nn.ModuleDict(
            {m: SequenceEncoder(input_dims[m], hidden_dim, dropout) for m in self.modalities}
        )
        self.head = MultiTaskHead(hidden_dim * len(self.modalities), hidden_dim, dropout)

    def forward(self, features, observed, valid=None):
        pooled = [self.encoders[m](features[m], observed[m]) for m in self.modalities]
        output = self.head(torch.cat(pooled, dim=-1))
        output["embedding"] = torch.cat(pooled, dim=-1)
        return output


class GatedMultimodalMultiTask(nn.Module):
    """Mask-aware gated fusion with explicit availability/coverage signals."""

    def __init__(
        self,
        input_dims: Mapping[str, int] = DEFAULT_DIMS,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.modalities = tuple(input_dims)
        self.encoders = nn.ModuleDict(
            {m: SequenceEncoder(input_dims[m], hidden_dim, dropout) for m in self.modalities}
        )
        self.missing_tokens = nn.ParameterDict(
            {m: nn.Parameter(torch.zeros(hidden_dim)) for m in self.modalities}
        )
        self.gates = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(hidden_dim + 2, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1)
                )
                for m in self.modalities
            }
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.head = MultiTaskHead(hidden_dim, hidden_dim, dropout)

    def forward(self, features, observed, valid=None):
        if valid is None:
            valid = torch.ones_like(next(iter(observed.values())), dtype=torch.bool)
        representations, gate_logits, coverages = [], [], []
        valid_count = valid.sum(1).clamp_min(1).to(features[self.modalities[0]].dtype)
        availability = []
        for m in self.modalities:
            obs = observed[m] & valid
            pooled = self.encoders[m](features[m], obs)
            available = obs.any(1)
            coverage = obs.sum(1).to(pooled.dtype) / valid_count
            pooled = torch.where(available[:, None], pooled, self.missing_tokens[m][None])
            gate_input = torch.cat([pooled, coverage[:, None], available.to(pooled.dtype)[:, None]], dim=-1)
            logit = self.gates[m](gate_input).squeeze(-1)
            representations.append(pooled)
            gate_logits.append(logit)
            coverages.append(coverage)
            availability.append(available)
        logits = torch.stack(gate_logits, dim=1)
        available = torch.stack(availability, dim=1)
        # All-missing is still defined via learned tokens; normally text is always available.
        all_missing = ~available.any(1)
        available = available.clone()
        available[all_missing] = True
        logits = logits.masked_fill(~available, -1e4)
        gates = torch.softmax(logits, dim=1)
        reps = torch.stack(representations, dim=1)
        fused = self.fusion_norm((reps * gates.unsqueeze(-1)).sum(1))
        output = self.head(fused)
        output.update(
            {
                "embedding": fused,
                "gates": gates,
                "coverage": torch.stack(coverages, dim=1),
            }
        )
        return output


def build_model(config: Mapping) -> nn.Module:
    model_cfg = config.get("model", config)
    kind = model_cfg.get("type", "gated")
    common = {
        "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
    }
    if kind in {"single", "concat", "baseline"}:
        return ConcatBaseline(modalities=model_cfg.get("modalities", ["text", "audio", "vision"]), **common)
    if kind == "gated":
        modalities = tuple(model_cfg.get("modalities", DEFAULT_DIMS))
        if not modalities or not set(modalities).issubset(DEFAULT_DIMS):
            raise ValueError(f"Invalid modalities {modalities}")
        return GatedMultimodalMultiTask(
            input_dims={name: DEFAULT_DIMS[name] for name in modalities}, **common
        )
    raise ValueError(f"Unknown model type: {kind}")
