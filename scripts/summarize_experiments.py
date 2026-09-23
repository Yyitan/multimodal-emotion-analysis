#!/usr/bin/env python3
"""Summarize real robustness outputs and generate paper-ready tables/figures."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ft2font import FT2Font
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.robustness import METRIC_NAMES


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _find_cjk_font() -> tuple[str | None, str]:
    """Select a font only after verifying representative Chinese glyphs."""
    candidates = (
        "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
        "SimHei", "WenQuanYi Zen Hei", "PingFang SC",
    )
    required = tuple(ord(char) for char in "缺失率性能模态位置混淆矩阵")
    installed = {font.name: font.fname for font in fm.fontManager.ttflist}
    for name in candidates:
        path = installed.get(name)
        if path:
            try:
                chars = FT2Font(path).get_charmap()
                if all(code in chars for code in required):
                    return name, f"Chinese labels enabled with verified font: {name}"
            except (OSError, RuntimeError):
                pass
    return None, (
        "No installed font was verified to contain all required Chinese glyphs; "
        "figures use English labels. The Chinese paper caption should state this explicitly."
    )


def _save_figure(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8g")


def _load_metrics(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise ValueError("No robustness metrics were supplied")
    result = pd.concat(frames, ignore_index=True)
    required = {
        "model", "family", "role", "training_seed", "split", "condition",
        "modalities", "missing_rate", "position", "n_spans", *METRIC_NAMES,
        "n_samples", "n_video_groups",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Metrics tables are missing columns: {sorted(missing)}")
    return result


def _multi_seed_table(data: pd.DataFrame) -> pd.DataFrame:
    keys = ["family", "role", "split", "condition", "modalities", "missing_rate", "position", "n_spans"]
    grouped = data.groupby(keys, dropna=False, sort=True)
    rows = []
    for values, part in grouped:
        row = dict(zip(keys, values))
        row["n_seeds"] = int(part["training_seed"].nunique())
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=1)) if len(part) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _model_card_multiseed(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read recorded validation candidates without loading candidate checkpoints."""
    if not path.is_file():
        return pd.DataFrame(), pd.DataFrame()
    card = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for candidate in card.get("candidates", []):
        tag = str(candidate.get("tag", ""))
        seed_text = tag.rsplit("seed", 1)[-1] if "seed" in tag else ""
        row: dict[str, Any] = {
            "tag": tag,
            "training_seed": int(seed_text) if seed_text.isdigit() else np.nan,
            "best_epoch": candidate.get("best_epoch"),
            "selection_score": candidate.get("selection_score"),
            "selected": tag == card.get("selected", {}).get("tag"),
        }
        for metric in METRIC_NAMES:
            row[metric] = candidate.get("valid", {}).get(metric, np.nan)
        rows.append(row)
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates, pd.DataFrame()
    summary: dict[str, Any] = {
        "split": str(card.get("selection_split", "valid")),
        "n_seeds": int(candidates["training_seed"].nunique()),
        "selection_rule": card.get("selection_rule", ""),
    }
    for metric in METRIC_NAMES:
        summary[f"{metric}_mean"] = float(candidates[metric].mean())
        summary[f"{metric}_std"] = float(candidates[metric].std(ddof=1)) if len(candidates) > 1 else 0.0
    summary["selection_score_mean"] = float(candidates["selection_score"].mean())
    summary["selection_score_std"] = float(candidates["selection_score"].std(ddof=1)) if len(candidates) > 1 else 0.0
    return candidates, pd.DataFrame([summary])


def _model_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, role, split), part in data.groupby(["family", "role", "split"], sort=True):
        clean = part[part["missing_rate"] == 0]
        missing = part[part["missing_rate"] > 0]
        row: dict[str, Any] = {
            "family": family,
            "role": role,
            "split": split,
            "n_models": int(part["model"].nunique()),
            "n_seeds": int(part["training_seed"].nunique()),
            "n_conditions": int(part["condition"].nunique()),
            "n_samples": int(part["n_samples"].max()),
            "n_video_groups": int(part["n_video_groups"].max()),
        }
        for metric in METRIC_NAMES:
            row[f"clean_{metric}_mean"] = float(clean[metric].mean()) if len(clean) else np.nan
            row[f"clean_{metric}_std"] = float(clean[metric].std(ddof=1)) if len(clean) > 1 else 0.0
            row[f"missing_grid_{metric}_mean"] = float(missing[metric].mean()) if len(missing) else np.nan
            row[f"missing_grid_{metric}_std"] = float(missing[metric].std(ddof=1)) if len(missing) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _comparison_table(data: pd.DataFrame, primary: str) -> pd.DataFrame:
    keys = ["split", "condition", "modalities", "missing_rate", "position", "n_spans"]
    means = data.groupby(["family", *keys], as_index=False)[list(METRIC_NAMES)].mean()
    reference = means[means["family"] == primary].drop(columns="family")
    if reference.empty:
        raise ValueError(f"Primary family {primary!r} is absent from metrics")
    rows = []
    for family in sorted(set(means["family"]) - {primary}):
        other = means[means["family"] == family]
        merged = reference.merge(other, on=keys, suffixes=("_primary", "_comparison"))
        for _, item in merged.iterrows():
            row = {key: item[key] for key in keys}
            row.update({"primary_family": primary, "comparison_family": family})
            for metric in METRIC_NAMES:
                first = float(item[f"{metric}_primary"])
                second = float(item[f"{metric}_comparison"])
                row[f"primary_{metric}"] = first
                row[f"comparison_{metric}"] = second
                row[f"delta_primary_minus_comparison_{metric}"] = first - second
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_rate_curves(data: pd.DataFrame, family: str, split: str, labels: dict[str, str], output: Path) -> list[str]:
    part = data[(data["family"] == family) & (data["split"] == split)]
    reduced = part.groupby(["modalities", "missing_rate"], as_index=False)[list(METRIC_NAMES)].mean()
    clean = reduced[reduced["missing_rate"] == 0]
    reduced = reduced[(reduced["modalities"] != "NONE") & (reduced["missing_rate"] > 0)]
    if not clean.empty and not reduced.empty:
        clean_metrics = clean[list(METRIC_NAMES)].mean()
        clean_rows = []
        for modality in sorted(reduced["modalities"].unique()):
            clean_rows.append({"modalities": modality, "missing_rate": 0.0, **clean_metrics.to_dict()})
        reduced = pd.concat([pd.DataFrame(clean_rows), reduced], ignore_index=True)
    if reduced.empty:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for modality, group in reduced.groupby("modalities"):
        group = group.sort_values("missing_rate")
        axes[0].plot(group["missing_rate"], group["macro_f1"], marker="o", label=modality)
        axes[1].plot(group["missing_rate"], group["mae"], marker="o", label=modality)
    axes[0].set_ylabel(labels["macro_f1"])
    axes[1].set_ylabel("MAE")
    for axis in axes:
        axis.set_xlabel(labels["missing_rate"])
        axis.grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"{family} — {split}")
    fig.tight_layout()
    return _save_figure(fig, output / "missing_rate_performance_curves")


def _plot_heatmap(data: pd.DataFrame, family: str, split: str, labels: dict[str, str], output: Path) -> list[str]:
    part = data[(data["family"] == family) & (data["split"] == split) & (data["missing_rate"] > 0)]
    table = part.groupby(["modalities", "missing_rate"])["macro_f1"].mean().unstack()
    if table.empty:
        return []
    fig, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
    axis.set_yticks(range(len(table.index)), table.index)
    axis.set_xlabel(labels["missing_rate"])
    axis.set_ylabel(labels["modality"])
    axis.set_title(labels["heatmap"])
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            value = table.iloc[i, j]
            axis.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if value < table.to_numpy().mean() else "black", fontsize=8)
    fig.colorbar(image, ax=axis, label=labels["macro_f1"])
    fig.tight_layout()
    return _save_figure(fig, output / "modality_type_heatmap")


def _plot_position(data: pd.DataFrame, family: str, split: str, labels: dict[str, str], output: Path) -> list[str]:
    part = data[(data["family"] == family) & (data["split"] == split) & (data["missing_rate"] > 0)]
    table = part.groupby(["position", "modalities"])["macro_f1"].mean().unstack()
    if table.empty:
        return []
    order = [position for position in ("head", "middle", "tail", "random") if position in table.index]
    table = table.loc[order]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    table.plot(kind="bar", ax=axis, width=0.82)
    axis.set_xlabel(labels["position"])
    axis.set_ylabel(labels["macro_f1"])
    axis.set_title(labels["position_effect"])
    axis.legend(title=labels["modality"], ncol=4, fontsize=8, frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return _save_figure(fig, output / "position_effect")


def _confusion(predictions: Path, family: str, split: str, labels: dict[str, str], table_dir: Path, figure_dir: Path) -> list[str]:
    if not predictions.is_file():
        return []
    frame = pd.read_csv(predictions)
    part = frame[(frame["family"] == family) & (frame["split"] == split)]
    if part.empty:
        return []
    # Average seed predictions are not categorical; emit one honest matrix per
    # checkpoint/model instead of silently mixing repeated labels across seeds.
    outputs = []
    for model, values in part.groupby("model"):
        classes = sorted(set(values["true_class"].astype(int)) | set(values["predicted_class"].astype(int)))
        matrix = confusion_matrix(values["true_class"], values["predicted_class"], labels=classes)
        table = pd.DataFrame(matrix, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes])
        table_path = table_dir / f"confusion_matrix_{split}_{model}.csv"
        table_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, encoding="utf-8-sig")
        fig, axis = plt.subplots(figsize=(5.4, 4.6))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(classes)), classes)
        axis.set_yticks(range(len(classes)), classes)
        axis.set_xlabel(labels["predicted"])
        axis.set_ylabel(labels["true"])
        axis.set_title(f"{labels['confusion']} — {model}")
        for i in range(len(classes)):
            for j in range(len(classes)):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        outputs.extend(_save_figure(fig, figure_dir / f"confusion_matrix_{split}_{model}"))
        outputs.append(str(table_path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/robustness.yaml")
    parser.add_argument("--metrics", nargs="+", help="One or more robustness CSV files")
    parser.add_argument("--predictions", help="Clean prediction CSV for confusion matrices")
    parser.add_argument("--split", choices=("valid", "test"), default=None)
    parser.add_argument("--primary-family")
    parser.add_argument("--table-dir")
    parser.add_argument("--figure-dir")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    table_dir = Path(args.table_dir or config.get("output", {}).get("metrics_dir", "artifacts/metrics"))
    figure_dir = Path(args.figure_dir or config.get("output", {}).get("figures_dir", "artifacts/figures"))
    metric_paths = [Path(path) for path in args.metrics] if args.metrics else [table_dir / "robustness_valid.csv"]
    data = _load_metrics(metric_paths)
    split = args.split or str(config.get("summary", {}).get("figure_split", "valid"))
    primary = args.primary_family or str(config.get("summary", {}).get("primary_family", "gated"))
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    seed_table = _multi_seed_table(data)
    summary = _model_summary(data)
    comparison = _comparison_table(data, primary)
    roles = data[["family", "role"]].drop_duplicates().set_index("family")["role"].to_dict()
    if comparison.empty:
        ablation = comparison.copy()
    else:
        ablation = comparison[
            comparison["comparison_family"].map(roles).astype(str) == "ablation"
        ].copy()
    model_card_value = config.get("summary", {}).get("multiseed_model_card")
    card_candidates, card_summary = (
        _model_card_multiseed(_expand(str(model_card_value)))
        if model_card_value else (pd.DataFrame(), pd.DataFrame())
    )
    _write_table(seed_table, table_dir / "robustness_multiseed_mean_std.csv")
    _write_table(summary, table_dir / "robustness_model_summary.csv")
    _write_table(comparison, table_dir / "robustness_baseline_comparison.csv")
    _write_table(ablation, table_dir / "robustness_ablation.csv")
    _write_table(card_candidates, table_dir / "training_seed_validation_metrics.csv")
    _write_table(card_summary, table_dir / "training_seed_mean_std.csv")

    font, font_note = _find_cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font]
        plt.rcParams["axes.unicode_minus"] = False
        labels = {
            "missing_rate": "缺失率", "macro_f1": "宏平均 F1", "modality": "缺失模态",
            "position": "缺失位置", "heatmap": "模态类型影响", "position_effect": "缺失位置影响",
            "predicted": "预测类别", "true": "真实类别", "confusion": "混淆矩阵",
        }
        language = "Chinese"
    else:
        labels = {
            "missing_rate": "Missing rate", "macro_f1": "Macro-F1", "modality": "Missing modality",
            "position": "Span position", "heatmap": "Effect of missing modality type",
            "position_effect": "Effect of span position", "predicted": "Predicted class",
            "true": "True class", "confusion": "Confusion matrix",
        }
        language = "English"

    outputs = []
    outputs += _plot_rate_curves(data, primary, split, labels, figure_dir)
    outputs += _plot_heatmap(data, primary, split, labels, figure_dir)
    outputs += _plot_position(data, primary, split, labels, figure_dir)
    prediction_path = Path(args.predictions) if args.predictions else table_dir / f"robustness_{split}_clean_predictions.csv"
    outputs += _confusion(prediction_path, primary, split, labels, table_dir, figure_dir)
    status = {
        "source_metrics": [str(path) for path in metric_paths],
        "source_predictions": str(prediction_path),
        "primary_family": primary,
        "figure_split": split,
        "font_language": language,
        "font_note_for_chinese_caption": font_note,
        "generated_outputs": outputs,
        "tables": [
            str(table_dir / "robustness_multiseed_mean_std.csv"),
            str(table_dir / "robustness_model_summary.csv"),
            str(table_dir / "robustness_baseline_comparison.csv"),
            str(table_dir / "robustness_ablation.csv"),
            str(table_dir / "training_seed_validation_metrics.csv"),
            str(table_dir / "training_seed_mean_std.csv"),
        ],
    }
    status_path = table_dir / "robustness_summary_manifest.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
