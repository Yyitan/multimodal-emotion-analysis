#!/usr/bin/env python3
"""Rebuild repository figures from committed artifacts only.

Figures regenerated here carry no data beyond what is already committed under
artifacts/, so they can be rebuilt on any machine with matplotlib + pandas.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts/figures"
PAPER = FIG / "paper"

BLUE = "#345C8C"
ORANGE = "#BB7C3C"
GREEN = "#4C8464"
RED = "#AC5454"
GRAY = "#647484"


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=110, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.with_suffix(".png").name, "and .pdf")


def fig01_overall_workflow() -> None:
    steps = [
        "Raw video\n& transcript",
        "T/A/V feature\nextraction",
        "Alignment\n& masks",
        "Robust gated\nfusion",
        "Dual-task\nprediction",
        "Shapley +\nlocal evidence",
    ]
    fig, ax = plt.subplots(figsize=(16, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [0.85 + i * 1.70 for i in range(6)]
    texts = []
    for center, label in zip(centers, steps):
        texts.append(ax.text(center, 0.62, label, ha="center", va="center",
                             fontsize=16.5, fontweight="bold", color="#1a1a1a",
                             bbox=dict(boxstyle="round,pad=0.55", facecolor="#EEF3F9",
                                       edgecolor=BLUE, linewidth=2.2)))
    fig.canvas.draw()
    inv = ax.transData.inverted()
    spans = []
    for text in texts:
        box = text.get_window_extent(fig.canvas.get_renderer())
        (x0, _), (x1, _) = inv.transform([(box.x0, box.y0), (box.x1, box.y1)])
        spans.append((x0, x1))
    for i in range(5):
        ax.annotate("", xy=(spans[i + 1][0] - 0.03, 0.62), xytext=(spans[i][1] + 0.03, 0.62),
                    arrowprops=dict(arrowstyle="-|>", color="#697686",
                                    linewidth=2.6, shrinkA=0, shrinkB=0))
    labels = [(2.55, "Module 1: auditable alignment", GREEN),
              (None, "Module 2: missingness robustness", ORANGE),
              (None, "Module 3: faithful explanation", RED)]
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def measure(text, size):
        probe = ax.text(0, 0.14, text, fontsize=size, alpha=0)
        box = probe.get_window_extent(renderer)
        (x0, _), (x1, _) = inv.transform([(box.x0, box.y0), (box.x1, box.y1)])
        probe.remove()
        return x1 - x0

    gap = 0.35
    for size in (15, 14.5, 14, 13.5, 13, 12.5):
        widths = [measure(t, size) for _, t, _ in labels]
        x1 = labels[0][0] - widths[0] / 2
        x2 = x1 + widths[0] + gap
        x3 = x2 + widths[1] + gap
        if x3 + widths[2] <= 9.95:
            for x, (label, color) in zip((x1, x2, x3),
                                         [(t, c) for _, t, c in labels]):
                ax.text(x, 0.14, label, ha="left", va="center",
                        fontsize=size, color=color)
            break
    save(fig, PAPER / "fig01_overall_workflow")


def fig03_quality_audit() -> None:
    panels = [
        ("Generation", [("success", 100, BLUE), ("failed", 0, ORANGE)]),
        ("Alignment", [("CTC", 88, BLUE), ("approx.", 12, ORANGE)]),
        ("Visual availability", [("face", 87, BLUE), ("no face", 13, ORANGE)]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19.8, 5.7))
    for ax, (title, bars) in zip(axes, panels):
        names = [b[0] for b in bars]
        values = [b[1] for b in bars]
        colors = [b[2] for b in bars]
        ax.bar(names, values, color=colors, width=0.62)
        for x, v in enumerate(values):
            ax.text(x, v + 2.5, str(v), ha="center", fontsize=15, fontweight="bold")
        ax.set_title(title, fontsize=17)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(labelsize=13)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Feature-extraction full-set quality audit (n=100)",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, PAPER / "fig03_quality_audit")


def fig05_modality_contribution() -> None:
    details = sorted((ROOT / "artifacts/explanations/explanation_set/details").glob("*.json"))
    degrees = {m: [] for m in ("text", "audio", "vision")}
    for path in details:
        record = json.loads(path.read_text(encoding="utf-8"))
        modalities = record["explanation"]["modalities"]
        for m in degrees:
            degrees[m].append(float(modalities[m]["degree"]))
    summary = pd.read_csv(ROOT / "artifacts/predictions/explanation_set_predictions_explanations.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.8, 7.5), width_ratios=[2.2, 1])
    x = np.arange(len(details))
    bottom = np.zeros(len(details))
    colors = {"text": BLUE, "audio": GREEN, "vision": ORANGE}
    for m in ("text", "audio", "vision"):
        values = np.asarray(degrees[m])
        ax1.bar(x, values, bottom=bottom, color=colors[m], width=0.72, label=m)
        bottom += values
    ax1.set_title("Per-sample absolute Shapley degree", fontsize=17)
    ax1.set_xticks(x, [p.stem for p in details], rotation=90, fontsize=12)
    ax1.set_ylim(0, 1.02)
    ax1.legend(loc="lower left", ncols=3, fontsize=14, frameon=False)
    ax1.tick_params(labelsize=13)

    means = summary[[f"{m}_degree" for m in ("text", "audio", "vision")]].mean()
    bars = ax2.bar(["text", "audio", "vision"], means.values,
                   color=[BLUE, GREEN, ORANGE], width=0.55)
    for bar, v in zip(bars, means.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", fontsize=15, fontweight="bold")
    ax2.set_title("Mean over evaluation set", fontsize=17)
    ax2.set_ylim(0, 1.02)
    ax2.tick_params(labelsize=14)
    fig.suptitle("Modality contribution (20 samples with explanations)",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, PAPER / "fig05_modality_contribution")


def fig07_faithfulness() -> None:
    payload = json.loads((ROOT / "artifacts/metrics/faithfulness_valid.json").read_text(encoding="utf-8"))
    targets = payload["summary"]["targets"]
    order = ["classification_predicted_probability", "regression_value"]
    labels = ["classification_predicted_probability", "regression_value"]
    ranked = [targets[t]["top_evidence_ranked_aopc_mean"] for t in order]
    random_aopc = [targets[t]["random_deletion_aopc_mean_mean"] for t in order]
    rates = [targets[t]["primary_drop_exceeds_secondary_rate"] for t in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.8, 6.6))
    x = np.arange(2)
    width = 0.32
    ax1.bar(x - width / 2, ranked, width, color=BLUE, label="ranked")
    ax1.bar(x + width / 2, random_aopc, width, color=GRAY, label="random")
    ax1.set_title("Mean AOPC", fontsize=17)
    ax1.set_xticks(x, labels)
    ax1.tick_params(labelsize=13)
    ax1.legend(fontsize=14, frameon=False)

    bars = ax2.bar(x, rates, 0.5, color=[GREEN, ORANGE])
    for bar, v in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v * 100:.2f}%",
                 ha="center", fontsize=15, fontweight="bold")
    ax2.axhline(0.5, ls="--", color="#697686", linewidth=2)
    ax2.set_title("P(primary deletion > secondary)", fontsize=17)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=13)
    fig.suptitle("Faithfulness on validation set (n=728, two targets)",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, PAPER / "fig07_faithfulness")


def fig08_evaluation_set_distribution() -> None:
    summary = pd.read_csv(ROOT / "artifacts/predictions/explanation_set_predictions_explanations.csv")
    counts = summary["polarity"].value_counts()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.8, 6.2))
    bars = ax1.bar(["Negative", "Neutral", "Positive"],
                   [counts.get(k, 0) for k in ("Negative", "Neutral", "Positive")],
                   color=[RED, GRAY, BLUE], width=0.55)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{int(bar.get_height())}", ha="center", fontsize=15, fontweight="bold")
    ax1.set_title("Predicted polarity", fontsize=17)
    ax1.set_ylim(0, 9.9)
    ax1.tick_params(labelsize=14)

    ax2.hist(summary["intensity"], bins=np.arange(-2.25, 1.5, 0.5),
             color=BLUE, edgecolor="white", linewidth=1.2)
    ax2.set_title("Predicted intensity", fontsize=17)
    ax2.tick_params(labelsize=14)
    fig.suptitle("Evaluation set: unlabeled outputs, not accuracy",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, PAPER / "fig08_evaluation_set_distribution")


def confusion_matrix(split: str) -> None:
    if split == "valid":
        matrix = pd.read_csv(
            ROOT / "artifacts/metrics/confusion_matrix_valid_frozen_robust_gated.csv",
            index_col=0).to_numpy()
        name = "confusion_matrix_valid_frozen_robust_gated"
    else:
        frame = pd.read_csv(ROOT / "artifacts/metrics/robustness_test_clean_predictions.csv")
        matrix = np.zeros((3, 3), dtype=int)
        for true, pred in zip(frame["true_class"], frame["predicted_class"]):
            matrix[int(true), int(pred)] += 1
        name = "confusion_matrix_test_frozen_robust_gated"

    fig, ax = plt.subplots(figsize=(7.7, 6.9))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=17)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xlabel("Predicted class", fontsize=15)
    ax.set_ylabel("True class", fontsize=15)
    ax.set_title(f"Confusion matrix — frozen_robust_gated", fontsize=18)
    ax.tick_params(labelsize=13)
    save(fig, FIG / name)
    # duplicate into paper/ with robust_ prefix (same figure, pre-registered split)
    prefix = "robust_valid_" if split == "valid" else "robust_test_"
    for ext in (".png", ".pdf"):
        source = (FIG / name).with_suffix(ext)
        target = (PAPER / (prefix + name)).with_suffix(ext)
        target.write_bytes(source.read_bytes())
    print("copied to", (PAPER / (prefix + name)).with_suffix(".png").name)


def main() -> None:
    fig01_overall_workflow()
    fig03_quality_audit()
    fig05_modality_contribution()
    fig07_faithfulness()
    fig08_evaluation_set_distribution()
    confusion_matrix("valid")
    confusion_matrix("test")


if __name__ == "__main__":
    main()
