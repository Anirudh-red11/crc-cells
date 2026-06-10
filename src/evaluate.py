"""
evaluate.py — Metrics, comparison, ensembling (workflow steps E, F, G)
======================================================================
  • compute_metrics      — accuracy + macro precision/recall/F1
  • plot_confusion       — single confusion matrix (counts + normalised)
  • comparison_table     — tidy dataframe ranking all models
  • plot_comparison      — bar chart of macro-F1 per model
  • plot_history         — CNN training curves
  • soft_vote            — probability-averaging ensemble (step E)
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion(y_true, y_pred, class_names: Sequence[str],
                   title: str = "Confusion matrix") -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(5 + 2 * len(class_names), 5))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "counts"),
                               (axes[1], cmn, ".2f", "row-normalised (recall)")):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=False, ax=ax, linewidths=0.3)
        ax.set(xlabel="predicted", ylabel="true", title=sub)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def comparison_table(val_metrics: Dict[str, Dict[str, float]],
                     test_metrics: Dict[str, Dict[str, float]] | None = None
                     ) -> pd.DataFrame:
    """Build a ranked table from {model_name: metrics_dict}."""
    rows = []
    for name, vm in val_metrics.items():
        row = {"model": name,
               "val_acc": vm["accuracy"], "val_f1": vm["f1_macro"]}
        if test_metrics and name in test_metrics:
            row["test_acc"] = test_metrics[name]["accuracy"]
            row["test_f1"] = test_metrics[name]["f1_macro"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("val_f1", ascending=False).reset_index(drop=True)
    return df.round(4)


def plot_comparison(table: pd.DataFrame, metric: str = "val_f1",
                    title: str = "Model comparison") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(table) + 1.5))
    t = table.sort_values(metric)
    ax.barh(t["model"], t[metric], color="#4c72b0")
    for i, v in enumerate(t[metric]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    ax.set(xlabel=f"macro {metric}", title=title, xlim=(0, 1))
    fig.tight_layout()
    return fig


def plot_history(hist, title: str = "Training history") -> plt.Figure:
    ep = range(1, len(hist.train_loss) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(ep, hist.train_loss, label="train"); a1.plot(ep, hist.val_loss, "--", label="val")
    a1.set(title="loss", xlabel="epoch"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(ep, hist.train_acc, label="train acc")
    a2.plot(ep, hist.val_acc, "--", label="val acc")
    a2.plot(ep, hist.val_f1, ":", label="val macroF1")
    a2.set(title="accuracy / F1", xlabel="epoch", ylim=(0, 1)); a2.legend(); a2.grid(alpha=.3)
    fig.suptitle(title, fontweight="bold"); fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Ensembling (step E)
# ──────────────────────────────────────────────────────────────────────────────
def soft_vote(prob_list: List[np.ndarray], weights: List[float] | None = None
              ) -> np.ndarray:
    """
    Average class probabilities from several models → ensemble prediction.

    prob_list : list of (N, C) probability arrays (same N, C, sample order)
    returns   : (N,) predicted class indices
    """
    weights = weights or [1.0] * len(prob_list)
    w = np.array(weights) / np.sum(weights)
    avg = sum(wi * p for wi, p in zip(w, prob_list))
    return avg.argmax(axis=1)
