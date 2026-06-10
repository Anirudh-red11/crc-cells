"""
eda.py — Exploratory Data Analysis (workflow step B)
=====================================================
Unsupervised exploration of the cell-image data:
  • class-frequency bar charts (both tasks)
  • mean image per class
  • PCA (linear variance directions)
  • t-SNE / UMAP 2-D embeddings
  • k-means clustering on PCA features vs. true labels

All functions return the Matplotlib Figure so the notebook can display/save it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from data import CELLTYPE_NAMES

sns.set_style("whitegrid")
_PALETTE = {
    "fibroblast": "#4c72b0",
    "inflammatory": "#55a868",
    "epithelial": "#c44e52",
    "others": "#8172b3",
    "non-cancerous": "#4c72b0",
    "cancerous": "#c44e52",
}


# ──────────────────────────────────────────────────────────────────────────────
def plot_class_distributions(df: pd.DataFrame) -> plt.Figure:
    """Bar charts for isCancerous and cellTypeName frequencies."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    binary = df["isCancerous"].map({0: "non-cancerous", 1: "cancerous"}).value_counts()
    axes[0].bar(binary.index, binary.values,
                color=[_PALETTE[k] for k in binary.index])
    axes[0].set_title("Binary task — isCancerous", fontweight="bold")
    axes[0].set_ylabel("cell count")
    for i, v in enumerate(binary.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    ct = df["cellTypeName"].dropna().value_counts()
    order = [CELLTYPE_NAMES[i] for i in range(4) if CELLTYPE_NAMES[i] in ct.index]
    ct = ct.reindex(order)
    axes[1].bar(ct.index, ct.values, color=[_PALETTE[k] for k in ct.index])
    axes[1].set_title("Multiclass task — cell type", fontweight="bold")
    axes[1].set_ylabel("cell count")
    for i, v in enumerate(ct.values):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Class distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_mean_images(images: np.ndarray, labels: np.ndarray,
                     class_names: dict) -> plt.Figure:
    """Average image per class — reveals colour/texture differences."""
    classes = sorted(class_names)
    fig, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 3.2))
    if len(classes) == 1:
        axes = [axes]
    for ax, c in zip(axes, classes):
        mean_img = images[labels == c].mean(axis=0)
        ax.imshow(np.clip(mean_img, 0, 1))
        ax.set_title(f"{class_names[c]}\n(n={int((labels == c).sum()):,})", fontsize=10)
        ax.axis("off")
    fig.suptitle("Mean image per class", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
def run_pca(flat: np.ndarray, n_components: int = 50) -> Tuple[np.ndarray, PCA]:
    """Standardise then PCA. Returns (transformed, fitted_pca)."""
    flat = flat - flat.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_components, random_state=0)
    return pca.fit_transform(flat), pca


def plot_pca_variance(pca: PCA) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    cum = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(0.9, ls="--", c="grey", lw=1, label="90% variance")
    ax.set(xlabel="principal components", ylabel="cumulative explained variance",
           title="PCA — explained variance")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_embedding(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: dict,
    method: str = "t-SNE",
) -> plt.Figure:
    """Scatter of a 2-D embedding coloured by true label."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in sorted(class_names):
        m = labels == c
        name = class_names[c]
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5,
                   label=name, color=_PALETTE.get(name))
    ax.set(title=f"{method} embedding (coloured by true label)",
           xlabel="dim 1", ylabel="dim 2")
    ax.legend(markerscale=2, fontsize=9)
    fig.tight_layout()
    return fig


def run_tsne(pca_feats: np.ndarray, seed: int = 0, perplexity: int = 30) -> np.ndarray:
    return TSNE(n_components=2, random_state=seed, perplexity=perplexity,
                init="pca", learning_rate="auto").fit_transform(pca_feats)


def run_umap(pca_feats: np.ndarray, seed: int = 0) -> Optional[np.ndarray]:
    """UMAP if available, else None (falls back to t-SNE in the notebook)."""
    try:
        import umap
    except ImportError:
        return None
    return umap.UMAP(n_components=2, random_state=seed).fit_transform(pca_feats)


# ──────────────────────────────────────────────────────────────────────────────
def kmeans_vs_labels(
    pca_feats: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """
    Cluster with k-means and measure agreement with true labels via the
    Adjusted Rand Index (1.0 = perfect, 0.0 = random).  Returns (ari, cluster_ids).
    """
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    clusters = km.fit_predict(pca_feats)
    ari = adjusted_rand_score(labels, clusters)
    return ari, clusters
