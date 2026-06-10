"""
data.py — Loading, patient-level splitting, and dataset objects
================================================================
Workflow steps (C): patient-disjoint splits, label encoding, normalization.

Two tasks:
  • binary      → isCancerous ∈ {0,1}      (labels for ALL 99 patients)
  • multiclass  → cellType    ∈ {0,1,2,3}  (labels for patients 1–60 only)

CRITICAL SPLITTING RULE
-----------------------
No patient may appear in more than one of {train, val, test}.  Cells from the
same patient/slide are highly correlated, so a random per-image split would
leak information and inflate scores.  We split by *patientID*.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Cell-type integer → human-readable name (from the CSV)
CELLTYPE_NAMES: Dict[int, str] = {
    0: "fibroblast",
    1: "inflammatory",
    2: "epithelial",
    3: "others",
}
BINARY_NAMES: Dict[int, str] = {0: "non-cancerous", 1: "cancerous"}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load & merge the label CSVs
# ──────────────────────────────────────────────────────────────────────────────
def load_labels(
    main_csv: str,
    extra_csv: Optional[str] = None,
    image_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return a single dataframe with columns:
        InstanceID, patientID, ImageName, path, cellType, cellTypeName,
        isCancerous, source

    `cellType`/`cellTypeName` are NaN for extra-data rows (no cell-type label).
    If `image_dir` is given, `path` is filled and rows whose image file is
    missing are dropped (with a warning count).
    """
    main = pd.read_csv(main_csv)
    main["source"] = "main"

    frames = [main]
    if extra_csv and os.path.exists(extra_csv):
        extra = pd.read_csv(extra_csv)
        extra["source"] = "extra"
        # extra has no cellType / cellTypeName — add as NaN so concat aligns
        for col in ("cellType", "cellTypeName"):
            if col not in extra.columns:
                extra[col] = np.nan
        frames.append(extra)

    df = pd.concat(frames, ignore_index=True, sort=False)

    # Resolve image paths
    if image_dir is not None:
        df["path"] = df["ImageName"].apply(lambda n: str(Path(image_dir) / n))
        exists = df["path"].apply(os.path.exists)
        missing = int((~exists).sum())
        if missing:
            print(f"⚠️  {missing} image files referenced in CSV are missing — dropping them.")
        df = df[exists].reset_index(drop=True)
    else:
        df["path"] = np.nan

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Patient-level train / val / test split
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    patient_assignment: Dict[int, str] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, d in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append({
                "split": name,
                "patients": d["patientID"].nunique(),
                "cells": len(d),
                "cancerous": int(d["isCancerous"].sum()),
                "frac_cancerous": round(d["isCancerous"].mean(), 3),
            })
        return pd.DataFrame(rows)


def patient_level_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
    seed: int = 42,
    stratify_source: bool = True,
) -> SplitResult:
    """
    Assign whole patients to train/val/test.

    We stratify the *patient* assignment by data source (main vs extra) so that
    the labelled main-data patients are spread proportionally across splits —
    this guarantees the val/test sets contain cell-type-labelled patients for
    the multiclass task, not only binary-only patients.
    """
    rng = np.random.default_rng(seed)
    assignment: Dict[int, str] = {}

    groups = (
        df.groupby("source")["patientID"].unique().to_dict()
        if stratify_source and "source" in df.columns
        else {"all": df["patientID"].unique()}
    )

    for _, patients in groups.items():
        patients = np.array(sorted(patients))
        rng.shuffle(patients)
        n = len(patients)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        for p in patients[:n_test]:
            assignment[int(p)] = "test"
        for p in patients[n_test:n_test + n_val]:
            assignment[int(p)] = "val"
        for p in patients[n_test + n_val:]:
            assignment[int(p)] = "train"

    split_col = df["patientID"].map(assignment)
    return SplitResult(
        train=df[split_col == "train"].reset_index(drop=True),
        val=df[split_col == "val"].reset_index(drop=True),
        test=df[split_col == "test"].reset_index(drop=True),
        patient_assignment=assignment,
    )


def multiclass_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a cell-type label (main data only)."""
    return df[df["cellType"].notna()].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Image loading helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_images_as_array(
    df: pd.DataFrame,
    img_size: int = 27,
    dtype=np.float32,
) -> np.ndarray:
    """
    Load all images referenced by `df['path']` into an (N, H, W, 3) array
    scaled to [0, 1].  Used by the classical (flat) models and EDA.
    """
    from PIL import Image

    n = len(df)
    out = np.empty((n, img_size, img_size, 3), dtype=dtype)
    for i, p in enumerate(df["path"].values):
        img = Image.open(p).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))
        out[i] = np.asarray(img, dtype=dtype) / 255.0
    return out


def compute_norm_stats(images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean & std from an (N,H,W,3) array in [0,1]."""
    mean = images.reshape(-1, 3).mean(axis=0)
    std = images.reshape(-1, 3).std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def flatten_images(images: np.ndarray) -> np.ndarray:
    """(N,H,W,3) → (N, H*W*3) for sklearn models."""
    return images.reshape(images.shape[0], -1)
