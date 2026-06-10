# Colon Histopathology Cell Classification

A complete machine-learning workflow for two tasks on 27×27 colon-cell patches:

- **Binary** — `isCancerous` (labels for all patients)
- **Multiclass** — cell type: *fibroblast · inflammatory · epithelial · others* (patients 1–60)

The project follows the assignment workflow **(A)–(H)** end-to-end, from
unsupervised EDA to a final, test-set-validated model with an ethics discussion.

---

## Project layout

```
crc-cells/
├── crc_classification.ipynb   ← MAIN deliverable: runs the whole A–H workflow
├── build_notebook.py          ← regenerates the notebook from source
├── src/
│   ├── data.py        (C) loading, patient-level splits, normalisation
│   ├── eda.py         (B) PCA, t-SNE, UMAP, k-means, distributions
│   ├── classical.py   (D) LogReg, SVM, RandomForest, XGBoost, MLP + val tuning
│   ├── cnn.py         (D,E) SmallCNN, TransferCNN(ResNet18), training (MPS-ready)
│   └── evaluate.py    (E,F,G) metrics, confusion matrices, comparison, ensembling
├── data/
│   ├── data_labels_mainData.csv
│   ├── data_labels_extraData.csv
│   └── images/        ← put all the .png cell patches here
├── requirements.txt
└── README.md
```

## How each workflow step maps to the code

| Step | What happens | Where |
|------|--------------|-------|
| A | Task definition | notebook §A |
| B | EDA: class balance, mean images, PCA, t-SNE/UMAP, k-means ARI | `eda.py` |
| C | Patient-level 60/20/20 split, label encoding, normalisation | `data.py` |
| D | 5 classical models (val-tuned) + a from-scratch CNN | `classical.py`, `cnn.py` |
| E | Augmentation · transfer learning (ResNet18) · soft-vote ensemble | `cnn.py`, `evaluate.py` |
| F | Unified comparison table + bar chart (macro-F1) | `evaluate.py` |
| G | Pick best on val → evaluate **once** on test; confusion matrices | notebook §G |
| H | Limitations & ethics | notebook §H |

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # see note on torch below
```

**PyTorch on Apple Silicon (your MacBook):** the standard pip wheel already
includes the **MPS** GPU backend. No CUDA, no extra flags — the notebook prints
`Device for CNNs: mps` and trains on the GPU automatically. (If you ever see
`cpu`, update macOS / PyTorch; MPS needs macOS 12.3+.)

---

## Run

1. Drop the cell images into `data/images/` (filenames must match the
   `ImageName` column, e.g. `22405.png`).
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook crc_classification.ipynb
   ```
3. Edit the three path constants in the first code cell if needed, then
   **Run All**. Every seed is fixed, so results reproduce.

The last two cells print the headline numbers for your report:

```
Binary     (...)  macro-F1 = 0.9x   target ≥0.90
Multiclass (...)  macro-F1 = 0.6x   target ≥0.60
```

---

## Design notes worth mentioning in the report

- **Patient-level splitting** prevents the most common leakage error in
  histopathology — cells from one slide are correlated, so a random per-image
  split would inflate every score. We split whole patients.
- **`isCancerous` ≡ epithelial** in the labelled data (perfect cross-tab). The
  notebook surfaces this in EDA and discusses the generalisation risk in ethics.
- **Macro-F1 everywhere** because `others`/`fibroblast` are minority classes;
  plain accuracy would hide poor minority recall.
- **Graceful degradation:** if `xgboost` or `umap-learn` aren't installed, the
  pipeline substitutes sklearn's `HistGradientBoosting` / skips UMAP rather than
  crashing.

## Regenerating the notebook

The notebook is generated from `build_notebook.py` so the narrative and code
stay version-controllable:

```bash
python build_notebook.py
```
