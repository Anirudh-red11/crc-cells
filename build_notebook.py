"""Builds crc_classification.ipynb following the assignment workflow A–H."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# ── Title ─────────────────────────────────────────────────────────────────────
md("""# Colon Histopathology Cell Classification
### A complete ML workflow — binary (`isCancerous`) and multiclass (cell type)

This notebook follows the assignment workflow **(A) → (H)**:

| Step | Section |
|------|---------|
| A | Task definition |
| B | Data description & EDA (unsupervised) |
| C | Preprocessing (patient-level splits) |
| D | Model development (LR, SVM, RF, XGBoost, MLP, CNN) |
| E | Advanced techniques (augmentation · transfer learning · ensembling) |
| F | Model comparison |
| G | Final model selection + test evaluation |
| H | Limitations & ethics |

All heavy lifting lives in the `src/` modules so each cell stays readable.

> **Before running:** put the cell images in `data/images/` and the two label
> CSVs in `data/`. Set `IMAGE_DIR` below to match. On Apple Silicon the CNNs
> automatically use the **MPS** GPU backend.""")

code("""import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data, eda, classical, cnn, evaluate

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── EDIT THESE PATHS ──────────────────────────────────────────────
MAIN_CSV  = "data/data_labels_mainData.csv"
EXTRA_CSV = "data/data_labels_extraData.csv"
IMAGE_DIR = "data/images"        # folder containing all the .png patches
# ──────────────────────────────────────────────────────────────────

print("Device for CNNs:", cnn.get_device())""")

# ── A ──────────────────────────────────────────────────────────────────────────
md("""## (A) Task Definition

Two parallel image-classification problems on 27×27 RGB colon-cell patches:

1. **Binary** — predict `isCancerous` ∈ {0, 1}. Labels exist for **all** patients.
2. **Multiclass** — predict cell type ∈ {fibroblast, inflammatory, epithelial, others}.
   Labels exist **only** for the main-data patients (1–60).

The two tasks share the same images and the same patient-level split protocol,
so their results are directly comparable.""")

# ── B ──────────────────────────────────────────────────────────────────────────
md("""## (B) Data Description & EDA

We first load the labels and inspect class balance, then explore the *image*
structure with unsupervised methods (PCA, t-SNE/UMAP, k-means) — no labels used
in the projection, only afterwards for colouring.""")

code("""df = data.load_labels(MAIN_CSV, EXTRA_CSV, image_dir=IMAGE_DIR)

print(f"Total cells       : {len(df):,}")
print(f"Unique patients   : {df.patientID.nunique()}")
print(f"With cell-type lbl : {df.cellType.notna().sum():,}  (main data)")
print(f"Binary-only        : {df.cellType.isna().sum():,}  (extra data)")
df.head()""")

code("""# Class distributions for both tasks
fig = eda.plot_class_distributions(df)
plt.show()""")

md("""**Key structural finding.** In the labelled main data, every `epithelial`
cell is cancerous and every non-epithelial cell is not — the binary label is
*perfectly* determined by cell type. The cross-tab below confirms this. This has
two consequences we revisit in (H): (i) on labelled data the binary task is
effectively "is this epithelial?"; (ii) the **extra** patients contain cancerous
cells whose type we never observe, so the binary model must generalise beyond the
clean main-data correlation.""")

code("""print(pd.crosstab(df.cellTypeName, df.isCancerous, margins=True))""")

code("""# Mean image per cell type — quick look at colour/texture differences
mc = data.multiclass_subset(df)
mc_imgs = data.load_images_as_array(mc)
fig = eda.plot_mean_images(mc_imgs, mc.cellType.astype(int).values,
                           data.CELLTYPE_NAMES)
plt.show()""")

md("""### Unsupervised structure — PCA, t-SNE / UMAP, clustering

We flatten each image to a 2 187-d vector, reduce with PCA, then embed to 2-D.
k-means on the PCA features is compared to the true labels via the **Adjusted
Rand Index** (ARI: 1 = perfect agreement, 0 = random).""")

code("""flat = data.flatten_images(mc_imgs)
pca_feats, pca = eda.run_pca(flat, n_components=50)

fig = eda.plot_pca_variance(pca); plt.show()
print(f"50 components capture {pca.explained_variance_ratio_.sum():.1%} of variance")""")

code("""# t-SNE (always available)
tsne_xy = eda.run_tsne(pca_feats, seed=RANDOM_SEED)
fig = eda.plot_embedding(tsne_xy, mc.cellType.astype(int).values,
                         data.CELLTYPE_NAMES, method="t-SNE")
plt.show()

# UMAP if the package is installed, else skip gracefully
umap_xy = eda.run_umap(pca_feats, seed=RANDOM_SEED)
if umap_xy is not None:
    fig = eda.plot_embedding(umap_xy, mc.cellType.astype(int).values,
                             data.CELLTYPE_NAMES, method="UMAP")
    plt.show()
else:
    print("UMAP not installed — install `umap-learn` to see the UMAP embedding.")""")

code("""ari, _ = eda.kmeans_vs_labels(pca_feats, mc.cellType.astype(int).values, n_clusters=4)
print(f"k-means (k=4) vs true cell type — Adjusted Rand Index = {ari:.3f}")
print("Low ARI ⇒ raw pixels alone don't cleanly separate the classes — "
      "motivating supervised models, especially the CNN.")""")

# ── C ──────────────────────────────────────────────────────────────────────────
md("""## (C) Preprocessing

**Patient-level split** is the single most important methodological choice here:
cells from one patient/slide are correlated, so a random per-image split leaks
information and inflates scores. We assign *whole patients* to train/val/test
(≈60/20/20), stratified by data source so labelled patients land in every split.""")

code("""# Binary task uses ALL patients; multiclass uses only the labelled subset.
split_bin = data.patient_level_split(df, val_frac=0.2, test_frac=0.2, seed=RANDOM_SEED)
split_mc  = data.patient_level_split(data.multiclass_subset(df),
                                     val_frac=0.2, test_frac=0.2, seed=RANDOM_SEED)

print("BINARY split (all patients):")
print(split_bin.summary().to_string(index=False))
print("\\nMULTICLASS split (labelled patients):")
print(split_mc.summary().to_string(index=False))

# Sanity check: no patient leaks across splits
for s in (split_bin, split_mc):
    tr, va, te = set(s.train.patientID), set(s.val.patientID), set(s.test.patientID)
    assert not (tr & va) and not (tr & te) and not (va & te)
print("\\n✓ No patient appears in more than one split.")""")

code("""# Load images once per split, scale to [0,1], compute train normalisation stats.
def load_split(split):
    return {k: data.load_images_as_array(getattr(split, k))
            for k in ("train", "val", "test")}

imgs_mc = load_split(split_mc)
mean, std = data.compute_norm_stats(imgs_mc["train"])
print("Train per-channel mean:", mean.round(3), " std:", std.round(3))

# Flattened + standardised features for the classical models
flat_mc = {k: data.flatten_images(v) for k, v in imgs_mc.items()}
y_mc = {k: getattr(split_mc, k).cellType.astype(int).values
        for k in ("train", "val", "test")}
y_bin_mc = {k: getattr(split_mc, k).isCancerous.values
            for k in ("train", "val", "test")}""")

# ── D ──────────────────────────────────────────────────────────────────────────
md("""## (D) Model Development

We train six model families. Classical models (LR, SVM, RF, XGBoost, MLP) use
flattened pixels and are tuned on the **validation** split. The CNN uses the raw
image tensor. We report macro-F1 throughout because the classes are imbalanced.

### D.1 Classical models — multiclass (cell type)""")

code("""results_mc = classical.tune_all(flat_mc["train"], y_mc["train"],
                                flat_mc["val"],   y_mc["val"])""")

code("""val_metrics_mc = {
    name: evaluate.compute_metrics(y_mc["val"], r.estimator.predict(flat_mc["val"]))
    for name, r in results_mc.items()
}
table_mc = evaluate.comparison_table(val_metrics_mc)
display(table_mc)
fig = evaluate.plot_comparison(table_mc, "val_f1",
                               "Classical models — multiclass (val macro-F1)")
plt.show()""")

md("""### D.2 CNN — multiclass (from scratch)

A compact CNN sized for 27×27 inputs. We pass inverse-frequency class weights to
the loss to counter imbalance, and early-stop on validation macro-F1.""")

code("""cw_mc = cnn.class_weights_from_labels(y_mc["train"], num_classes=4)

train_ds = cnn.CellImageDataset(split_mc.train, "cellType", augment=False,
                                mean=tuple(mean), std=tuple(std))
val_ds   = cnn.CellImageDataset(split_mc.val,   "cellType",
                                mean=tuple(mean), std=tuple(std))

cnn_mc, hist_mc = cnn.train_model(
    cnn.SmallCNN(num_classes=4),
    cnn.make_loader(train_ds, 64, shuffle=True),
    cnn.make_loader(val_ds,   64),
    num_classes=4, epochs=40, lr=1e-3, class_weights=cw_mc,
)
fig = evaluate.plot_history(hist_mc, "SmallCNN (multiclass) — training history")
plt.show()""")

# ── E ──────────────────────────────────────────────────────────────────────────
md("""## (E) Advanced Techniques

We apply **all three** techniques the workflow lists and measure each one's
effect on validation macro-F1.

### E.1 Data augmentation
Re-train the same CNN with random flips / rotations / colour jitter on the
training set only.""")

code("""train_ds_aug = cnn.CellImageDataset(split_mc.train, "cellType", augment=True,
                                    mean=tuple(mean), std=tuple(std))
cnn_mc_aug, hist_mc_aug = cnn.train_model(
    cnn.SmallCNN(num_classes=4),
    cnn.make_loader(train_ds_aug, 64, shuffle=True),
    cnn.make_loader(val_ds, 64),
    num_classes=4, epochs=40, lr=1e-3, class_weights=cw_mc,
)
print(f"val macro-F1  no-aug = {max(hist_mc.val_f1):.4f}  |  "
      f"with-aug = {max(hist_mc_aug.val_f1):.4f}")""")

md("""### E.2 Transfer learning
A ResNet-18 pretrained on ImageNet, with the patches upscaled to 64×64. Generic
edge/texture filters transfer well and usually converge faster on small data.""")

code("""train_ds_T = cnn.CellImageDataset(split_mc.train, "cellType", augment=True,
                                  upscale=64, mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
val_ds_T   = cnn.CellImageDataset(split_mc.val, "cellType", upscale=64,
                                  mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
cnn_transfer, hist_T = cnn.train_model(
    cnn.TransferCNN(num_classes=4, freeze_backbone=False),
    cnn.make_loader(train_ds_T, 64, shuffle=True),
    cnn.make_loader(val_ds_T, 64),
    num_classes=4, epochs=25, lr=3e-4, class_weights=cw_mc,
)
print(f"Transfer-learning val macro-F1 = {max(hist_T.val_f1):.4f}")""")

md("""### E.3 Ensembling
Average the class probabilities of the strongest models (soft voting). Ensembles
usually trade a little complexity for robustness.""")

code("""# Collect probabilities on the validation set from several models
prob_sources = {}

# best classical model
best_classical = max(results_mc.values(), key=lambda r: r.best_val_f1)
prob_sources[best_classical.name] = best_classical.estimator.predict_proba(flat_mc["val"])

# augmented CNN
p_cnn, _ = cnn.predict_proba(cnn_mc_aug, cnn.make_loader(val_ds, 64))
prob_sources["SmallCNN+aug"] = p_cnn

# transfer CNN
p_T, _ = cnn.predict_proba(cnn_transfer, cnn.make_loader(val_ds_T, 64))
prob_sources["TransferCNN"] = p_T

ens_pred = evaluate.soft_vote(list(prob_sources.values()))
ens_f1 = evaluate.compute_metrics(y_mc["val"], ens_pred)["f1_macro"]
print("Ensemble members:", list(prob_sources))
print(f"Ensemble val macro-F1 = {ens_f1:.4f}")""")

# ── F ──────────────────────────────────────────────────────────────────────────
md("""## (F) Model Comparison

All models, same split, same metric (validation macro-F1). This is where we
check whether the extra complexity of CNNs / ensembles actually pays off versus
the simple baselines — the bias–variance trade-off in practice.""")

code("""summary = dict(val_metrics_mc)  # classical models
summary["SmallCNN"]      = evaluate.compute_metrics(
    y_mc["val"], cnn.predict_proba(cnn_mc,      cnn.make_loader(val_ds, 64))[0].argmax(1))
summary["SmallCNN+aug"]  = evaluate.compute_metrics(
    y_mc["val"], p_cnn.argmax(1))
summary["TransferCNN"]   = evaluate.compute_metrics(
    y_mc["val"], p_T.argmax(1))
summary["Ensemble"]      = evaluate.compute_metrics(y_mc["val"], ens_pred)

table_all = evaluate.comparison_table(summary)
display(table_all)
fig = evaluate.plot_comparison(table_all, "val_f1",
                               "All models — multiclass (val macro-F1)")
plt.show()""")

# ── G ──────────────────────────────────────────────────────────────────────────
md("""## (G) Final Model Selection & Test Evaluation

We pick the model with the best **validation** macro-F1, then evaluate it **once**
on the held-out test patients — the only time the test set is touched. Targets:
macro-F1 ≥ 0.90 on `isCancerous`, ≥ 0.60 on cell type.""")

code("""best_name = table_all.iloc[0]["model"]
print(f"Selected multiclass model: {best_name}")

# Resolve the chosen model to a test-set prediction
def predict_test_multiclass(name):
    if name in results_mc:                       # classical
        return results_mc[name].estimator.predict(flat_mc["test"])
    if name == "SmallCNN":
        ds = cnn.CellImageDataset(split_mc.test, "cellType", mean=tuple(mean), std=tuple(std))
        return cnn.predict_proba(cnn_mc, cnn.make_loader(ds, 64))[0].argmax(1)
    if name == "SmallCNN+aug":
        ds = cnn.CellImageDataset(split_mc.test, "cellType", mean=tuple(mean), std=tuple(std))
        return cnn.predict_proba(cnn_mc_aug, cnn.make_loader(ds, 64))[0].argmax(1)
    if name == "TransferCNN":
        ds = cnn.CellImageDataset(split_mc.test, "cellType", upscale=64,
                                  mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        return cnn.predict_proba(cnn_transfer, cnn.make_loader(ds, 64))[0].argmax(1)
    if name == "Ensemble":
        dsc = cnn.CellImageDataset(split_mc.test, "cellType", mean=tuple(mean), std=tuple(std))
        dsT = cnn.CellImageDataset(split_mc.test, "cellType", upscale=64,
                                   mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
        probs = [best_classical.estimator.predict_proba(flat_mc["test"]),
                 cnn.predict_proba(cnn_mc_aug, cnn.make_loader(dsc, 64))[0],
                 cnn.predict_proba(cnn_transfer, cnn.make_loader(dsT, 64))[0]]
        return evaluate.soft_vote(probs)
    raise ValueError(name)

test_pred_mc = predict_test_multiclass(best_name)
test_metrics_mc = evaluate.compute_metrics(y_mc["test"], test_pred_mc)
print("Multiclass TEST metrics:", {k: round(v,4) for k,v in test_metrics_mc.items()})

fig = evaluate.plot_confusion(y_mc["test"], test_pred_mc,
                              [data.CELLTYPE_NAMES[i] for i in range(4)],
                              f"{best_name} — multiclass test confusion")
plt.show()""")

md("""### Binary task — train, select, and test on the full patient cohort
We repeat the protocol for `isCancerous` using all patients (the extra data
enlarges the binary training set).""")

code("""# Load binary-split images (all patients). This is the larger load.
imgs_bin = {k: data.load_images_as_array(getattr(split_bin, k))
            for k in ("train", "val", "test")}
flat_bin = {k: data.flatten_images(v) for k, v in imgs_bin.items()}
y_bin = {k: getattr(split_bin, k).isCancerous.values for k in ("train","val","test")}
mean_b, std_b = data.compute_norm_stats(imgs_bin["train"])

# Classical sweep
results_bin = classical.tune_all(flat_bin["train"], y_bin["train"],
                                 flat_bin["val"], y_bin["val"])

# CNN with augmentation (usually the strongest single binary model)
cw_bin = cnn.class_weights_from_labels(y_bin["train"], num_classes=2)
tb = cnn.CellImageDataset(split_bin.train, "isCancerous", augment=True,
                          mean=tuple(mean_b), std=tuple(std_b))
vb = cnn.CellImageDataset(split_bin.val, "isCancerous",
                          mean=tuple(mean_b), std=tuple(std_b))
cnn_bin, hist_bin = cnn.train_model(
    cnn.SmallCNN(num_classes=2), cnn.make_loader(tb, 64, shuffle=True),
    cnn.make_loader(vb, 64), num_classes=2, epochs=40, lr=1e-3, class_weights=cw_bin)

# Compare on val
val_bin = {n: evaluate.compute_metrics(y_bin["val"], r.estimator.predict(flat_bin["val"]))
           for n, r in results_bin.items()}
p_cnn_bin, _ = cnn.predict_proba(cnn_bin, cnn.make_loader(vb, 64))
val_bin["SmallCNN+aug"] = evaluate.compute_metrics(y_bin["val"], p_cnn_bin.argmax(1))
table_bin = evaluate.comparison_table(val_bin)
display(table_bin)""")

code("""# Select best binary model and evaluate on test once
best_bin = table_bin.iloc[0]["model"]
print("Selected binary model:", best_bin)
if best_bin in results_bin:
    test_pred_bin = results_bin[best_bin].estimator.predict(flat_bin["test"])
else:
    tb_test = cnn.CellImageDataset(split_bin.test, "isCancerous",
                                   mean=tuple(mean_b), std=tuple(std_b))
    test_pred_bin = cnn.predict_proba(cnn_bin, cnn.make_loader(tb_test, 64))[0].argmax(1)

test_metrics_bin = evaluate.compute_metrics(y_bin["test"], test_pred_bin)
print("Binary TEST metrics:", {k: round(v,4) for k,v in test_metrics_bin.items()})
fig = evaluate.plot_confusion(y_bin["test"], test_pred_bin,
                              ["non-cancerous", "cancerous"],
                              f"{best_bin} — binary test confusion")
plt.show()""")

code("""print("="*52)
print("FINAL TEST RESULTS")
print("="*52)
print(f"  Binary     ({best_bin:<14}) macro-F1 = {test_metrics_bin['f1_macro']:.4f}   target ≥0.90")
print(f"  Multiclass ({best_name:<14}) macro-F1 = {test_metrics_mc['f1_macro']:.4f}   target ≥0.60")""")

# ── H ──────────────────────────────────────────────────────────────────────────
md("""## (H) Limitations & Ethics

**Limitations**
- **Tiny context (27×27).** Each patch shows one cell with almost no tissue
  context, which caps achievable accuracy and makes models sensitive to staining
  artifacts.
- **Label structure.** In the labelled data `isCancerous` is collinear with the
  *epithelial* class, so a binary model can score well by simply detecting
  epithelial morphology — which may not transfer to cancerous cells of other
  appearances present in the unlabeled extra cohort.
- **Class imbalance.** `others` and `fibroblast` are minority classes; macro-F1
  (used throughout) prevents the majority classes from masking poor minority
  performance, but minority recall remains the weak point (see confusion matrices).
- **Patient correlation.** We mitigated leakage with patient-level splits, but
  with only ~12 test patients the test estimate still has meaningful variance.

**Ethics**
- **False negatives** in cancer detection are the most dangerous error; recall on
  the cancerous class deserves more scrutiny than headline accuracy.
- **Interpretability.** Deep models are opaque; any clinical use needs
  explainability (e.g. Grad-CAM) and prospective validation.
- **Role.** This is a decision-support tool to assist pathologists, not replace
  them — outputs should be reported with calibrated uncertainty and monitored
  over time. Images here are de-identified patches; fairness across patient
  subgroups should be checked if demographic metadata becomes available.""")

md("""---
### Reproducing the report
- Re-run top-to-bottom; every random seed is fixed via `RANDOM_SEED`.
- Swap `IMAGE_DIR` for your local path.
- The final two cells print the headline test metrics for the write-up.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
nbf.write(nb, "crc_classification.ipynb")
print("✅ Notebook written with", len(cells), "cells")
