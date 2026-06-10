"""
classical.py — Flat-feature models (workflow step D)
=====================================================
Logistic Regression, SVM (RBF), Random Forest, Gradient Boosting (XGBoost),
and an MLP — all trained on flattened, standardised pixel vectors.

Each builder returns an unfitted sklearn-style estimator wrapped in a
StandardScaler pipeline.  `tune_on_validation` does a small manual grid
search using the *validation* split (not cross-val on train) to match the
workflow's explicit train/val/test protocol and avoid leaking the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_XGB = False


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


# ──────────────────────────────────────────────────────────────────────────────
# Model builders with their validation grids
# ──────────────────────────────────────────────────────────────────────────────
def build_logreg(C: float = 1.0) -> Pipeline:
    # sklearn ≥1.7 infers multinomial vs OvR automatically (multi_class arg removed)
    return _pipe(LogisticRegression(C=C, max_iter=2000,
                                    class_weight="balanced"))


def build_svm(C: float = 1.0, gamma="scale") -> Pipeline:
    return _pipe(SVC(C=C, gamma=gamma, kernel="rbf", probability=True,
                     class_weight="balanced", random_state=0))


def build_rf(n_estimators: int = 300, max_depth=None) -> Pipeline:
    # trees don't need scaling, but pipeline keeps the interface uniform
    return _pipe(RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", n_jobs=-1, random_state=0))


def build_gboost(learning_rate: float = 0.1, max_depth: int = 5) -> Pipeline:
    if _HAS_XGB:
        clf = XGBClassifier(
            n_estimators=400, learning_rate=learning_rate, max_depth=max_depth,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            tree_method="hist", n_jobs=-1, random_state=0)
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_depth=max_depth, random_state=0)
    return _pipe(clf)


def build_mlp(hidden=(256, 128), alpha: float = 1e-3) -> Pipeline:
    return _pipe(MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=300,
        early_stopping=True, n_iter_no_change=12, random_state=0))


# Grid definitions: name → (builder, list-of-kwargs)
MODEL_GRIDS = {
    "LogReg": (build_logreg, [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]),
    "SVM-RBF": (build_svm, [{"C": c} for c in (1.0, 10.0)]),
    "RandomForest": (build_rf, [{"max_depth": d} for d in (None, 12, 20)]),
    "GradBoost": (build_gboost, [{"max_depth": d} for d in (3, 5)]),
    "MLP": (build_mlp, [{"hidden": h} for h in ((256, 128), (512, 256, 128))]),
}


@dataclass
class TuneResult:
    name: str
    best_params: dict
    best_val_f1: float
    estimator: Pipeline           # fitted on train with best params
    val_f1_by_params: List[Tuple[dict, float]]


def tune_on_validation(
    name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> TuneResult:
    """Fit each grid point on train, score macro-F1 on val, keep the best."""
    builder, grid = MODEL_GRIDS[name]
    results: List[Tuple[dict, float]] = []
    best = None
    for params in grid:
        est = builder(**params)
        est.fit(X_train, y_train)
        f1 = f1_score(y_val, est.predict(X_val), average="macro")
        results.append((params, f1))
        if best is None or f1 > best[1]:
            best = (params, f1, est)
    return TuneResult(
        name=name, best_params=best[0], best_val_f1=best[1],
        estimator=best[2], val_f1_by_params=results,
    )


def tune_all(
    X_train, y_train, X_val, y_val,
    models: List[str] | None = None,
) -> Dict[str, TuneResult]:
    """Tune every classical model; returns name → TuneResult."""
    models = models or list(MODEL_GRIDS)
    out: Dict[str, TuneResult] = {}
    for name in models:
        print(f"  • tuning {name} …", flush=True)
        out[name] = tune_on_validation(name, X_train, y_train, X_val, y_val)
        print(f"    best {out[name].best_params} → val macroF1={out[name].best_val_f1:.4f}")
    return out
