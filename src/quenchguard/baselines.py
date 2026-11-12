"""Non-neural baselines the LSTM has to beat.

A deep model on this problem is only defensible if it beats cheap models trained on
the *same* features. We therefore train:

* ``logreg``  – class-weighted logistic regression (fully linear, interpretable)
* ``hist_gbdt`` – histogram gradient boosting (strong tabular learner)

Both see the **last window only** (a single snapshot), while the LSTM additionally
sees the 12 h history — so if the LSTM wins, it wins because of temporal context.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .config import Config, default_config


def last_step(X: np.ndarray) -> np.ndarray:
    """(n, seq_len, f) -> (n, f): the most recent window's features."""
    return X[:, -1, :]


def train_baselines(cfg: Config | None = None, X_train: np.ndarray | None = None,
                    y_train: np.ndarray | None = None,
                    X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                    verbose: bool = True) -> dict:
    """Fit the two baselines; returns {name: fitted model}."""
    cfg = cfg or default_config()
    Xt, yt = last_step(X_train), y_train
    Xv = last_step(X_val)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1,
                                solver="lbfgs", random_state=cfg.train.seed)
    logreg.fit(Xt, yt)

    gbdt = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=None,
        class_weight="balanced", random_state=cfg.train.seed,
    )
    gbdt.fit(Xt, yt)

    if verbose:
        print(f"baselines fitted on {len(yt):,} training sequences "
              f"(last-window features only)")
    return {"logreg": logreg, "hist_gbdt": gbdt}


def baseline_scores(models: dict, X: np.ndarray) -> dict[str, np.ndarray]:
    """Probability scores for a split, one array per baseline."""
    Xs = last_step(X)
    out = {}
    for name, model in models.items():
        proba = model.predict_proba(Xs)
        out[name] = proba[:, 1]
    return out
