"""Per-instrument direction model: P(SELL), P(HOLD), P(BUY) over the next `horizon` days.

Training protocol
  1. time-ordered split; the last `test_ratio` of labelled rows is the hold-out
  2. training rows whose label window reaches into the hold-out are purged
  3. the model is scored on the hold-out against two baselines
     (always predict the majority class; always predict the class frequencies)
  4. `has_edge` is True only if it beats both on unseen data
  5. the production model is then refit on all labelled rows

The ML agent abstains whenever `has_edge` is False.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from app.ml.features import FEATURES, compute_features, make_labels

CLASSES = ["SELL", "HOLD", "BUY"]
MIN_TRAIN_ROWS = 300


@dataclass
class ModelMetrics:
    train_rows: int
    test_rows: int
    test_start: str
    accuracy: float
    majority_baseline: float
    log_loss: float
    prior_log_loss: float
    has_edge: bool
    test_label_share: dict[str, float]


@dataclass
class ModelBundle:
    instrument: str
    horizon: int
    features: list[str]
    trained_at: str
    last_bar: str
    metrics: ModelMetrics
    model: HistGradientBoostingClassifier = field(repr=False)
    holdout_model: HistGradientBoostingClassifier | None = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "instrument": self.instrument,
            "horizon_days": self.horizon,
            "trained_at": self.trained_at,
            "last_bar": self.last_bar,
            "metrics": asdict(self.metrics),
        }


@dataclass
class MLOutput:
    instrument: str
    as_of: str
    probabilities: dict[str, float]
    predicted: str
    confidence: float
    has_edge: bool
    metrics: dict


def _new_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
        min_samples_leaf=40, random_state=seed,
    )


def train_model(df: pd.DataFrame, instrument: str, horizon: int = 5, test_ratio: float = 0.25,
                seed: int = 7) -> ModelBundle:
    X = compute_features(df)
    y = make_labels(df, horizon)
    rows = X.notna().all(axis=1) & y.notna()
    Xl, yl = X[rows], y[rows].astype(int)
    n = len(Xl)
    split = int(n * (1 - test_ratio))
    if split - horizon < MIN_TRAIN_ROWS:
        raise ValueError(f"{instrument}: need more history ({n} labelled rows, want {MIN_TRAIN_ROWS + horizon} to train)")

    X_tr, y_tr = Xl.iloc[: split - horizon], yl.iloc[: split - horizon]  # purge overlap
    X_te, y_te = Xl.iloc[split:], yl.iloc[split:]
    if sorted(y_tr.unique()) != [0, 1, 2]:
        raise ValueError(f"{instrument}: training labels must contain SELL, HOLD and BUY")

    holdout = _new_model(seed).fit(X_tr, y_tr)
    proba = holdout.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    prior = np.bincount(y_tr, minlength=3) / len(y_tr)
    majority = int(prior.argmax())
    acc = float((pred == y_te.to_numpy()).mean())
    base_acc = float((y_te.to_numpy() == majority).mean())
    ll = float(log_loss(y_te, proba, labels=[0, 1, 2]))
    prior_ll = float(log_loss(y_te, np.tile(prior, (len(y_te), 1)), labels=[0, 1, 2]))

    metrics = ModelMetrics(
        train_rows=len(X_tr),
        test_rows=len(X_te),
        test_start=str(X_te.index[0].date()),
        accuracy=round(acc, 4),
        majority_baseline=round(base_acc, 4),
        log_loss=round(ll, 4),
        prior_log_loss=round(prior_ll, 4),
        has_edge=bool(acc > base_acc + 0.01 and ll < prior_ll),
        test_label_share={CLASSES[k]: round(float((y_te == k).mean()), 4) for k in range(3)},
    )
    final = _new_model(seed).fit(Xl, yl)
    return ModelBundle(
        instrument=instrument,
        horizon=horizon,
        features=list(FEATURES),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        last_bar=str(df.index[-1].date()),
        metrics=metrics,
        model=final,
        holdout_model=holdout,
    )


def predict_row(bundle: ModelBundle, features_row: pd.Series, use_holdout: bool = False) -> dict[str, float]:
    model = bundle.holdout_model if use_holdout and bundle.holdout_model is not None else bundle.model
    x = features_row[bundle.features].to_frame().T.astype(float)
    p = model.predict_proba(x)[0]
    return {CLASSES[k]: float(p[k]) for k in range(3)}


def predict_latest(bundle: ModelBundle, df: pd.DataFrame) -> MLOutput:
    feats = compute_features(df).dropna()
    if feats.empty:
        raise ValueError(f"{bundle.instrument}: not enough history to compute features")
    probs = predict_row(bundle, feats.iloc[-1])
    predicted = max(probs, key=probs.get)
    return MLOutput(
        instrument=bundle.instrument,
        as_of=str(feats.index[-1].date()),
        probabilities={k: round(v, 4) for k, v in probs.items()},
        predicted=predicted,
        confidence=round(probs[predicted], 4),
        has_edge=bundle.metrics.has_edge,
        metrics=asdict(bundle.metrics),
    )


def save_bundle(bundle: ModelBundle, model_dir: str) -> Path:
    path = Path(model_dir) / f"{bundle.instrument}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(instrument: str, model_dir: str) -> ModelBundle | None:
    path = Path(model_dir) / f"{instrument}.joblib"
    return joblib.load(path) if path.exists() else None
