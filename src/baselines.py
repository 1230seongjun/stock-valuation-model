"""
Non-ML baselines for comparison against the Ridge/XGBoost/RandomForest
models (technical-spec.md section 9's "Baseline 4종 비교": Zero-return /
Sector-average / Rule-based score lookup / regression).

This answers a DIFFERENT question than factor_validation.py does. That
module asks "is this indicator's own relationship with returns
statistically significant" (answer so far, even on 66 tickers / 16 years:
no, not after FDR correction). This module asks "does the regression model
actually beat naive baselines on held-out Test data" — which is the
practical bar for whether the model is worth using for anything, regardless
of what the significance tests said. A model can fail the first question
(no single indicator is provably non-random) and still pass the second
(the combination still reduces prediction error versus guessing) — that
would be a legitimate reason to use it. It can also fail both, which would
be a legitimate reason not to.

All baselines fit ONLY on Train rows (same rule as pipeline.time_split: no
leakage from Val/Test into what a baseline "knows"), then predict on
whatever rows they're asked to.

IMPORTANT (added 2026-09-15, per discussion with user): `improvement_vs_
zero_return` is NOT a fair test of whether valuation actually predicts
anything. US equities have a positive long-run average return, so a model
that has learned nothing except "returns are usually positive" can beat the
always-predict-0 baseline purely by capturing that broad market drift —
this is exactly the "long-term uptrend means a random up-call is often
right" confound the user raised. It's also why the earlier real-data runs
showed ML models "beating" zero_return at 6-12 months even though
quantile_spread/information_coefficient/univariate_fama_macbeth (which are
cross-sectional at each period, so period-wide drift cancels out) all found
nothing significant at any horizon — those two facts aren't in tension, the
zero_return comparison was just measuring the wrong thing.

`improvement_vs_sector_average` is the real bar: sector_average already
captures "this stock is in a sector that generally does well" using only
Train data, so for a model to show real improvement over it, it has to be
separating stocks WITHIN a sector by their valuation percentiles, not just
riding broad (or sector-wide) drift. improvement_vs_zero_return is kept
alongside it only as context, not as the metric that decides whether a
model earns its keep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ZeroReturnBaseline:
    """Always predicts 0 — "valuation tells you nothing, guess no
    movement." The floor every other approach has to beat to be worth
    anything at all."""

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "ZeroReturnBaseline":
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


class SectorAverageBaseline:
    """Predicts that row's sector's average forward return, computed from
    Train only. Tests whether the valuation percentiles add anything beyond
    just knowing which industry a stock is in."""

    def __init__(self):
        self.sector_means: dict[str, float] = {}
        self.overall_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "SectorAverageBaseline":
        clean = train_df[["sector", target_col]].dropna()
        self.sector_means = clean.groupby("sector")[target_col].mean().to_dict()
        self.overall_mean = clean[target_col].mean() if not clean.empty else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["sector"].map(self.sector_means).fillna(self.overall_mean).to_numpy()


class RuleBasedScoreBaseline:
    """Buckets composite_score into quintiles using Train's own bin edges,
    then predicts that bucket's average Train forward return — the
    "rule-based lookup table" from technical-spec.md 3.2/9: no regression at
    all, just "stocks that scored like this historically returned about
    this much.\""""

    def __init__(self, n_buckets: int = 5, score_col: str = "composite_score"):
        self.n_buckets = n_buckets
        self.score_col = score_col
        self.bin_edges: np.ndarray | None = None
        self.bucket_means: dict[int, float] = {}
        self.overall_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame, target_col: str) -> "RuleBasedScoreBaseline":
        clean = train_df[[self.score_col, target_col]].dropna()
        self.overall_mean = clean[target_col].mean() if not clean.empty else 0.0
        if len(clean) < self.n_buckets * 5:
            self.bin_edges = None  # not enough rows to cut into buckets meaningfully
            return self

        _, self.bin_edges = pd.qcut(clean[self.score_col], self.n_buckets, retbins=True, duplicates="drop")
        bucket_idx = np.digitize(clean[self.score_col], self.bin_edges[1:-1])
        self.bucket_means = clean.assign(_bucket=bucket_idx).groupby("_bucket")[target_col].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.bin_edges is None:
            return np.full(len(df), self.overall_mean)
        scores = df[self.score_col]
        bucket_idx = np.digitize(scores.fillna(np.inf), self.bin_edges[1:-1])
        preds = np.array([self.bucket_means.get(b, self.overall_mean) for b in bucket_idx], dtype=float)
        preds[scores.isna().to_numpy()] = self.overall_mean
        return preds


def compare_baselines(panel: pd.DataFrame, horizon: int, train_end: pd.Timestamp, val_end: pd.Timestamp) -> pd.DataFrame:
    """Fits all 3 naive baselines + Ridge/XGBoost/RandomForest on the exact
    same Train/Test split and rows (same NaN-dropping as pipeline.prepare_xy,
    so this isn't comparing MAE across subtly different samples), and
    reports each one's Test MAE/RMSE side by side, plus how much each
    improves on the zero-return floor.
    """
    from pipeline import FEATURE_COLS, prepare_xy, time_split
    from models import RandomForestQuantileModel, RidgeQuantileModel, XGBQuantileModel, evaluate

    target_col = f"fwd_return_{horizon}m"
    train, _, test = time_split(panel, train_end=train_end, val_end=val_end)

    X_train, y_train, _ = prepare_xy(train, horizon)
    X_test, y_test, _ = prepare_xy(test, horizon)

    if len(X_train) < len(FEATURE_COLS) * 5:
        return pd.DataFrame(
            [{"baseline": "all", "test_mae": np.nan, "test_rmse": np.nan, "n_test": 0,
              "note": "not enough training rows for this feature count"}]
        )

    # Same exact rows the regression models are scored on (aligned by
    # index) — every baseline below is compared on an identical sample.
    train_meta = train.loc[X_train.index, ["sector", "composite_score"]].join(y_train.rename(target_col))
    test_meta = test.loc[X_test.index, ["sector", "composite_score"]].join(y_test.rename(target_col))

    rows = []

    def _score(name: str, y_true, y_pred) -> None:
        rows.append(
            {
                "baseline": name,
                "test_mae": mean_absolute_error(y_true, y_pred),
                "test_rmse": mean_squared_error(y_true, y_pred) ** 0.5,
                "n_test": len(y_true),
            }
        )

    zero = ZeroReturnBaseline().fit(train_meta, target_col)
    _score("zero_return", test_meta[target_col], zero.predict(test_meta))

    sector_avg = SectorAverageBaseline().fit(train_meta, target_col)
    _score("sector_average", test_meta[target_col], sector_avg.predict(test_meta))

    rule_based = RuleBasedScoreBaseline().fit(train_meta, target_col)
    _score("rule_based_score", test_meta[target_col], rule_based.predict(test_meta))

    ridge = RidgeQuantileModel().fit(X_train, y_train)
    m = evaluate(ridge, X_test, y_test)
    rows.append({"baseline": "ridge", "test_mae": m["mae"], "test_rmse": m["rmse"], "n_test": m["n"]})

    try:
        xgb_model = XGBQuantileModel().fit(X_train, y_train)
        m = evaluate(xgb_model, X_test, y_test)
        rows.append({"baseline": "xgboost", "test_mae": m["mae"], "test_rmse": m["rmse"], "n_test": m["n"]})
    except ImportError:
        rows.append({"baseline": "xgboost", "test_mae": np.nan, "test_rmse": np.nan, "n_test": 0, "note": "xgboost not installed"})

    rf = RandomForestQuantileModel().fit(X_train, y_train)
    m = evaluate(rf, X_test, y_test)
    rows.append({"baseline": "random_forest", "test_mae": m["mae"], "test_rmse": m["rmse"], "n_test": m["n"]})

    result = pd.DataFrame(rows)
    zero_mae = result.loc[result["baseline"] == "zero_return", "test_mae"].iloc[0]
    sector_mae = result.loc[result["baseline"] == "sector_average", "test_mae"].iloc[0]
    result["improvement_vs_zero_return"] = zero_mae - result["test_mae"]
    # The real bar (see module docstring, 2026-09-15) — positive here means
    # a model beats "just knowing the sector," not just broad market drift.
    result["improvement_vs_sector_average"] = sector_mae - result["test_mae"]
    return result
