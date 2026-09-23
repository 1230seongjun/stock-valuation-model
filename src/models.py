"""
Expected-return models — Ridge (primary) vs XGBoost (nonlinearity check) vs
RandomForest (overfitting cross-check). No neural nets (technical-spec.md
3.2 — data is too small for that to be a good idea).

All three expose the same tiny interface: fit(X, y) -> obj with
.predict_quantiles(X) -> DataFrame[median, low, high]. That's so
pipeline.py / a future serving layer doesn't need to special-case any of
them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config import RANDOM_SEED

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


class RidgeQuantileModel:
    """Median from RidgeCV; 25/75%ile from linear quantile regression
    (statsmodels QuantReg) on the same features. This is the primary model
    per technical-spec.md 3.2 — interpretable coefficients, low overfitting
    risk on a small panel."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0)):
        self.alphas = alphas
        self.median_model = None
        self.quantile_models = {}
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        import statsmodels.api as sm
        from statsmodels.regression.quantile_regression import QuantReg

        self.feature_names_ = list(X.columns)
        self.median_model = RidgeCV(alphas=self.alphas).fit(X, y)

        X_const = sm.add_constant(X, has_constant="add")
        for q, key in [(0.25, "low"), (0.75, "high")]:
            try:
                self.quantile_models[key] = QuantReg(y, X_const).fit(q=q, max_iter=2000)
            except Exception:
                self.quantile_models[key] = None
        return self

    def coefficients(self) -> pd.Series:
        return pd.Series(self.median_model.coef_, index=self.feature_names_)

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        import statsmodels.api as sm

        median = self.median_model.predict(X)
        X_const = sm.add_constant(X, has_constant="add")
        out = {"median": median}
        for key in ("low", "high"):
            model = self.quantile_models.get(key)
            out[key] = model.predict(X_const) if model is not None else median
        return pd.DataFrame(out, index=X.index)


class XGBQuantileModel:
    """XGBoost regression for median + native quantile objective for
    25/75%ile (xgboost>=2.0's reg:quantileerror). Comparison model only —
    used to check whether nonlinear relationships actually exist."""

    def __init__(self, **xgb_params):
        if xgb is None:
            raise ImportError("pip install xgboost")
        self.params = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "random_state": RANDOM_SEED, **xgb_params}
        self.median_model = None
        self.quantile_models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.median_model = xgb.XGBRegressor(objective="reg:squarederror", **self.params).fit(X, y)
        for q, key in [(0.25, "low"), (0.75, "high")]:
            model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q, **self.params)
            model.fit(X, y)
            self.quantile_models[key] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {"median": self.median_model.predict(X)}
        for key, model in self.quantile_models.items():
            out[key] = model.predict(X)
        return pd.DataFrame(out, index=X.index)

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.median_model.feature_importances_, index=self.median_model.feature_names_in_)


class RandomForestQuantileModel:
    """RandomForest median + quantiles derived from the spread of individual
    trees' predictions (a standard, library-free way to get RF quantiles).
    Used as an overfitting cross-check against XGBoost — bagging is more
    conservative than boosting on small data."""

    def __init__(self, **rf_params):
        self.params = {"n_estimators": 300, "max_depth": 4, "random_state": RANDOM_SEED, **rf_params}
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = RandomForestRegressor(**self.params).fit(X, y)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        # individual trees were fit on the raw array (no feature names), so
        # predict with .to_numpy() too — avoids a sklearn UserWarning spam
        # without changing any actual predictions.
        X_arr = X.to_numpy()
        all_tree_preds = np.stack([t.predict(X_arr) for t in self.model.estimators_], axis=0)
        return pd.DataFrame(
            {
                "median": np.median(all_tree_preds, axis=0),
                "low": np.percentile(all_tree_preds, 25, axis=0),
                "high": np.percentile(all_tree_preds, 75, axis=0),
            },
            index=X.index,
        )

    def feature_importance(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=self.model.feature_names_in_)


class FamaMacBethModel:
    """Classic factor-investing regression method (technical-spec.md 3.2):
    run one cross-sectional OLS per as_of date, then average the
    coefficients across dates. The t-stat on that average coefficient is
    the standard way finance papers claim "this factor has a significant
    risk premium" — it accounts for cross-sectional correlation between
    stocks in a way a single pooled regression does not.

    Needs the as_of column alongside X/y, so its fit() signature differs
    slightly from the other two models (they don't need per-period
    grouping).

    KNOWN LIMITATION (confirmed on real data 2026-09-14, not just a
    theoretical worry): fitting ALL indicators jointly in one regression per
    period is exactly what breaks when several indicators are correlated
    enough — statsmodels raised SingularMatrixWarning (rank-deficient design
    matrix) in a real run, meaning the per-feature coefficients/p-values
    this produces can't be trusted as-is. pipeline.py's default report no
    longer uses this class for significance testing — it calls
    factor_validation.univariate_fama_macbeth instead, which runs the same
    per-period-OLS-then-t-test logic one feature at a time so it can't go
    rank-deficient from another feature's collinearity. This class is kept
    as-is (still fine for a smaller/less collinear feature set, e.g. just
    composite_score) rather than modified in place.
    """

    def __init__(self):
        self.period_coefs: pd.DataFrame | None = None
        self.mean_coef: pd.Series | None = None
        self.intercept: float = 0.0
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, as_of: pd.Series):
        import statsmodels.api as sm

        self.feature_names_ = list(X.columns)
        rows = []
        for period in as_of.unique():
            mask = as_of == period
            X_p, y_p = X[mask], y[mask]
            if len(X_p) < len(self.feature_names_) + 3:  # need more obs than params
                continue
            X_const = sm.add_constant(X_p, has_constant="add")
            fit = sm.OLS(y_p, X_const).fit()
            rows.append(fit.params)
        if not rows:
            raise ValueError("No period had enough cross-sectional observations to fit")

        self.period_coefs = pd.DataFrame(rows).reset_index(drop=True)
        self.mean_coef = self.period_coefs.mean()
        self.intercept = self.mean_coef.get("const", 0.0)
        return self

    def significance_report(self) -> pd.DataFrame:
        """Per-feature Fama-MacBeth t-stat/p-value, ready for FDR correction
        (factor_validation.fdr_correct) across all features at once."""
        n = len(self.period_coefs)
        rows = []
        for feat in self.feature_names_:
            series = self.period_coefs[feat]
            se = series.std(ddof=1) / (n ** 0.5)
            t_stat = series.mean() / se if se > 0 else float("nan")
            p_value = 2 * (1 - _norm_cdf(abs(t_stat))) if t_stat == t_stat else float("nan")
            rows.append({"feature": feat, "mean_coef": series.mean(), "t_stat": t_stat, "p_value": p_value, "n_periods": n})
        return pd.DataFrame(rows)

    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        median = self.intercept + X[self.feature_names_].dot(self.mean_coef[self.feature_names_])
        # band from the dispersion of period-level residual scale isn't
        # tracked here to keep this simple — reuse the residual std of the
        # coefficients' own fit as a rough spread. Replace with a proper
        # residual-based interval before this feeds anything user-facing.
        spread = self.period_coefs[self.feature_names_].std().dot(X[self.feature_names_].abs().T).values
        return pd.DataFrame({"median": median, "low": median - spread, "high": median + spread}, index=X.index)


def _norm_cdf(x: float) -> float:
    from scipy.stats import norm

    return float(norm.cdf(x))


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    preds = model.predict_quantiles(X)["median"]
    return {
        "mae": mean_absolute_error(y, preds),
        "rmse": mean_squared_error(y, preds) ** 0.5,
        "n": len(y),
    }


def train_val_test_gap(model, X_train, y_train, X_val, y_val) -> dict:
    """Overfitting check: how much worse is Val than Train? A big gap means
    the model memorized Train rather than learning something general —
    technical-spec.md 3.5 says this should push the final pick toward
    Ridge/Fama-MacBeth over the tree models when it happens."""
    train_metrics = evaluate(model, X_train, y_train)
    val_metrics = evaluate(model, X_val, y_val)
    return {
        "train_mae": train_metrics["mae"],
        "val_mae": val_metrics["mae"],
        "overfit_gap_mae": val_metrics["mae"] - train_metrics["mae"],
    }
