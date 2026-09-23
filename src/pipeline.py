"""
Orchestrates: panel -> time split -> train 3 models -> evaluate -> factor
validity checks -> one report dict. This is what train.py and the
synthetic smoke test both call.
"""
from __future__ import annotations

import pandas as pd

from config import ALL_INDICATORS
from factor_validation import fdr_correct, information_coefficient, quantile_spread, univariate_fama_macbeth
from models import RandomForestQuantileModel, RidgeQuantileModel, XGBQuantileModel, train_val_test_gap

FEATURE_COLS = [f"{k}_pct" for k in ALL_INDICATORS]


def time_split(
    panel: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    train_end: pd.Timestamp | None = None,
    val_end: pd.Timestamp | None = None,
):
    """Split by as_of date, not by row — keeps every ticker's observations
    for a given date on the same side, and enforces Train < Val < Test in
    time (technical-spec.md section 5: no random split, ever).

    Two ways to call this:
    - Pass `train_end`/`val_end` (real calendar dates, e.g. config.TRAIN_END/
      VAL_END) to split on the actually-decided Train/Val/Test boundary.
      This is what the real-data run should always use.
    - Leave them as None to fall back to the old train_frac/val_frac
      fractional split (whatever % of however many as_of dates are in
      `panel`). Only meant for exploratory/synthetic runs where no real
      calendar boundary has been decided — tests/synthetic_smoke_test.py
      relies on this default.
    """
    dates = sorted(panel["as_of"].unique())
    if train_end is None or val_end is None:
        n = len(dates)
        train_end = dates[int(n * train_frac) - 1]
        val_end = dates[int(n * (train_frac + val_frac)) - 1]

    train = panel[panel["as_of"] <= train_end]
    val = panel[(panel["as_of"] > train_end) & (panel["as_of"] <= val_end)]
    test = panel[panel["as_of"] > val_end]
    return train, val, test


def prepare_xy(panel: pd.DataFrame, horizon: int, feature_cols: list[str] = FEATURE_COLS):
    target_col = f"fwd_return_{horizon}m"
    cols = feature_cols + [target_col, "as_of"]
    clean = panel[cols].dropna()
    return clean[feature_cols], clean[target_col], clean["as_of"]


def run_pipeline(
    panel: pd.DataFrame,
    horizon: int,
    train_end: pd.Timestamp | None = None,
    val_end: pd.Timestamp | None = None,
) -> dict:
    train, val, test = time_split(panel, train_end=train_end, val_end=val_end)
    X_train, y_train, as_of_train = prepare_xy(train, horizon)
    X_val, y_val, _ = prepare_xy(val, horizon)
    X_test, y_test, _ = prepare_xy(test, horizon)

    report = {"horizon_months": horizon, "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)}

    if len(X_train) < len(FEATURE_COLS) * 5:
        report["error"] = "not enough training rows for this feature count — widen the universe or as_of date range"
        return report

    # --- primary: Ridge ---
    ridge = RidgeQuantileModel().fit(X_train, y_train)
    report["ridge"] = {
        "train_val_gap": train_val_test_gap(ridge, X_train, y_train, X_val, y_val),
        "coefficients": ridge.coefficients().to_dict(),
    }

    # --- Fama-MacBeth significance, per feature (univariate) ---
    # NOTE: models.FamaMacBethModel (joint, all-features-at-once) is NOT used
    # here anymore — real data (2026-09-14 run) hit statsmodels'
    # SingularMatrixWarning (rank-deficient design matrix) from
    # multicollinearity between indicators, which makes joint per-feature
    # coefficients/p-values untrustworthy. univariate_fama_macbeth tests each
    # feature on its own instead, which can't go rank-deficient from other
    # features' collinearity (see its docstring for the interpretation
    # trade-off this implies). models.FamaMacBethModel itself is unchanged
    # and still available if a smaller/less collinear feature set is used
    # later.
    report["fama_macbeth_univariate"] = {
        "significance": univariate_fama_macbeth(X_train, y_train, as_of_train).to_dict(orient="records"),
    }

    # --- comparison: XGBoost ---
    try:
        xgb_model = XGBQuantileModel().fit(X_train, y_train)
        report["xgboost"] = {"train_val_gap": train_val_test_gap(xgb_model, X_train, y_train, X_val, y_val)}
    except ImportError:
        report["xgboost"] = {"error": "xgboost not installed"}

    # --- overfitting cross-check: Random Forest ---
    rf_model = RandomForestQuantileModel().fit(X_train, y_train)
    report["random_forest"] = {"train_val_gap": train_val_test_gap(rf_model, X_train, y_train, X_val, y_val)}

    # --- factor validity (composite score vs. this horizon's forward return) ---
    valid_panel = pd.concat([train, val])
    report["factor_validity"] = {
        "quantile_spread": quantile_spread(valid_panel, "composite_score", f"fwd_return_{horizon}m"),
        "information_coefficient": information_coefficient(valid_panel, "composite_score", f"fwd_return_{horizon}m"),
    }

    # --- same checks for the diagnostic value-only composite (config.
    # VALUE_COMPOSITE_INDICATORS' docstring, added 2026-09-18) — does
    # dropping the indicators that showed the "wrong" sign inside
    # composite_score change IC/quantile_spread? ---
    if "value_composite_score" in valid_panel.columns:
        report["factor_validity_value_only"] = {
            "quantile_spread": quantile_spread(valid_panel, "value_composite_score", f"fwd_return_{horizon}m"),
            "information_coefficient": information_coefficient(valid_panel, "value_composite_score", f"fwd_return_{horizon}m"),
        }

    return report


def run_all_horizons(
    panel: pd.DataFrame,
    horizons: list[int],
    train_end: pd.Timestamp | None = None,
    val_end: pd.Timestamp | None = None,
) -> dict:
    return {h: run_pipeline(panel, h, train_end=train_end, val_end=val_end) for h in horizons}


def apply_fdr_across_report(reports_by_horizon: dict) -> pd.DataFrame:
    """Pulls every factor-validity p-value across all horizons — quantile
    spread, IC, AND now every individual indicator's univariate Fama-MacBeth
    p-value — into one FDR correction pass (technical-spec.md 3.4: required
    once you're testing more than one horizon/indicator at a time, and with
    11 indicators x 4 horizons that's ~44 extra tests alongside the 8
    quantile-spread/IC ones, so this matters more than it did before the
    per-feature test existed). Also includes the diagnostic value-only
    composite's quantile_spread/IC (factor_validity_value_only, added
    2026-09-18) in the SAME correction pass — it's an additional test run
    against the same data, so it has to count toward the multiple-testing
    burden like everything else, not be checked separately as if it were
    the only test performed."""
    rows = []
    for horizon, report in reports_by_horizon.items():
        qs = report.get("factor_validity", {}).get("quantile_spread", {})
        if "p_value" in qs:
            rows.append({"horizon": horizon, "test": "quantile_spread", "p_value": qs["p_value"]})
        ic = report.get("factor_validity", {}).get("information_coefficient", {})
        if "p_value" in ic:
            rows.append({"horizon": horizon, "test": "information_coefficient", "p_value": ic["p_value"]})
        for row in report.get("fama_macbeth_univariate", {}).get("significance", []):
            if pd.notna(row.get("p_value")):
                rows.append({"horizon": horizon, "test": f"fama_macbeth:{row['feature']}", "p_value": row["p_value"]})

        qs_vo = report.get("factor_validity_value_only", {}).get("quantile_spread", {})
        if "p_value" in qs_vo:
            rows.append({"horizon": horizon, "test": "quantile_spread:value_only", "p_value": qs_vo["p_value"]})
        ic_vo = report.get("factor_validity_value_only", {}).get("information_coefficient", {})
        if "p_value" in ic_vo:
            rows.append({"horizon": horizon, "test": "information_coefficient:value_only", "p_value": ic_vo["p_value"]})

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    corrected = fdr_correct(df["p_value"].tolist())
    return pd.concat([df.reset_index(drop=True), corrected[["p_value_adj", "significant_after_fdr"]]], axis=1)
