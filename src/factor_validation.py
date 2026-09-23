"""
Factor-validity tests — this is the actual "does the factor mean anything"
check (technical-spec.md section 3.4), independent of any predictive model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def quantile_spread(panel: pd.DataFrame, score_col: str, return_col: str, n_quantiles: int = 5) -> dict:
    """Mean/median forward return in the top vs. bottom quantile of
    `score_col`, plus a t-test on whether the spread is nonzero."""
    df = panel[[score_col, return_col]].dropna()
    if len(df) < n_quantiles * 5:
        return {"error": f"not enough rows ({len(df)}) for {n_quantiles} quantiles"}

    df = df.copy()
    df["q"] = pd.qcut(df[score_col], n_quantiles, labels=False, duplicates="drop")
    top = df[df["q"] == df["q"].max()][return_col]
    bottom = df[df["q"] == df["q"].min()][return_col]

    t_stat, p_value = stats.ttest_ind(top, bottom, equal_var=False)
    return {
        "top_quantile_mean_return": top.mean(),
        "bottom_quantile_mean_return": bottom.mean(),
        "spread": top.mean() - bottom.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "n_top": len(top),
        "n_bottom": len(bottom),
    }


def information_coefficient(panel: pd.DataFrame, score_col: str, return_col: str) -> dict:
    """Spearman IC computed per as_of date, then averaged across dates
    (standard practice — avoids pooling correlated cross-sectional
    observations into one inflated correlation)."""
    ics = []
    for as_of, group in panel.groupby("as_of"):
        sub = group[[score_col, return_col]].dropna()
        if len(sub) < 5:
            continue
        ic, _ = stats.spearmanr(sub[score_col], sub[return_col])
        if not np.isnan(ic):
            ics.append(ic)
    if not ics:
        return {"error": "no valid periods"}
    ics = np.array(ics)
    t_stat, p_value = stats.ttest_1samp(ics, 0)
    return {
        "mean_ic": ics.mean(),
        "std_ic": ics.std(),
        "n_periods": len(ics),
        "t_stat": t_stat,
        "p_value": p_value,
    }


def univariate_fama_macbeth(X: pd.DataFrame, y: pd.Series, as_of: pd.Series) -> pd.DataFrame:
    """Per-feature Fama-MacBeth significance test, one feature at a time —
    the fix for the multicollinearity problem found on real data (2026-09-14
    run: statsmodels raised SingularMatrixWarning inside
    models.FamaMacBethModel, which fits ALL indicators jointly in one
    regression per period; several valuation/quality indicators turned out
    correlated enough that the joint design matrix went rank-deficient in
    some periods, making its per-feature coefficients/p-values untrustworthy).

    For each feature: run a separate cross-sectional OLS (`feature ~ const`)
    per as_of period, collect that period's coefficient, then t-test
    (scipy.stats.ttest_1samp) whether the average coefficient across periods
    is nonzero — same Fama-MacBeth logic as models.FamaMacBethModel, just
    one predictor at a time. A single predictor + constant is rank 2, so
    this can't go rank-deficient from some OTHER feature's collinearity —
    only from too few observations in a period, or a feature that's
    literally constant that period, both handled by skipping.

    Trade-off to keep in mind reading the output: this measures each
    feature's OWN relationship with forward returns, not its marginal
    contribution once the other features are controlled for. Two highly
    correlated features (e.g. trailing_pe and price_to_book) can both come
    back significant here even though a joint model would only need one of
    them. Read a row as "does this indicator alone carry a signal", not
    "how many independent signals do we have" — that second question is
    exactly what the joint fit couldn't answer reliably, which is why this
    exists.
    """
    import statsmodels.api as sm

    rows = []
    for feat in X.columns:
        period_coefs = []
        for period in as_of.unique():
            mask = (as_of == period) & X[feat].notna() & y.notna()
            x_p, y_p = X.loc[mask, feat], y[mask]
            if len(x_p) < 4:  # need more than just const+slope to say anything
                continue
            x_const = sm.add_constant(x_p, has_constant="add")
            fit = sm.OLS(y_p, x_const).fit()
            period_coefs.append(fit.params[feat])

        if len(period_coefs) < 2:
            rows.append(
                {"feature": feat, "mean_coef": np.nan, "t_stat": np.nan, "p_value": np.nan, "n_periods": len(period_coefs)}
            )
            continue

        coefs = np.array(period_coefs)
        t_stat, p_value = stats.ttest_1samp(coefs, 0)
        rows.append({"feature": feat, "mean_coef": coefs.mean(), "t_stat": t_stat, "p_value": p_value, "n_periods": len(coefs)})

    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def fdr_correct(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg correction across multiple indicators/horizons
    tested at once (technical-spec.md 3.4 — required whenever more than one
    test is run)."""
    reject, adj_p, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return pd.DataFrame({"p_value": p_values, "p_value_adj": adj_p, "significant_after_fdr": reject})
