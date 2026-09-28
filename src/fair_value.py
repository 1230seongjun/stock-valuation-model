"""
Fair-value model: what multiples (PER, PBR, PSR, EV/EBITDA, P/FCF — see
config.FAIR_VALUE_TARGETS) would the market normally pay for THIS company's
fundamentals, and how far are the actual multiples from that? The views are
combined into one verdict (valuation_gap, with gap_agreement = how many of
them point the same way); a stock is only called cheap/expensive when that
combined view is in the tails.

WHY: a plain "PER vs. sector peers" rank calls every high-growth, high-ROE
company expensive and every shrinking, low-return one cheap. A human analyst
adjusts for that; this model does the same adjustment systematically.

METHOD (per as_of date, per multiple in config.FAIR_VALUE_TARGETS):
  1. Cross-section = every stock at that as_of whose multiple is within the
     config bounds [min, max] (outside = undefined, distorted or bad data).
  2. Features = config.FAIR_VALUE_FEATURES, winsorized and median-imputed
     within that cross-section (+ a missing flag per feature), plus sector
     one-hot. Target = log(multiple).
  3. Ridge regression, out-of-fold by ticker (GroupKFold): each stock's fair
     multiple comes from a model that never saw that stock, so its own price
     cannot pull its own fair value toward itself.
  4. fair_<m> = exp(prediction); <m>_gap = log(actual / fair). A gap of
     -0.3 means the stock trades ~26% below the multiple its fundamentals
     would justify at that date.
  5. valuation_gap = mean of the available <m>_gap values (none for
     loss-makers — see loss_flag). On the 2026-09-23 panel the PER and PBR
     gaps had a mean per-date Spearman correlation of 0.64, and 97% of
     labelled stocks had both pointing the same way, so the combined verdict
     is not driven by one multiple.

POINT-IN-TIME: step 1 uses only rows of the same as_of, which features.py
already restricted to data known on that date. No future rows, no forward
returns. Each date's market-wide multiple level is learned from that date
itself, so a market-wide re-rating shifts fair values rather than labeling
every stock cheap/expensive at once.

EXPLANATION: with standardized features, prediction = mean(log multiple) +
sum_j coef_j * z_j, so <m>_contrib_<feature> (in log units) says how much
each fundamental moved this stock's fair multiple away from the average
stock at that date — e.g. +0.20 on return_on_equity for PBR means "high ROE
justifies a ~22% higher PBR than average".

RESULT ON THE REAL PANEL (2026-09-23, 280 tickers, snapshots up to
2026-07-01, out-of-fold R^2 of log multiple averaged over dates):
               sector-median baseline   Ridge (this model)
    PBR train        0.20                  0.47
        val          0.25                  0.52
        test         0.16                  0.55
    PER train        0.12                  0.27
        val          0.16                  0.31
        test         0.16                  0.28
Fundamentals explain PBR well (ROE's coefficient had the same sign on 100%
of dates) and PER only moderately — PER depends on expected future earnings,
which trailing fundamentals don't capture. Typical out-of-fold error is
~0.32-0.43 in log units, so a gap of +-20% is within noise; only the tails
(screening's top/bottom 20%) carry information. gap_return_test on the same
panel: 0/24 significant after FDR — cheap-vs-fair did not predict returns.
PSR (added 2026-09-28): sector median 0.19/0.12/0.11 -> Ridge 0.45/0.37/0.48;
operating margin's coefficient had the same sign on 99% of dates. Finnhub's
psTTM confirmed trailing-12-month and period-end priced (rescaled AAPL 10.60
vs. yfinance 10.66). Run `python src/main.py evaluate` for current numbers.

LIMITATIONS:
  - The residual is "cheap relative to what this model can see". Anything
    the model can't see (brand, moat, pipeline, pending M&A, one-off
    earnings) ends up in the gap, which is why screening shows the drivers.
  - A gap is not a return forecast. gap_return_test checks whether cheap
    stocks later outperformed — treat its result as a hypothesis test, not a
    selling point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

from config import (
    FAIR_VALUE_CV_FOLDS,
    FAIR_VALUE_FEATURES,
    FAIR_VALUE_MIN_ROWS,
    FAIR_VALUE_TARGETS,
    FAIR_VALUE_WINSOR_QUANTILE,
    HORIZONS_MONTHS,
    RIDGE_ALPHAS,
    TRAIN_END,
    VAL_END,
)

FEATURE_LABELS_KO = {
    "return_on_equity": "ROE",
    "operating_margin": "영업이익률",
    "revenue_growth_yoy": "매출성장률",
    "debt_to_equity": "부채비율",
    "payout_ratio_ttm": "배당성향",
    "volatility_63d": "변동성",
    "roic": "ROIC",
    "gross_margin": "매출총이익률",
    "fcf_margin": "FCF이익률",
    "net_debt_to_capital": "순부채비율",
    "current_ratio": "유동비율",
    "asset_turnover": "자산회전율",
    "sga_to_sales": "판관비율",
    "revenue_cagr_3y": "3년 매출성장률",
    "op_margin_volatility": "이익률 변동성",
    "dividend_growth_3y": "3년 배당성장률",
    "dividend_years_no_cut": "배당 무삭감 연수",
    "sector": "섹터",
}


def loss_flag(df: pd.DataFrame) -> pd.Series:
    """PER unavailable AND (latest EPS <= 0 or TTM ROE < 0): the company is
    losing money, not just missing data (ROE catches TTM losses whose latest
    quarter happens to be positive, e.g. HAS/TAP after impairments).

    These rows keep their per-multiple gaps for reference but get no
    valuation_gap, i.e. no verdict ("판단 보류(적자)" in screening):
      - PBR alone: with no PER and a negative ROE the fair PBR is an
        extrapolation (XRX came out as the #2 "저평가" stock).
      - PSR alone (tried 2026-09-28, reverted the same day): 8 of the 15
        loss-makers landed in the 15 cheapest / 15 most expensive of 278.
        One-off impairments (GILD, APD, TTWO, ARE, INTC) turn TTM operating
        margin negative, and since margin is the PSR model's strongest
        driver their fair PSR collapsed (+200~700% "고평가"); structurally
        shrinking names (XRX, F, AMC) looked cheap on sales — a value trap,
        not a mispricing. A loss year says too little about which case it is."""
    pe_missing = df["trailing_pe"].isna() | (df["trailing_pe"] <= 0)
    losing = (df["eps"] <= 0) | (df["return_on_equity"] < 0)
    return (pe_missing & losing).fillna(False).astype(bool)


def split_of(as_of: pd.Timestamp) -> str:
    if as_of <= TRAIN_END:
        return "train"
    if as_of <= VAL_END:
        return "val"
    return "test"


def _prepare_features(
    cross_section: pd.DataFrame, sectors: list[str], features: list[str] = FAIR_VALUE_FEATURES
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Design matrix for one as_of cross-section + a map from each reported
    driver ("return_on_equity", ..., "sector") to its underlying columns.
    A feature the panel doesn't have (built before it existed) is skipped."""
    X = pd.DataFrame(index=cross_section.index)
    groups: dict[str, list[str]] = {}
    for feat in (f for f in features if f in cross_section.columns):
        raw = cross_section[feat].astype(float)
        if raw.notna().sum() >= 3:
            lo, hi = raw.quantile([FAIR_VALUE_WINSOR_QUANTILE, 1 - FAIR_VALUE_WINSOR_QUANTILE])
            raw = raw.clip(lo, hi)
        median = raw.median()
        X[feat] = raw.fillna(0.0 if pd.isna(median) else median)
        groups[feat] = [feat]
        if raw.isna().any():
            X[f"{feat}_na"] = raw.isna().astype(float)
            groups[feat].append(f"{feat}_na")
    for sector in sectors:
        X[f"sector_{sector}"] = (cross_section["sector"] == sector).astype(float)
    groups["sector"] = [f"sector_{s}" for s in sectors]
    return X, groups


def _fit_ridge(X: pd.DataFrame, y: pd.Series) -> tuple[RidgeCV, pd.Series, pd.Series]:
    """Ridge on standardized columns. Constant columns (e.g. a sector absent
    from this fold) get scale 1 so they standardize to 0 instead of NaN."""
    mean = X.mean()
    scale = X.std(ddof=0).replace(0.0, 1.0)
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit((X - mean) / scale, y)
    return model, mean, scale


def _fit_cross_section(
    cs: pd.DataFrame, key: str, spec: dict, sectors: list[str], features: list[str] = FAIR_VALUE_FEATURES
) -> tuple[pd.DataFrame, dict] | None:
    """Out-of-fold fair multiple + per-driver contributions for one as_of and
    one multiple, plus diagnostics (Ridge vs. sector-median baseline on the
    exact same folds, and full-fit standardized coefficients)."""
    col = spec["column"]
    if col not in cs.columns:  # panel built before this multiple existed
        return None
    eligible = cs[(cs[col] >= spec["min"]) & (cs[col] <= spec["max"])]
    if len(eligible) < FAIR_VALUE_MIN_ROWS:
        return None

    X, groups = _prepare_features(eligible, sectors, features)
    y = np.log(eligible[col].astype(float))
    pred = pd.Series(np.nan, index=eligible.index)
    baseline = pd.Series(np.nan, index=eligible.index)
    contrib = pd.DataFrame(np.nan, index=eligible.index, columns=list(groups))

    n_splits = min(FAIR_VALUE_CV_FOLDS, eligible["ticker"].nunique())
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, eligible["ticker"]):
        tr, te = eligible.index[train_idx], eligible.index[test_idx]
        model, mean, scale = _fit_ridge(X.loc[tr], y.loc[tr])
        z = (X.loc[te] - mean) / scale
        pred.loc[te] = model.predict(z)
        terms = z * model.coef_
        for driver, cols in groups.items():
            contrib.loc[te, driver] = terms[cols].sum(axis=1)

        sector_median = y.loc[tr].groupby(eligible.loc[tr, "sector"]).median()
        baseline.loc[te] = eligible.loc[te, "sector"].map(sector_median).fillna(y.loc[tr].median()).to_numpy()

    full_model, _, _ = _fit_ridge(X, y)
    coefs = pd.Series(full_model.coef_, index=X.columns)

    out = pd.DataFrame(index=eligible.index)
    out[f"fair_{key}"] = np.exp(pred)
    out[f"{key}_gap"] = y - pred
    for driver in groups:
        out[f"{key}_contrib_{driver}"] = contrib[driver]

    sst = float(((y - y.mean()) ** 2).sum())
    diag = {
        "target": key,
        "n": len(eligible),
        "r2_model": 1 - float(((y - pred) ** 2).sum()) / sst if sst else np.nan,
        "r2_sector_median": 1 - float(((y - baseline) ** 2).sum()) / sst if sst else np.nan,
        "mae_model": float((y - pred).abs().mean()),
        "mae_sector_median": float((y - baseline).abs().mean()),
        "alpha": float(full_model.alpha_),
        **{f"coef_{f}": float(coefs[f]) for f in features if f in coefs.index},
    }
    return out, diag


def add_fair_value(panel: pd.DataFrame, features: list[str] = FAIR_VALUE_FEATURES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds fair_<m>, <m>_gap, <m>_contrib_<driver> for each multiple, plus
    the combined verdict: valuation_gap (mean gap), n_gaps, valuation_basis
    and gap_agreement (share of the available views pointing the same way as
    valuation_gap — 1.0 = every multiple agrees). Returns (panel,
    diagnostics), one diagnostics row per (as_of, multiple) that could be fit."""
    df = panel.reset_index(drop=True)
    sectors = sorted(df["sector"].dropna().unique())
    pieces: dict[str, list[pd.DataFrame]] = {key: [] for key in FAIR_VALUE_TARGETS}
    diagnostics = []

    for as_of, cs in df.groupby("as_of"):
        for key, spec in FAIR_VALUE_TARGETS.items():
            result = _fit_cross_section(cs, key, spec, sectors, features)
            if result is None:
                continue
            out, diag = result
            pieces[key].append(out)
            diagnostics.append({"as_of": as_of, "split": split_of(as_of), **diag})

    for key, frames in pieces.items():
        cols = [f"fair_{key}", f"{key}_gap"] + [f"{key}_contrib_{d}" for d in [*features, "sector"]]
        fitted = pd.concat(frames) if frames else pd.DataFrame(columns=cols, dtype=float)
        df = df.drop(columns=[c for c in cols if c in df.columns]).join(fitted.reindex(columns=cols))

    df["loss_flag"] = loss_flag(df)
    gaps = df[[f"{key}_gap" for key in FAIR_VALUE_TARGETS]].copy()
    gaps.loc[df["loss_flag"]] = np.nan  # no verdict for loss-makers, see loss_flag
    df["valuation_gap"] = gaps.mean(axis=1, skipna=True)
    df["n_gaps"] = gaps.notna().sum(axis=1)
    same_side = np.sign(gaps).eq(np.sign(df["valuation_gap"]), axis=0) & gaps.notna()
    df["gap_agreement"] = (same_side.sum(axis=1) / df["n_gaps"]).where(df["n_gaps"] > 0)
    labels = [spec["label"] for spec in FAIR_VALUE_TARGETS.values()]
    df["valuation_basis"] = gaps.notna().apply(
        lambda row: "+".join(lbl for lbl, has in zip(labels, row) if has), axis=1
    )
    return df, pd.DataFrame(diagnostics)


def compare_feature_sets(panel: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    """Out-of-fold R^2 per (feature set, multiple, split), same folds and
    bounds for every set — how model features are chosen. Decide on the
    train/val columns only; test is for the final report."""
    rows = []
    for name, features in feature_sets.items():
        _, diag = add_fair_value(panel, features)
        if diag.empty:
            continue
        summary = diag.groupby(["target", "split"])["r2_model"].mean().rename(name)
        rows.append(summary)
    table = pd.concat(rows, axis=1)
    baseline = diag.groupby(["target", "split"])["r2_sector_median"].mean().rename("sector_median")
    return pd.concat([baseline, table], axis=1).reindex(["train", "val", "test"], level="split")


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Mean per (multiple, split). The honest bar is r2_model vs.
    r2_sector_median: beating it means fundamentals explain multiples beyond
    "which sector is this"."""
    metrics = ["r2_sector_median", "r2_model", "mae_sector_median", "mae_model"]
    summary = diagnostics.groupby(["target", "split"])[metrics].mean()
    summary["n_dates"] = diagnostics.groupby(["target", "split"]).size()
    return summary.reindex(["train", "val", "test"], level="split")


def coefficient_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Mean standardized coefficient per feature and multiple, plus the share
    of dates where it had that same sign — a quick read of what the market
    has consistently paid up for."""
    rows = []
    features = [c.removeprefix("coef_") for c in diagnostics.columns if c.startswith("coef_")]
    for key, group in diagnostics.groupby("target"):
        for feat in features:
            coefs = group[f"coef_{feat}"].dropna()
            if coefs.empty:
                continue
            sign = np.sign(coefs.mean())
            rows.append({
                "target": key,
                "feature": feat,
                "mean_coef": coefs.mean(),
                "same_sign_share": float((np.sign(coefs) == sign).mean()) if sign else np.nan,
            })
    return pd.DataFrame(rows)


def _newey_west_mean_test(series: np.ndarray, lags: int) -> tuple[float, float, float]:
    """Mean, t-stat, p-value of `series` with HAC (Newey-West) standard
    errors. Needed because quarterly snapshots with 6/12-month forward
    returns overlap, which makes consecutive values correlated and a plain
    t-test overconfident (this was a flaw in the pre-refactor tests)."""
    import statsmodels.api as sm

    n = len(series)
    fit = sm.OLS(series, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags}) if lags > 0 else sm.OLS(series, np.ones(n)).fit()
    t_stat = float(fit.tvalues[0])
    p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
    return float(series.mean()), t_stat, p_value


def gap_return_test(panel: pd.DataFrame, horizons: list[int] = HORIZONS_MONTHS, n_quantiles: int = 5) -> pd.DataFrame:
    """Hypothesis test: did stocks the model called cheap (low valuation_gap)
    later outperform, WITHIN the same as_of? Per as_of it computes the
    Spearman IC between cheapness (-valuation_gap) and fwd_return_<h>m, and
    the cheapest-minus-most-expensive quintile return spread; each series is
    tested against 0 with Newey-West errors, separately per split, and BH-FDR
    is applied across every row of the table at once.

    Cross-sectional per date, so market-wide drift cancels out (a strategy
    doesn't "work" just because stocks went up). Uses exactly the rows that
    screening labels (every row with a valuation_gap; never loss-makers)."""
    df = panel.dropna(subset=["valuation_gap"]).copy()
    df["cheapness"] = -df["valuation_gap"]
    df["split"] = df["as_of"].map(split_of)
    rows = []
    for h in horizons:
        target = f"fwd_return_{h}m"
        per_date = []
        for as_of, g in df.dropna(subset=[target]).groupby("as_of"):
            if len(g) < n_quantiles * 5:
                continue
            ic = stats.spearmanr(g["cheapness"], g[target]).statistic
            q = pd.qcut(g["cheapness"].rank(method="first"), n_quantiles, labels=False)
            spread = g.loc[q == n_quantiles - 1, target].mean() - g.loc[q == 0, target].mean()
            per_date.append({"as_of": as_of, "split": split_of(as_of), "ic": ic, "spread": spread})
        per_date = pd.DataFrame(per_date)
        if per_date.empty:
            continue
        lags = max(0, int(np.ceil(h / 3)) - 1)  # quarterly snapshots overlap for h > 3 months
        for split, g in per_date.groupby("split"):
            for test in ("ic", "spread"):
                series = g[test].dropna().to_numpy()
                if len(series) < 4:
                    continue
                mean, t_stat, p_value = _newey_west_mean_test(series, lags)
                rows.append({"split": split, "horizon_m": h, "test": test, "mean": mean,
                             "t_stat": t_stat, "p_value": p_value, "n_dates": len(series)})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    reject, p_adj, _, _ = multipletests(result["p_value"], method="fdr_bh")
    result["p_value_fdr"] = p_adj
    result["significant_after_fdr"] = reject
    order = {"train": 0, "val": 1, "test": 2}
    return result.sort_values(["split", "horizon_m", "test"], key=lambda s: s.map(order) if s.name == "split" else s).reset_index(drop=True)
