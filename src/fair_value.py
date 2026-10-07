"""
Fair-value model: what multiple (config.FAIR_VALUE_TARGETS) would the market
normally pay for this company's fundamentals, and how far is the actual
multiple from it? The "fair" multiple is the market's reference level for
these fundamentals at that date, not an intrinsic value.

Per as_of date and multiple:
  1. Rows = stocks at that date with the multiple inside [min, max], minus
     exclude_sectors.
  2. Features = FAIR_VALUE_FEATURES + extra_features, as percentiles within
     the date (median-imputed, + missing flags), plus sector one-hot.
     Target = log(multiple).
  3. Ridge, out-of-fold by ticker (GroupKFold): a stock's fair multiple comes
     from a model that never saw it, so its own price can't pull it along.
  4. fair_<m> = exp(prediction), <m>_gap = log(actual / fair);
     valuation_gap = mean of the in-verdict gaps (loss-makers: see loss_flag).

Only rows of the same as_of are used (features.py already limited them to
what was known then), so a market-wide re-rating moves fair values instead
of labelling everything cheap or expensive. <m>_contrib_<feature> is each
feature's model contribution in log units, not a cause. Anything the model
can't see (moat, pipeline, M&A, one-offs) ends up in the gap, and a gap is
not a return forecast (gap_return_test). R^2 vs. the sector-median baseline:
`python src/main.py evaluate`.
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
    FAIR_VALUE_FEATURE_EXCLUDE_SECTORS,
    FAIR_VALUE_FEATURE_TRANSFORM,
    FAIR_VALUE_FEATURES,
    FAIR_VALUE_MIN_ROWS,
    FAIR_VALUE_TARGETS,
    FAIR_VALUE_WINSOR_QUANTILE,
    HORIZONS_MONTHS,
    N_JOBS,
    RIDGE_ALPHAS,
    SEALED_TEST_START,
    TRAIN_END,
    VAL_END,
)

# Below this many dates (the tests' small panels) process start-up costs
# more than parallel fitting saves.
_MIN_DATES_FOR_PARALLEL = 8

# Features that are labels, one-hot encoded instead of ranked (the industry
# group, universe.industry_groups). Sector is always in the model.
CATEGORICAL_FEATURES = ("industry",)

FEATURE_LABELS_KO = {
    "return_on_equity": "ROE",
    "accruals": "발생액(이익−영업현금흐름)",
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
    "op_margin_avg_3y": "3년 평균 영업이익률",
    "log_revenue": "매출 규모",
    "log_book_value": "자본 규모",
    "roe_avg_3y": "3년 평균 ROE",
    "eps_cagr_3y": "3년 EPS 성장률",
    "growth_consistency_3y": "성장 꾸준함",
    "eps_volatility_3y": "이익 변동성",
    "roe_volatility_3y": "ROE 변동성",
    "gross_margin_volatility_3y": "매출총이익률 변동성",
    "loss_share_3y": "적자 분기 비율",
    "cash_conversion_3y": "이익의 현금 전환율",
    "roe_spike": "ROE의 평소 대비 급등",
    "eps_spike": "이익의 평소 대비 급등",
    "sector": "섹터",
    "industry": "세부 업종",
    "revenue_growth_accel": "매출성장 가속도",
    "revenue_growth_ttm": "12개월 매출성장률",
    "revenue_growth_ttm_accel": "12개월 매출성장 가속도",
    "op_margin_change_1y": "영업이익률 1년 변화",
    "gross_margin_change_1y": "매출총이익률 1년 변화",
}


# multiples a loss-maker still has (no PER): its own verdict, see loss_flag
LOSS_VIEW_KEYS = ("ps", "pb", "ev_ebitda", "pfcf")


def loss_flag(df: pd.DataFrame) -> pd.Series:
    """PER unavailable AND (latest EPS <= 0 or TTM ROE < 0 or TTM EPS <= 0):
    the company is losing money, not just missing data (ROE catches TTM
    losses whose latest quarter happens to be positive, e.g. HAS/TAP after
    impairments; TTM EPS catches them when equity is negative and ROE is
    missing — AAL, CAR, CCOI, PTCT were 할인 on 2026-09-30, AAL #1, on a
    single positive quarter).

    They get no valuation_gap (a lone PBR or PSR view put one-off impairments
    and structural decliners at the extremes). Their own verdict is
    loss_valuation_gap on LOSS_VIEW_KEYS, ranked against every stock's gap
    on the same multiples (screening "적자 · <band>"); one-off vs. structural
    losses are still not told apart."""
    pe_missing = df["trailing_pe"].isna() | (df["trailing_pe"] <= 0)
    losing = (df["eps"] <= 0) | (df["return_on_equity"] < 0)
    if "eps_ttm" in df.columns:  # panels built before 2026-09-29 lack it
        losing |= df["eps_ttm"] <= 0
    return (pe_missing & losing).fillna(False).astype(bool)


def split_of(as_of: pd.Timestamp) -> str:
    if as_of <= TRAIN_END:
        return "train"
    if as_of <= VAL_END:
        return "val"
    return "sealed" if as_of >= SEALED_TEST_START else "test"


def target_features(spec: dict, base: list[str] = FAIR_VALUE_FEATURES) -> list[str]:
    """The features one multiple is fitted on: `base` plus that multiple's
    extra_features (config.FAIR_VALUE_TARGETS), without duplicates."""
    return [*base, *(f for f in spec.get("extra_features", ()) if f not in base)]


def _prepare_features(
    cross_section: pd.DataFrame, sectors: list[str], features: list[str] = FAIR_VALUE_FEATURES,
    transform: str = FAIR_VALUE_FEATURE_TRANSFORM,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Design matrix for one as_of cross-section + a map from each reported
    driver ("return_on_equity", ..., "sector") to its underlying columns.
    A feature the panel doesn't have (built before it existed) is skipped.
    transform: "winsor" clips to the config quantiles, "rank" replaces each
    value by its percentile in this cross-section (config
    FAIR_VALUE_FEATURE_TRANSFORM)."""
    X = pd.DataFrame(index=cross_section.index)
    groups: dict[str, list[str]] = {}
    for feat in (f for f in features if f in CATEGORICAL_FEATURES and f in cross_section.columns):
        # one-hot like sector; one driver in explanations
        levels = sorted(cross_section[feat].dropna().unique())
        cols = [f"{feat}_{level}" for level in levels]
        for level, col in zip(levels, cols):
            X[col] = (cross_section[feat] == level).astype(float)
        groups[feat] = cols
    for feat in (f for f in features if f not in CATEGORICAL_FEATURES and f in cross_section.columns):
        raw = cross_section[feat].astype(float)
        # a feature meaningless for a sector (config) is treated as missing there
        raw = raw.where(~cross_section["sector"].isin(FAIR_VALUE_FEATURE_EXCLUDE_SECTORS.get(feat, ())))
        if transform == "rank":
            raw = raw.rank(pct=True)
        elif raw.notna().sum() >= 3:
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


def _score_features(train: pd.DataFrame, rows: pd.DataFrame, X_train: pd.DataFrame, groups: dict[str, list[str]],
                    transform: str = FAIR_VALUE_FEATURE_TRANSFORM) -> pd.DataFrame:
    """Design matrix for `rows` that were not in the fit, on the scale of
    _prepare_features(train): a value becomes its percentile among the
    training rows ("rank") or is clipped to their quantiles ("winsor"); a
    missing one gets the training median and its missing flag."""
    X = pd.DataFrame(0.0, index=rows.index, columns=X_train.columns)
    for feat, cols in groups.items():
        if feat == "sector":
            for c in cols:
                X[c] = (rows["sector"] == c[len("sector_"):]).astype(float)
            continue
        if feat in CATEGORICAL_FEATURES:
            for c in cols:
                X[c] = (rows[feat] == c[len(feat) + 1:]).astype(float)
            continue
        excluded = FAIR_VALUE_FEATURE_EXCLUDE_SECTORS.get(feat, ())
        ref = train[feat].astype(float).where(~train["sector"].isin(excluded)).dropna()
        raw = rows[feat].astype(float).where(~rows["sector"].isin(excluded))
        if transform == "rank":
            ranks = np.searchsorted(np.sort(ref.to_numpy()), raw.fillna(0.0).to_numpy(), side="right") / max(len(ref), 1)
            val = pd.Series(ranks, index=raw.index).where(raw.notna())
        else:
            lo, hi = ref.quantile([FAIR_VALUE_WINSOR_QUANTILE, 1 - FAIR_VALUE_WINSOR_QUANTILE]) if len(ref) >= 3 else (None, None)
            val = raw.clip(lo, hi)
        known = X_train[feat][X_train[f"{feat}_na"] == 0] if f"{feat}_na" in X_train else X_train[feat]
        median = known.median()
        X[feat] = val.fillna(0.0 if pd.isna(median) else median)
        if f"{feat}_na" in cols:
            X[f"{feat}_na"] = raw.isna().astype(float)
    return X


def _fit_ridge(X: pd.DataFrame, y: pd.Series) -> tuple[RidgeCV, pd.Series, pd.Series]:
    """Ridge on standardized columns. Constant columns (e.g. a sector absent
    from this fold) get scale 1 so they standardize to 0 instead of NaN."""
    mean = X.mean()
    scale = X.std(ddof=0).replace(0.0, 1.0)
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit((X - mean) / scale, y)
    return model, mean, scale


def _fit_cross_section(
    cs: pd.DataFrame, key: str, spec: dict, sectors: list[str], features: list[str] = FAIR_VALUE_FEATURES,
    transform: str = FAIR_VALUE_FEATURE_TRANSFORM, contributions: bool = True,
) -> tuple[pd.DataFrame | None, dict] | None:
    """Out-of-fold fair multiple + per-driver contributions for one as_of and
    one multiple, plus diagnostics (Ridge vs. sector-median baseline on the
    exact same folds, and full-fit standardized coefficients).
    contributions=False returns (None, diagnostics) and skips the
    per-driver bookkeeping — compare_feature_sets only needs R^2."""
    col = spec["column"]
    if col not in cs.columns:  # panel built before this multiple existed
        return None
    in_scope = ~cs["sector"].isin(spec.get("exclude_sectors", ()))
    eligible = cs[in_scope & (cs[col] >= spec["min"]) & (cs[col] <= spec["max"])]
    if len(eligible) < FAIR_VALUE_MIN_ROWS:
        return None

    X, groups = _prepare_features(eligible, sectors, features, transform)
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
        if contributions:
            terms = z * model.coef_
            for driver, cols in groups.items():
                contrib.loc[te, driver] = terms[cols].sum(axis=1)

        sector_median = y.loc[tr].groupby(eligible.loc[tr, "sector"]).median()
        baseline.loc[te] = eligible.loc[te, "sector"].map(sector_median).fillna(y.loc[tr].median()).to_numpy()

    full_model, full_mean, full_scale = _fit_ridge(X, y)
    coefs = pd.Series(full_model.coef_, index=X.columns)

    out = None
    if contributions:
        out = pd.DataFrame(index=eligible.index)
        out[f"fair_{key}"] = np.exp(pred)
        out[f"{key}_gap"] = y - pred
        for driver in groups:
            out[f"{key}_contrib_{driver}"] = contrib[driver]
        out[f"{key}_capped"] = False
        # A multiple above the range stays out of the fit (near-zero denominators
        # would pull every fair multiple), but still counts in the verdict: the
        # date's model gives its fair multiple and the gap is log(max / fair),
        # "at least this expensive". Dropping it would hide the most expensive view.
        high = cs[in_scope & (cs[col] > spec["max"])] if spec.get("in_verdict", True) else cs.iloc[:0]
        if not high.empty:
            z = (_score_features(eligible, high, X, groups, transform)[X.columns] - full_mean) / full_scale
            fair = np.exp(full_model.predict(z))
            capped = pd.DataFrame(index=high.index)
            capped[f"fair_{key}"] = fair
            capped[f"{key}_gap"] = np.log(spec["max"] / fair).clip(min=0.0)
            terms = z * full_model.coef_
            for driver, cols in groups.items():
                capped[f"{key}_contrib_{driver}"] = terms[cols].sum(axis=1)
            capped[f"{key}_capped"] = True
            out = pd.concat([out, capped])

    sst = float(((y - y.mean()) ** 2).sum())
    diag = {
        "target": key,
        "n": len(eligible),
        "r2_model": 1 - float(((y - pred) ** 2).sum()) / sst if sst else np.nan,
        "r2_sector_median": 1 - float(((y - baseline) ** 2).sum()) / sst if sst else np.nan,
        "mae_model": float((y - pred).abs().mean()),
        "mae_sector_median": float((y - baseline).abs().mean()),
        # does the model order the stocks right? (R^2 can be carried by a few outliers)
        "spearman_model": float(stats.spearmanr(y, pred).statistic),
        "alpha": float(full_model.alpha_),
        **{f"coef_{f}": float(coefs[f]) for f in features if f in coefs.index},
    }
    return out, diag


def _model_features(spec: dict, features: list[str], drop: tuple[str, ...] = ()) -> list[str]:
    """target_features minus `drop` (compare-features' "-feature" rows)."""
    return [f for f in target_features(spec, features) if f not in drop]


def _fit_date(
    as_of: pd.Timestamp, cs: pd.DataFrame, sectors: list[str], features: list[str], transform: str,
    drop: tuple[str, ...], contributions: bool = True,
) -> list[tuple[str, pd.DataFrame | None, dict]]:
    """Every multiple's fit for one as_of cross-section. Dates never share
    data, so they can run in any order or in parallel."""
    results = []
    for key, spec in FAIR_VALUE_TARGETS.items():
        result = _fit_cross_section(cs, key, spec, sectors, _model_features(spec, features, drop), transform,
                                    contributions)
        if result is not None:
            out, diag = result
            results.append((key, out, {"as_of": as_of, "split": split_of(as_of), **diag}))
    return results


def add_fair_value(
    panel: pd.DataFrame, features: list[str] = FAIR_VALUE_FEATURES, transform: str = FAIR_VALUE_FEATURE_TRANSFORM,
    drop: tuple[str, ...] = (), n_jobs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds fair_<m>, <m>_gap, <m>_contrib_<driver> for each multiple
    (fitted on target_features(spec, features), minus `drop`), plus the
    combined verdict over the multiples with in_verdict (default True):
    valuation_gap (mean gap), n_gaps, valuation_basis and gap_agreement
    (share of the available views pointing the same way as valuation_gap —
    1.0 = every multiple agrees). Returns (panel, diagnostics), one
    diagnostics row per (as_of, multiple) that could be fit. Dates are fitted
    in parallel (n_jobs, default config.N_JOBS); the result doesn't depend
    on it."""
    df = panel.reset_index(drop=True)
    sectors = sorted(df["sector"].dropna().unique())
    pieces: dict[str, list[pd.DataFrame]] = {key: [] for key in FAIR_VALUE_TARGETS}
    diagnostics = []

    n_jobs = N_JOBS if n_jobs is None else n_jobs
    dates = list(df.groupby("as_of"))
    if n_jobs == 1 or len(dates) < _MIN_DATES_FOR_PARALLEL:
        per_date = [_fit_date(as_of, cs, sectors, features, transform, drop) for as_of, cs in dates]
    else:
        from joblib import Parallel, delayed

        per_date = Parallel(n_jobs=n_jobs)(
            delayed(_fit_date)(as_of, cs, sectors, features, transform, drop) for as_of, cs in dates
        )
    for results in per_date:
        for key, out, diag in results:
            pieces[key].append(out)
            diagnostics.append(diag)

    for key, frames in pieces.items():
        drivers = [*_model_features(FAIR_VALUE_TARGETS[key], features, drop), "sector"]
        cols = [f"fair_{key}", f"{key}_gap", f"{key}_capped"] + [f"{key}_contrib_{d}" for d in drivers]
        fitted = pd.concat(frames) if frames else pd.DataFrame(columns=cols, dtype=float)
        df = df.drop(columns=[c for c in cols if c in df.columns]).join(fitted.reindex(columns=cols))

    for key in FAIR_VALUE_TARGETS:
        df[f"{key}_capped"] = df[f"{key}_capped"].fillna(False).astype(bool)
    df = df.copy()  # defragment after the per-multiple joins
    df["loss_flag"] = loss_flag(df)
    verdict = {k: s for k, s in FAIR_VALUE_TARGETS.items() if s.get("in_verdict", True)}
    gaps = df[[f"{key}_gap" for key in verdict]].copy()
    gaps.loc[df["loss_flag"]] = np.nan  # no verdict for loss-makers, see loss_flag
    df["valuation_gap"] = gaps.mean(axis=1, skipna=True)
    df["n_gaps"] = gaps.notna().sum(axis=1)
    same_side = np.sign(gaps).eq(np.sign(df["valuation_gap"]), axis=0) & gaps.notna()
    df["gap_agreement"] = (same_side.sum(axis=1) / df["n_gaps"]).where(df["n_gaps"] > 0)
    labels = [spec["label"] for spec in verdict.values()]
    df["valuation_basis"] = gaps.notna().apply(
        lambda row: "+".join(lbl for lbl, has in zip(labels, row) if has), axis=1
    )
    # loss-makers' own verdict (see loss_flag): the multiples a loss leaves, loss rows only
    loss_keys = [k for k in LOSS_VIEW_KEYS if k in verdict]
    lg = df[[f"{k}_gap" for k in loss_keys]].where(df["loss_flag"], axis=0)
    df["loss_valuation_gap"] = lg.mean(axis=1, skipna=True)
    df["loss_n_gaps"] = lg.notna().sum(axis=1)
    same_side = np.sign(lg).eq(np.sign(df["loss_valuation_gap"]), axis=0) & lg.notna()
    df["loss_gap_agreement"] = (same_side.sum(axis=1) / df["loss_n_gaps"]).where(df["loss_n_gaps"] > 0)
    loss_labels = [verdict[k]["label"] for k in loss_keys]
    df["loss_valuation_basis"] = lg.notna().apply(
        lambda row: "+".join(lbl for lbl, has in zip(loss_labels, row) if has), axis=1
    )
    return df, pd.DataFrame(diagnostics)


def _set_diagnostics(panel: pd.DataFrame, spec: list[str] | tuple | dict) -> pd.DataFrame:
    """Diagnostics of one compare_feature_sets entry (sequential inside: the
    sets themselves are what runs in parallel)."""
    if isinstance(spec, dict):
        features = spec.get("features", FAIR_VALUE_FEATURES)
        transform = spec.get("transform", FAIR_VALUE_FEATURE_TRANSFORM)
        drop = tuple(spec.get("drop", ()))
    elif isinstance(spec, tuple):
        (features, transform), drop = spec, ()
    else:
        features, transform, drop = spec, FAIR_VALUE_FEATURE_TRANSFORM, ()
    sectors = sorted(panel["sector"].dropna().unique())
    rows = [diag for as_of, cs in panel.groupby("as_of")
            for _, _, diag in _fit_date(as_of, cs, sectors, features, transform, drop, contributions=False)]
    return pd.DataFrame(rows)


def compare_feature_sets(
    panel: pd.DataFrame, feature_sets: dict[str, list[str] | tuple[list[str], str] | dict], n_jobs: int | None = None,
) -> pd.DataFrame:
    """Out-of-fold R^2 per (feature set, multiple, split), same folds and
    bounds for every set — how model features are chosen. A set is a
    feature list (each multiple's extra_features are added to it),
    (features, transform), or a dict with any of "features", "transform" and
    "drop" (features taken out of every multiple, extra_features included).
    Sets run in parallel (n_jobs, default config.N_JOBS). Decide on the
    train/val columns only; test is for the final report."""
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    names, specs = list(feature_sets), list(feature_sets.values())
    # only what the fits read, so each worker gets a small copy of the panel
    used = {f for spec in specs for f in (spec.get("features", FAIR_VALUE_FEATURES) if isinstance(spec, dict)
                                          else spec[0] if isinstance(spec, tuple) else spec)}
    used |= {f for spec in FAIR_VALUE_TARGETS.values() for f in (spec["column"], *spec.get("extra_features", ()))}
    keep = ["as_of", "ticker", "sector", "eps", "trailing_pe", "return_on_equity", *sorted(used)]
    slim = panel[list(dict.fromkeys(c for c in keep if c in panel.columns))]
    if n_jobs == 1 or len(specs) == 1 or slim["as_of"].nunique() < _MIN_DATES_FOR_PARALLEL:
        diags = [_set_diagnostics(slim, spec) for spec in specs]
    else:
        from joblib import Parallel, delayed

        diags = Parallel(n_jobs=n_jobs)(delayed(_set_diagnostics)(slim, spec) for spec in specs)
    rows = [diag.groupby(["target", "split"])["r2_model"].mean().rename(name)
            for name, diag in zip(names, diags) if not diag.empty]
    table = pd.concat(rows, axis=1)
    # the baseline depends only on the rows and folds, which every set shares
    baseline = next(d for d in diags if not d.empty).groupby(["target", "split"])["r2_sector_median"].mean()
    return pd.concat([baseline.rename("sector_median"), table], axis=1).reindex(["train", "val", "test"], level="split")


def r2_by_group(panel: pd.DataFrame, group_col: str = "size_group") -> pd.DataFrame:
    """Out-of-fold R^2 of each multiple within each group (e.g. size_group),
    mean per date, from the gaps add_fair_value left in `panel`: the model is
    fitted on everyone, this asks how well it explains the multiples WITHIN
    large / mid / small caps (R^2 against that group's own mean at that
    date, so a group-wide level offset counts as unexplained)."""
    rows = []
    for key, spec in FAIR_VALUE_TARGETS.items():
        gap = f"{key}_gap"
        if gap not in panel.columns or spec["column"] not in panel.columns:  # panel built before this multiple
            continue
        df = panel.dropna(subset=[gap]).assign(y=lambda d: np.log(d[spec["column"]]))
        for (as_of, group), g in df.groupby(["as_of", group_col]):
            if len(g) < 10:
                continue
            sst = float(((g["y"] - g["y"].mean()) ** 2).sum())
            if sst > 0:
                rows.append({"target": key, "split": split_of(as_of), "group": group,
                             "r2": 1 - float((g[gap] ** 2).sum()) / sst, "n": len(g)})
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).groupby(["target", "split", "group"]).agg(r2=("r2", "mean"), rows_per_date=("n", "mean"))
    return out.reindex(["train", "val", "test"], level="split")


def summarize_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Mean per (multiple, split). The honest bar is r2_model vs.
    r2_sector_median: beating it means fundamentals explain multiples beyond
    "which sector is this"."""
    metrics = ["r2_sector_median", "r2_model", "spearman_model", "mae_sector_median", "mae_model"]
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
                "median_coef": coefs.median(),
                "q25_coef": coefs.quantile(0.25),
                "q75_coef": coefs.quantile(0.75),
                "same_sign_share": float((np.sign(coefs) == sign).mean()) if sign else np.nan,
            })
    return pd.DataFrame(rows)


def _labels_from_gap(gap: pd.Series, as_of: pd.Series) -> pd.Series:
    """할인 / 중립 / 프리미엄 by the same 20% tails as screening, from a
    combined gap (lower = cheaper)."""
    rank = (-gap).groupby(as_of).rank(pct=True)
    return pd.Series(np.where(rank > 0.8, "cheap", np.where(rank <= 0.2, "rich", "mid")), index=gap.index).where(rank.notna())


def stability_summary(panel: pd.DataFrame, gap_col: str = "valuation_gap") -> pd.DataFrame:
    """How much the verdict moves between consecutive snapshots, for tickers
    present in both: Spearman of the combined gap, the share whose 3-way
    label changed, and the share that jumped straight between 할인 and
    프리미엄. Mean per split. A tool that relabels half the market every
    quarter would be describing noise."""
    df = panel[["ticker", "as_of", gap_col]].dropna()
    df = df.assign(label=_labels_from_gap(df[gap_col], df["as_of"]))
    dates = sorted(df["as_of"].unique())
    rows = []
    for prev, cur in zip(dates, dates[1:]):
        a = df[df["as_of"] == prev].set_index("ticker")
        b = df[df["as_of"] == cur].set_index("ticker")
        common = a.index.intersection(b.index)
        if len(common) < 30:
            continue
        la, lb = a.loc[common, "label"], b.loc[common, "label"]
        rows.append({
            "as_of": cur, "split": split_of(cur),
            "rank_corr": float(stats.spearmanr(a.loc[common, gap_col], b.loc[common, gap_col]).statistic),
            "label_changed": float((la != lb).mean()),
            "cheap_rich_flip": float(((la == "cheap") & (lb == "rich") | (la == "rich") & (lb == "cheap")).mean()),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.groupby("split")[["rank_corr", "label_changed", "cheap_rich_flip"]].mean() \
        .reindex(["train", "val", "test"])


def top_overlap(gap_a: pd.Series, gap_b: pd.Series, as_of: pd.Series, side: str = "cheap") -> float:
    """Mean per-date Jaccard overlap of the cheapest (or richest) 20% under
    two versions of the combined gap (same rows) — do two model variants
    point at the same stocks, whatever their R^2?"""
    la, lb = _labels_from_gap(gap_a, as_of), _labels_from_gap(gap_b, as_of)
    scores = []
    for date in as_of.unique():
        m = as_of == date
        sa, sb = set(la[m][la[m] == side].index), set(lb[m][lb[m] == side].index)
        if sa or sb:
            scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores)) if scores else np.nan


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


def gap_return_test(panel: pd.DataFrame, horizons: list[int] = HORIZONS_MONTHS, n_quantiles: int = 5,
                    by: str | None = None) -> pd.DataFrame:
    """Hypothesis test: did stocks the model called cheap (low valuation_gap)
    later outperform, WITHIN the same as_of? Per as_of it computes the
    Spearman IC between cheapness (-valuation_gap) and fwd_return_<h>m, and
    the cheapest-minus-most-expensive quintile return spread; each series is
    tested against 0 with Newey-West errors, separately per split, and BH-FDR
    is applied across every row of the table at once.

    Cross-sectional per date, so market-wide drift cancels out (a strategy
    doesn't "work" just because stocks went up). Uses exactly the rows that
    screening labels (every row with a valuation_gap; never loss-makers).

    by="size_group": one more set of rows per group, each tested within
    that group only (cheap small caps vs. expensive small caps), with a
    "group" column; FDR still runs across the whole table, so testing more
    groups makes each result harder to call significant, as it should."""
    if by is not None:
        parts = [gap_return_test(panel, horizons, n_quantiles).assign(group="all")]
        for group, g in panel.groupby(by):
            parts.append(gap_return_test(g, horizons, n_quantiles).assign(group=group))
        parts = [p for p in parts if not p.empty]
        if not parts:
            return pd.DataFrame()
        result = pd.concat(parts, ignore_index=True)
        reject, p_adj, _, _ = multipletests(result["p_value"], method="fdr_bh")
        result["p_value_fdr"], result["significant_after_fdr"] = p_adj, reject
        return result
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
