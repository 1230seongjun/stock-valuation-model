"""
Earnings-deterioration risk (2026-10-07): the chance that a company's results
get meaningfully worse, or fail to recover, over the next year — for a
profitable company trailing-12-month EPS down 10% or more or a loss, for a
loss-maker a loss that is not even halved. One model and one threshold for
every stock; whether a company makes a loss, and what kind of loss
(loss_type), are inputs like any other. Why: the deepest discounts were mostly deserved — 56% of 큰
할인 stocks had lower EPS a year later and 12% turned to losses, against 28% /
6% for 큰 프리미엄 — while their trailing figures looked no worse than anyone
else's. A shrinking business priced low is not a discount, so a discount label
with a high risk is withheld (screening.add_deterioration_risk).

Point in time: the model used on a date is fitted, once a year (on January 1),
only on rows whose one-year outcome was already known then; features are
what was known on each row's own date, as percentiles within that date.
Price-based features (drawdown, trend) are allowed here: this is a separate
risk estimate, not an input to the fair multiple. Checked before adoption on
a fixed 2004-2019 fit: AUC within the discount bands 0.70 / 0.72 (Val /
Test), withheld stocks deteriorated 1.9x as often as the other discounted
ones, about 30% of discount labels withheld.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DETERIORATION_DROP, DETERIORATION_LOSS_IMPROVEMENT, DETERIORATION_MIN_TRAIN_ROWS

FEATURES = ["cheapness_rank", "pct_from_52w_high", "ma50_vs_ma200", "volatility_63d", "revenue_growth_yoy",
            "revenue_growth_accel", "op_margin_change_1y", "gross_margin_change_1y", "return_on_equity",
            "operating_margin", "gross_margin", "loss_share_3y", "eps_volatility_3y", "price_move_since_report",
            "log_revenue", "eps_spike", "payout_ratio_ttm", "fcf_margin", "net_debt_to_capital", "current_ratio",
            "cash_flow_positive"]
LOSS_TYPES = ["흑자", "일회성 손실(영업 흑자)", "현금은 버는 적자", "성장 투자형", "구조적 적자", "기타 적자"]


def loss_type(df: pd.DataFrame) -> pd.Series:
    """What kind of result the trailing 12 months are, from the statements:
    a profit; a net loss with an operating profit (usually one-off charges);
    a loss with positive operating cash flow; a loss while sales grow 15%+
    (investing for growth); falling sales and losses in most of the last 3
    years (structural); any other loss. Over 2004-2025 these recovered very
    differently: 68% of the first loss type were profitable a year later,
    15% of the structural ones."""
    eps = df["eps_ttm"] if "eps_ttm" in df.columns else pd.Series(np.nan, index=df.index)
    get = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    om, cfo, g, ls = get("operating_margin"), get("sec_operating_cash_flow_ttm"), get("revenue_growth_yoy"), get("loss_share_3y")
    out = pd.Series("기타 적자", index=df.index)
    out[(g < 0) & (ls >= 0.67)] = "구조적 적자"
    out[g >= 0.15] = "성장 투자형"
    out[cfo > 0] = "현금은 버는 적자"
    out[om > 0] = "일회성 손실(영업 흑자)"
    out[eps > 0] = "흑자"
    return out.where(eps.notna())
OUTCOME_DAYS, OUTCOME_TOLERANCE_DAYS = 365, 45


def outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """eps_1y (the same ticker's trailing EPS on the snapshot nearest to one
    year later, within OUTCOME_TOLERANCE_DAYS), the outcome's date, and
    deteriorated (1 = EPS down DETERIORATION_DROP or more, or a loss)."""
    snaps = df[["ticker", "as_of", "eps_ttm"]].dropna(subset=["as_of"]).sort_values("as_of")
    target = df[["ticker", "as_of"]].assign(_t=df["as_of"] + pd.Timedelta(days=OUTCOME_DAYS), _row=np.arange(len(df)))
    m = pd.merge_asof(target.sort_values("_t"), snaps.rename(columns={"as_of": "outcome_date", "eps_ttm": "eps_1y"}),
                      left_on="_t", right_on="outcome_date", by="ticker", direction="nearest",
                      tolerance=pd.Timedelta(days=OUTCOME_TOLERANCE_DAYS)).sort_values("_row")
    out = pd.DataFrame({"eps_1y": m["eps_1y"].to_numpy(), "outcome_date": m["outcome_date"].to_numpy()}, index=df.index)
    eps, e1 = df["eps_ttm"], out["eps_1y"]
    known = e1.notna() & eps.notna()
    profit_worse = (e1 <= 0) | (e1 < (1 - DETERIORATION_DROP) * eps)
    loss_stuck = (e1 <= 0) & (e1 < DETERIORATION_LOSS_IMPROVEMENT * eps)
    out["deteriorated"] = pd.Series(np.where(eps > 0, profit_worse, loss_stuck), index=df.index).astype(float).where(known)
    return out


def design(df: pd.DataFrame, sectors: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    cfo = df["sec_operating_cash_flow_ttm"] if "sec_operating_cash_flow_ttm" in df.columns else pd.Series(np.nan, index=df.index)
    extra = {"cash_flow_positive": (cfo > 0).astype(float).where(cfo.notna())}
    for f in FEATURES:
        col = extra[f] if f in extra else df[f].astype(float) if f in df.columns else pd.Series(np.nan, index=df.index)
        X[f] = col.groupby(df["as_of"]).rank(pct=True).fillna(0.5)
        X[f + "_na"] = col.isna().astype(float)
    X["fundamental_break"] = df.get("fundamental_break_flag", pd.Series(False, index=df.index)).astype(float)
    kind = loss_type(df)
    for t in LOSS_TYPES:
        X[f"type_{t}"] = (kind == t).astype(float)
    for s in sectors:
        X[f"sector_{s}"] = (df["sector"] == s).astype(float)
    return X


def risk(df: pd.DataFrame, eligible: pd.Series) -> pd.Series:
    """Deterioration probability for the eligible rows (labelled, with a known trailing EPS),
    each from the model fitted on January 1 of its year on the outcomes known
    by then. NaN while fewer than DETERIORATION_MIN_TRAIN_ROWS outcomes exist."""
    from sklearn.linear_model import LogisticRegression

    res = outcomes(df)
    X = design(df, sorted(df["sector"].dropna().unique()))
    p = pd.Series(np.nan, index=df.index)
    for year in sorted(df.loc[eligible, "as_of"].dt.year.unique()):
        cutoff = pd.Timestamp(year=int(year), month=1, day=1)
        train = eligible & res["deteriorated"].notna() & (res["outcome_date"] <= cutoff)
        if train.sum() < DETERIORATION_MIN_TRAIN_ROWS or res.loc[train, "deteriorated"].nunique() < 2:
            continue
        model = LogisticRegression(max_iter=2000, C=0.5).fit(X[train], res.loc[train, "deteriorated"])
        rows = eligible & (df["as_of"].dt.year == year)
        p.loc[rows] = model.predict_proba(X[rows])[:, 1]
    return p


def base_rates(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per label, what happened a year later in history: share deteriorated,
    share with a loss a year later, number of cases (rows with a known outcome)."""
    res = outcomes(df)
    known = res["deteriorated"].notna()
    g = pd.DataFrame({"label": labels[known], "det": res.loc[known, "deteriorated"],
                      "loss": (res.loc[known, "eps_1y"] <= 0).astype(float)}).groupby("label")
    return pd.DataFrame({"deteriorated": g["det"].mean(), "turned_loss": g["loss"].mean(), "n": g.size()})
