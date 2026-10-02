"""
Screening report: fair_value.py's gaps -> a label per stock, a plain-language
explanation, and flags that ask for a second look.

Descriptive only: "at this date the stock trades X% below/above the multiple
its fundamentals usually get". Not a return forecast (gap_return_test).

Labels: cheapness_rank = percentile of -valuation_gap among all stocks of the
same as_of, cut into five 20% bands (config.LABEL_BANDS). Loss-makers are
ranked on the same scale using the multiples that still work for them
(LOSS_VIEW_KEYS) and labelled "적자 · <band>". Verdicts are withheld for
heavily indebted loss-makers on the cheap side and for new listings judged on
one multiple (config FINANCIAL_RISK_* / NEW_LISTING_*). The naive sector rank
stays as sector_valuation_rank for reference.

Flags (warnings, labels unchanged):
  - meme_flag: |5-day move| > 15% and volume > 3 std above its 63-day mean;
    only sees the 5 days before each snapshot.
  - value_trap_flag: 매우 저평가 for 4 snapshots in a row with revenue growth in
    the sector's bottom 40% (never validated as a value-trap detector).
  - transition_flag/type: 매우 저평가 <-> 매우 고평가 flips, attributed to price vs. EPS.
  - report_lag_flag: price moved > REPORT_LAG_MOVE_LIMIT (log) since the
    fundamentals' quarter end, so the rescaled multiples may be stale.
  - fundamental_break_flag: a one-off EPS or per-share break inside the TTM
    window (features._add_fundamental_breaks).
  - single_view_flag: a non-neutral verdict resting on one multiple.
  - financial_risk_flag / heavy_debt_flag: heavy debt on a loss-maker / a
    profitable stock (heavy_debt).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CHEAP_THRESHOLD,
    EXPENSIVE_THRESHOLD,
    FINANCIAL_RISK_LABEL,
    FINANCIAL_RISK_NET_DEBT_TO_CAPITAL,
    FINANCIAL_RISK_TOP_SHARE,
    NEW_LISTING_DAYS,
    NEW_LISTING_LABEL,
    LABEL_BANDS,
    LABELS,
    VERY_CHEAP_LABEL,
    VERY_EXPENSIVE_LABEL,
    FAIR_VALUE_FEATURES,
    FAIR_VALUE_TARGETS,
    BASE_EFFECT_CAGR,
    IMPLIED_GROWTH_YEARS,
    LABEL_DETAIL_BAND,
    MOMENTUM_INDICATORS,
    QUALITY_INDICATORS,
    VALUATION_INDICATORS,
)
from fair_value import FEATURE_LABELS_KO, LOSS_VIEW_KEYS, add_fair_value, loss_flag, target_features

PRICE_SPIKE_THRESHOLD = 0.15
VOLUME_ZSCORE_THRESHOLD = 3.0

VALUE_TRAP_LOOKBACK_PERIODS = 4
VALUE_TRAP_WEAK_GROWTH_PERCENTILE = 40.0

# |log change| between consecutive snapshots that counts as "this moved".
TRANSITION_PRICE_THRESHOLD = 0.15
TRANSITION_EARNINGS_THRESHOLD = 0.15

# |log(price now / price at the fundamentals' period end)| above this
# (~ -33% / +49%) sets report_lag_flag. A heuristic, not tuned on any split.
REPORT_LAG_MOVE_LIMIT = 0.4

# Drivers smaller than this (log units, ~5%) are left out of explanations.
MIN_DRIVER_EFFECT = 0.05


def _pct_rank(values: pd.Series, by: pd.Series) -> pd.Series:
    """Percentile (0-100] of each value within its group; NaN if the value
    is missing or the group has fewer than 3 values."""
    grouped = values.groupby(by)
    counts = grouped.transform("count")
    return (100.0 * grouped.rank(method="max") / counts).where(counts >= 3)


LOSS_LABEL_PREFIX = "적자 · "  # loss-makers, ranked on the all-stock scale


def valuation_label(cheapness_rank: float) -> str:
    if pd.isna(cheapness_rank):
        return "데이터 부족"
    # extremes: >= CHEAP_THRESHOLD / <= EXPENSIVE_THRESHOLD; then 저평가 >= 60, 중립 >= 40, 고평가 > 20
    if cheapness_rank >= CHEAP_THRESHOLD:
        return VERY_CHEAP_LABEL
    if cheapness_rank <= EXPENSIVE_THRESHOLD:
        return VERY_EXPENSIVE_LABEL
    return next(label for edge, label in LABEL_BANDS[1:] if cheapness_rank >= edge)


def heavy_debt(df: pd.DataFrame) -> pd.Series:
    """Negative equity, or (outside Financials) net debt / capital above
    FINANCIAL_RISK_NET_DEBT_TO_CAPITAL or in the date's top
    FINANCIAL_RISK_TOP_SHARE among non-Financials (config)."""
    nan = pd.Series(np.nan, index=df.index)
    ndc, book = df.get("net_debt_to_capital", nan), df.get("book_value", nan)
    nonfin = df["sector"] != "Financials"
    ndc_nf = ndc.where(nonfin)
    top = ndc_nf.groupby(df["as_of"]).rank(pct=True) > 1 - FINANCIAL_RISK_TOP_SHARE
    return ((book <= 0) | (ndc_nf > FINANCIAL_RISK_NET_DEBT_TO_CAPITAL) | top).fillna(False).astype(bool)


def financial_risk(df: pd.DataFrame) -> pd.Series:
    """Loss-maker with heavy_debt."""
    loss = df["loss_flag"].astype(bool) if "loss_flag" in df.columns else pd.Series(False, index=df.index)
    return loss & heavy_debt(df)


def new_listing_single_view(df: pd.DataFrame) -> pd.Series:
    """A verdict resting on ONE multiple within NEW_LISTING_DAYS of the stock's
    first snapshot, unless that snapshot is the panel's first date (listed earlier)."""
    first = df.groupby("ticker")["as_of"].transform("min")
    young = (first > df["as_of"].min()) & ((df["as_of"] - first).dt.days < NEW_LISTING_DAYS)
    loss = df["loss_flag"].astype(bool) if "loss_flag" in df.columns else pd.Series(False, index=df.index)
    n = df["n_gaps"].where(~loss, df.get("loss_n_gaps", pd.Series(np.nan, index=df.index)))
    return (young & (n == 1)).fillna(False).astype(bool)


def flag_financial_risk(panel: pd.DataFrame) -> pd.DataFrame:
    """financial_risk_reason for rows add_labels flagged (financial_risk_flag)."""
    df = panel.copy()
    if "financial_risk_flag" not in df.columns:
        df["financial_risk_flag"] = financial_risk(df)
        df["financial_risk_withheld"] = False
    reasons = []
    for hit, ndc, book, withheld in zip(df["financial_risk_flag"], df.get("net_debt_to_capital", pd.Series(np.nan, index=df.index)),
                                        df.get("book_value", pd.Series(np.nan, index=df.index)), df["financial_risk_withheld"]):
        if not hit:
            reasons.append("")
            continue
        what = "자본잠식" if pd.notna(book) and book <= 0 else (f"순부채/총자본 {ndc:.2f}" if pd.notna(ndc) else "빚 부담 큼")
        reasons.append(f"적자 + 빚 부담 큼({what}) — 부도·재무 위험이 가격에 반영됐을 수 있는데 모델은 이 위험을 측정하지 못함"
                       + (": 그래서 '저평가'로 판단하지 않음" if withheld else ": 고평가 쪽 판단은 오히려 덜 잡혔을 수 있음"))
    df["financial_risk_reason"] = reasons
    if "heavy_debt_flag" not in df.columns:
        df["heavy_debt_flag"] = heavy_debt(df) & ~df["financial_risk_flag"] & ~df.get("loss_flag", False)
    df["heavy_debt_reason"] = np.where(
        df["heavy_debt_flag"],
        "흑자지만 빚이 많음 — 자본 구조·재무 위험이 배수에 반영돼 매우 저평가·매우 고평가 같은 극단 판정이 나오기 쉬움 "
        "(빚 많은 흑자 기업은 양 끝 비율이 각각 약 26%, 보통 20%)", "")
    return df


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """cheapness_rank + valuation_label from valuation_gap, plus context
    columns: sector_valuation_rank (naive multiple-vs-sector view),
    quality_score, momentum_score (means of sector percentiles).
    Loss-makers (fair_value.loss_flag) have no valuation_gap. Their
    loss_valuation_gap (PSR/PBR/EV-EBITDA/P-FCF) is ranked against EVERY
    stock's gap on the same four multiples that date (loss_cheapness_rank,
    2026-10-02: ranked among loss-makers only, 20% of "적자 · 중립" were in
    the most expensive 20% of all stocks) and labelled "적자 · " + the same
    five bands; loss_peer_rank keeps the rank among loss-makers for
    reference. No multiple at all: "판단 보류(적자)"."""
    df = panel.copy()
    if "loss_flag" not in df.columns:
        df["loss_flag"] = loss_flag(df)

    df["cheapness_rank"] = _pct_rank(-df["valuation_gap"], df["as_of"])
    df["valuation_label"] = df["cheapness_rank"].apply(valuation_label)
    # loss-makers: their gap ranked against every stock's gap on the same multiples
    if "loss_valuation_gap" in df.columns:
        same_basis = df[[f"{k}_gap" for k in LOSS_VIEW_KEYS if f"{k}_gap" in df.columns]].mean(axis=1, skipna=True)
        df["loss_cheapness_rank"] = _pct_rank(-same_basis, df["as_of"]).where(df["loss_valuation_gap"].notna())
        df["loss_peer_rank"] = _pct_rank(-df["loss_valuation_gap"], df["as_of"])
        ranked = df["loss_flag"] & df["loss_cheapness_rank"].notna()
        df.loc[ranked, "valuation_label"] = df.loc[ranked, "loss_cheapness_rank"].map(lambda r: LOSS_LABEL_PREFIX + valuation_label(r))
    df.loc[df["loss_flag"] & df["valuation_label"].eq("데이터 부족"), "valuation_label"] = "판단 보류(적자)"
    # distressed loss-makers: the model can't see default risk -> no cheap verdict (config)
    df["financial_risk_flag"] = financial_risk(df)
    cheap_side = df["valuation_label"].isin([LOSS_LABEL_PREFIX + VERY_CHEAP_LABEL, LOSS_LABEL_PREFIX + "저평가"])
    df["financial_risk_withheld"] = df["financial_risk_flag"] & cheap_side
    df.loc[df["financial_risk_withheld"], "valuation_label"] = FINANCIAL_RISK_LABEL
    df["heavy_debt_flag"] = heavy_debt(df) & ~df["loss_flag"].astype(bool)
    # a single-multiple verdict on a new listing: too little history to trust (config)
    df["new_listing_withheld"] = new_listing_single_view(df) & ~df["valuation_label"].str.startswith("판단 보류")
    df.loc[df["new_listing_withheld"], "valuation_label"] = NEW_LISTING_LABEL

    naive = df[[f"{k}_pct" for k in VALUATION_INDICATORS if f"{k}_pct" in df.columns]].mean(axis=1, skipna=True)
    df["sector_valuation_rank"] = _pct_rank(naive, [df["as_of"], df["sector"]])
    df["quality_score"] = df[[f"{k}_pct" for k in QUALITY_INDICATORS]].mean(axis=1, skipna=True)
    df["momentum_score"] = df[[f"{k}_pct" for k in MOMENTUM_INDICATORS]].mean(axis=1, skipna=True)
    return df


def add_expectations(panel: pd.DataFrame, years: int = IMPLIED_GROWTH_YEARS) -> pd.DataFrame:
    """What the price assumes about the future, next to what the company
    delivered (2026-09-30: a gap says "more than past fundamentals justify",
    not why — TSLA's premium is pure expectation, NVDA's mostly delivered):
      implied_excess_growth   — yearly EPS growth above the median stock's
                                that the PER needs over `years` (config
                                IMPLIED_GROWTH_YEARS); NaN without a PER
      earnings_cagr_3y_median — the median stock's earnings_cagr_3y (EPS, or
                                net income where Finnhub has no EPS —
                                features._add_durability) at that date
    The median PER is over PERs inside the model's bounds at the same as_of."""
    df = panel.copy()
    lo, hi = FAIR_VALUE_TARGETS["pe"]["min"], FAIR_VALUE_TARGETS["pe"]["max"]
    pe = df["trailing_pe"].where(df["trailing_pe"] > 0)
    median_pe = pe.where(pe.between(lo, hi)).groupby(df["as_of"]).transform("median")
    df["implied_excess_growth"] = (pe / median_pe) ** (1 / years) - 1
    if "earnings_cagr_3y" not in df.columns:  # panel built before 2026-09-30 evening
        df["earnings_cagr_3y"] = df["eps_cagr_3y"] if "eps_cagr_3y" in df.columns else np.nan
    df["earnings_cagr_3y_median"] = df["earnings_cagr_3y"].groupby(df["as_of"]).transform("median")
    return df


# Delivered (past 3-year) vs. required (priced-in) excess growth -> detail.
# Names describe the comparison only, not whether the future will back the price.
PAST_ABOVE = "과거 성장 > 요구 성장"
REQUIRED_ABOVE = "요구 성장 > 과거 성장"
SIMILAR = "요구 성장 ≈ 과거 성장"
TURNAROUND = "흑자 전환"
NO_GROWTH_HISTORY = "성장 이력 없음"
ONE_OFF_NOTE = " (일회성 손익 가능)"
BASE_EFFECT_NOTE = " (기저 효과 가능)"
DETAIL_NOTES = (ONE_OFF_NOTE, BASE_EFFECT_NOTE)


def add_label_detail(panel: pd.DataFrame, band: float = LABEL_DETAIL_BAND) -> pd.DataFrame:
    """Splits every verdict by WHY (user idea, 2026-09-30): compare the
    growth the price assumes (implied_excess_growth, add_expectations) with
    the growth the company delivered, both relative to the median stock:
      realized_excess_growth = (1 + earnings_cagr_3y) / (1 + median) - 1
    within `band` of each other -> SIMILAR; otherwise PAST_ABOVE or
    REQUIRED_ABOVE by which one is higher (for every label: a 저평가 stock
    whose price requires less growth than it delivered is PAST_ABOVE). A loss 3 years ago and a profit now (growth
    undefined; 155 of the 283 undivided verdicts on 2026-09-30) -> 흑자 전환;
    anything else without a growth rate (a loss now, under 3 years of
    history) -> 성장 이력 없음. With fundamental_break_flag the delivered
    growth may be a one-off, and above BASE_EFFECT_CAGR a tiny base, so the
    detail says so. valuation_label itself
    is unchanged (the flags key on it); label_detail is added and
    valuation_view = "고평가 · 요구 성장 > 과거 성장" for the report. "Delivered" = the
    past 3 years' growth rate — a description of the price, not a forecast."""
    df = panel.copy()
    median = df["earnings_cagr_3y_median"]
    df["realized_excess_growth"] = (1 + df["earnings_cagr_3y"]) / (1 + median) - 1
    diff = df["realized_excess_growth"] - df["implied_excess_growth"]
    one_off = df["fundamental_break_flag"] if "fundamental_break_flag" in df.columns else pd.Series(False, index=df.index)
    turnaround = (df["earnings_turnaround_3y"] == 1) if "earnings_turnaround_3y" in df.columns \
        else pd.Series(False, index=df.index)
    detail = []
    tiny_base = df["earnings_cagr_3y"] > BASE_EFFECT_CAGR
    for label, d, broken, turned, tiny in zip(df["valuation_label"], diff, one_off.fillna(False), turnaround,
                                              tiny_base):
        if label not in LABELS:
            detail.append("")
        elif turned:
            detail.append(TURNAROUND + (ONE_OFF_NOTE if broken else ""))
        elif pd.isna(d):
            detail.append(NO_GROWTH_HISTORY)
        else:
            text = SIMILAR if abs(d) < band else (PAST_ABOVE if d > 0 else REQUIRED_ABOVE)
            detail.append(text + (ONE_OFF_NOTE if broken else "") + (BASE_EFFECT_NOTE if tiny else ""))
    df["label_detail"] = detail
    df["valuation_view"] = [f"{lbl} · {d}" if d else lbl for lbl, d in zip(df["valuation_label"], detail)]
    return df


def _expectation_text(row: pd.Series) -> str:
    """'주가에 반영된 기대: ...' line of explain(), '' without a PER."""
    excess = row.get("implied_excess_growth")
    if pd.isna(excess):
        return ""
    years = IMPLIED_GROWTH_YEARS
    if abs(excess) < 0.005:
        need = f"앞으로 {years}년간 중간 종목과 비슷하게 성장하면 현재 PER이 설명됨"
    else:
        side = "더" if excess > 0 else "덜"
        need = f"앞으로 {years}년간 이익이 중간 종목보다 매년 약 {abs(excess):.0%} {side} 늘어야 현재 PER이 설명됨"
    growth, median = row.get("earnings_cagr_3y"), row.get("earnings_cagr_3y_median")
    if pd.notna(growth) and pd.notna(median):
        past = f"최근 3년 이익 성장률 연 {growth:+.0%}, 중간 종목 {median:+.0%}"
    elif row.get("earnings_turnaround_3y") == 1:
        past = "3년 전 적자에서 흑자로 전환해 성장률로 표시할 수 없음"
    else:
        past = "최근 3년 이익 성장률 계산 불가(적자 또는 3년 미만 이력)"
    return f"주가에 반영된 기대: {need} ({past}; {years}년 뒤 중간 종목 수준 PER, 같은 할인율 가정)"


def flag_meme_stock(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    spike = df["price_spike_5d"].abs() > PRICE_SPIKE_THRESHOLD
    volume = df["volume_zscore_63d"] > VOLUME_ZSCORE_THRESHOLD
    df["meme_flag"] = (spike & volume).fillna(False).astype(bool)
    df["meme_reason"] = ""
    hit = df["meme_flag"]
    direction = np.where(df.loc[hit, "price_spike_5d"] > 0, "급등", "급락")
    df.loc[hit, "meme_reason"] = [
        f"최근 5거래일 {d} ({s:+.1%}) + 거래량 이상치(63일 평균 대비 {z:.1f}표준편차) — 밸류에이션 판단 신뢰도 낮음"
        for d, s, z in zip(direction, df.loc[hit, "price_spike_5d"], df.loc[hit, "volume_zscore_63d"])
    ]
    return df


def flag_value_trap(panel: pd.DataFrame, lookback_periods: int = VALUE_TRAP_LOOKBACK_PERIODS) -> pd.DataFrame:
    """Flags a row when that ticker's last `lookback_periods` snapshots up to
    and including it (past rows only) were all 저평가 and its
    revenue_growth_yoy_pct is <= VALUE_TRAP_WEAK_GROWTH_PERCENTILE. Works for
    any as_of, not just the latest. Tickers with a shorter history are left
    unflagged, not guessed; a stock that only just became cheap is not
    flagged either."""
    df = panel.sort_values(["ticker", "as_of"]).copy()
    cheap = (df["valuation_label"] == VERY_CHEAP_LABEL).astype(int)  # the cheapest 20%
    streak = cheap.groupby(df["ticker"]).transform(lambda s: s.rolling(lookback_periods).sum())
    weak_growth = df["revenue_growth_yoy_pct"] <= VALUE_TRAP_WEAK_GROWTH_PERCENTILE
    df["value_trap_flag"] = ((streak == lookback_periods) & weak_growth).fillna(False).astype(bool)
    df["value_trap_reason"] = np.where(
        df["value_trap_flag"],
        f"최근 {lookback_periods}개 시점 연속 매우 저평가 + 매출성장률은 섹터 하위 "
        f"{VALUE_TRAP_WEAK_GROWTH_PERCENTILE:.0f}% 이내 — 모델이 못 보는 이유로 계속 싼 것일 수 있음 "
        "(지속 할인·저성장 경고; 밸류트랩인지는 검증하지 않은 규칙)",
        "",
    )
    return df


def flag_report_lag(panel: pd.DataFrame, limit: float = REPORT_LAG_MOVE_LIMIT) -> pd.DataFrame:
    """report_lag_flag (see module docstring). The price move since the
    fundamentals' period end is read back from the rescaling itself: every
    price multiple was multiplied by the same price ratio, so rescaled /
    reported recovers it (first multiple that has both). A panel without
    *_reported columns (built before rescaling) is left unflagged."""
    df = panel.copy()
    move = pd.Series(np.nan, index=df.index)
    for col in ("price_to_book", "price_to_sales", "trailing_pe", "price_to_fcf"):
        if f"{col}_reported" in df.columns:
            ratio = df[col] / df[f"{col}_reported"]
            move = move.fillna(np.log(ratio.where(ratio > 0)))
    df["price_move_since_report"] = np.expm1(move)
    df["report_lag_flag"] = (move.abs() > limit).fillna(False).astype(bool)
    period = df.get("fundamentals_period", pd.Series(pd.NaT, index=df.index))
    df["report_lag_reason"] = [
        f"마지막 재무 기준일({pd.Timestamp(p).date() if pd.notna(p) else '?'}) 이후 주가 {m:+.0%} — "
        "분사·인수합병·실적 충격이 아직 재무 지표에 반영되지 않았을 수 있음 (배수 괴리가 과장될 수 있으니 뉴스 먼저 확인)"
        if hit else ""
        for hit, p, m in zip(df["report_lag_flag"], period, df["price_move_since_report"])
    ]
    return df


def flag_fundamental_break(panel: pd.DataFrame) -> pd.DataFrame:
    """fundamental_break_flag + reason (see module docstring) from the
    *_period / *_ratio columns features.py adds. A panel built before those
    existed is left unflagged."""
    df = panel.copy()
    col = lambda c: df[c] if c in df.columns else pd.Series(np.nan, index=df.index)
    reasons = []
    for eps_when, eps_ratio, sps_when, sps_ratio in zip(col("eps_one_off_period"), col("eps_one_off_ratio"),
                                                        col("per_share_break_period"), col("per_share_break_ratio")):
        parts = []
        if pd.notna(eps_when):
            size = f"전년 같은 분기의 {eps_ratio:.1f}배" if pd.notna(eps_ratio) and eps_ratio > 0 else "전년 같은 분기와 부호가 반대(적자)"
            parts.append(f"{pd.Timestamp(eps_when).date()} 분기 EPS가 평소와 크게 다름({size}) — 일회성 손익 의심")
        if pd.notna(sps_when):
            parts.append(f"{pd.Timestamp(sps_when).date()} 분기 주당매출이 전년 대비 {sps_ratio:.1f}배로 급변 — 분사·인수합병·주식 수 기준 변경·데이터 오류 의심")
        reasons.append(" / ".join(parts) + (" (최근 12개월 배수가 서로 다른 분기를 섞고 있어 괴리를 그대로 믿기 어려움)" if parts else ""))
    df["fundamental_break_reason"] = reasons
    df["fundamental_break_flag"] = df["fundamental_break_reason"] != ""
    return df


def flag_single_view(panel: pd.DataFrame) -> pd.DataFrame:
    """single_view_flag: a non-neutral label (매우 저평가 ... 매우 고평가, loss-makers
    too) from ONE multiple, because the
    others were out of range, missing or excluded for the sector — nothing
    cross-checks it (see module docstring)."""
    df = panel.copy()
    n_gaps = df["n_gaps"] if "n_gaps" in df.columns else pd.Series(np.nan, index=df.index)
    basis = df["valuation_basis"] if "valuation_basis" in df.columns else pd.Series("", index=df.index)
    if "loss_n_gaps" in df.columns:  # loss-makers: their own verdict's multiples
        loss = df["valuation_label"].astype(str).str.startswith(LOSS_LABEL_PREFIX)
        n_gaps = n_gaps.where(~loss, df["loss_n_gaps"])
        basis = basis.where(~loss, df["loss_valuation_basis"])
    directional = [lbl for lbl in LABELS if lbl != "중립"]
    labelled = df["valuation_label"].isin(directional + [LOSS_LABEL_PREFIX + lbl for lbl in directional])
    df["single_view_flag"] = (labelled & (n_gaps == 1)).fillna(False).astype(bool)
    df["single_view_reason"] = [
        f"{b} 한 가지 배수로만 판단 — 다른 배수는 범위 밖이거나 값이 없어 교차 확인이 안 됨 (신뢰도 낮음)" if hit else ""
        for hit, b in zip(df["single_view_flag"], basis)
    ]
    return df


def _log_change(prev: float, cur: float) -> float:
    """ln(cur/prev), NaN when either side is missing or non-positive (a loss
    or a sign flip has no meaningful log change)."""
    if pd.notna(prev) and pd.notna(cur) and prev > 0 and cur > 0:
        return float(np.log(cur / prev))
    return np.nan


def classify_valuation_transition(
    panel: pd.DataFrame,
    price_threshold: float = TRANSITION_PRICE_THRESHOLD,
    earnings_threshold: float = TRANSITION_EARNINGS_THRESHOLD,
) -> pd.DataFrame:
    """Flags each snapshot where a ticker's label went 저평가 -> 고평가 versus
    its previous snapshot, and attributes it with that ticker's own price
    and EPS log changes over the same gap:
      price up, EPS held         -> 주가 상승형 (re-rating)
      EPS down, price held       -> 실적 악화형 (multiple rose mechanically)
      both                       -> 복합형
      price down, EPS held       -> 가격 급락형 (a falling price makes the
                                    stock cheaper, so the flip must come from
                                    fair value dropping — fundamentals such
                                    as ROE/growth deteriorated more)
      none past threshold / data -> 불분명 (other fundamentals moved the fair
                                    multiple, or the market re-priced them)
    Heuristic, snapshot-to-snapshot only. Uses valuation_label if present,
    else derives it from cheapness_rank (tests pass that directly), else
    from valuation_gap."""
    if "valuation_label" in panel.columns:
        df = panel.copy()
    elif "cheapness_rank" in panel.columns:
        df = panel.assign(valuation_label=panel["cheapness_rank"].apply(valuation_label))
    else:
        df = add_labels(panel)
    df = df.sort_values(["ticker", "as_of"])
    df["transition_flag"] = False
    df["transition_type"] = ""
    df["transition_reason"] = ""

    for _, group in df.groupby("ticker"):
        labels = group["valuation_label"].to_numpy()
        prices = group["price"].to_numpy(dtype=float)
        eps = group["eps"].to_numpy(dtype=float)
        for i in range(1, len(group)):
            if not (labels[i - 1] == VERY_CHEAP_LABEL and labels[i] == VERY_EXPENSIVE_LABEL):
                continue
            dp = _log_change(prices[i - 1], prices[i])
            de = _log_change(eps[i - 1], eps[i])
            price_rose = pd.notna(dp) and dp > price_threshold
            price_fell = pd.notna(dp) and dp < -price_threshold
            eps_fell = pd.notna(de) and de < -earnings_threshold
            p = f"{dp:+.1%}" if pd.notna(dp) else "데이터 부족"
            e = f"{de:+.1%}" if pd.notna(de) else "데이터 부족"

            if price_rose and eps_fell:
                kind, reason = "복합형", f"주가 {p} 상승과 EPS {e} 하락이 겹침 — 단순 재평가로 읽으면 안 됨"
            elif price_rose:
                kind, reason = "주가 상승형", f"주가가 {p} 오르며 적정가를 넘어섬 (EPS {e})"
            elif eps_fell:
                kind, reason = "실적 악화형", f"EPS가 {e} 줄며 배수가 기계적으로 상승 (주가 {p}) — 재평가보다 이익 훼손에 가까움"
            elif price_fell:
                kind, reason = (
                    "가격 급락형",
                    f"주가가 {p} 떨어졌는데도 고평가로 전환 (EPS {e}) — ROE·성장률 등 펀더멘털이 더 크게 나빠져 적정 배수가 내려갔을 가능성",
                )
            else:
                kind, reason = (
                    "불분명",
                    f"주가({p})·EPS({e}) 움직임으로는 설명되지 않음 — 다른 재무 지표 변화로 적정 배수가 바뀌었거나 시장 전체의 가격 결정이 달라진 영향",
                )
            idx = group.index[i]
            df.loc[idx, ["transition_flag", "transition_type", "transition_reason"]] = [True, kind, reason]
    return df


# Drivers that move a fair multiple through the multiple's own arithmetic
# (PER = PBR / ROE, P/FCF = PSR / FCF margin, ...), not because the market pays
# for them; their signs match the identity on 95-100% of dates (evaluate §3).
DENOMINATOR_DRIVERS = {
    "pe": {"return_on_equity"},
    "pe_norm": {"return_on_equity"},
    "pb": {"log_book_value"},
    # EV/EBITDA = (EV / assets) / (asset turnover x EBITDA margin)
    "ev_ebitda": {"asset_turnover"},
    "ps": {"asset_turnover", "log_revenue"},
    "pfcf": {"fcf_margin", "cash_conversion_3y", "log_revenue"},  # P/FCF = PSR / FCF margin: sales below
}


def _driver_text(row: pd.Series, key: str) -> str:
    """'ROE +22%, 매출성장률 +9% / 변동성 -12%' — how each fundamental moved
    this stock's fair multiple vs. the average stock at that date. Drivers in
    DENOMINATOR_DRIVERS are marked "(분모 효과)"."""
    drivers = [*target_features(FAIR_VALUE_TARGETS[key], FAIR_VALUE_FEATURES), "sector"]
    effects = {driver: row.get(f"{key}_contrib_{driver}", np.nan) for driver in drivers}
    effects = {d: v for d, v in effects.items() if pd.notna(v) and abs(v) >= MIN_DRIVER_EFFECT}
    if not effects:
        return "평균적인 종목과 비슷"
    ups = sorted((d for d in effects if effects[d] > 0), key=lambda d: -effects[d])[:2]
    downs = sorted((d for d in effects if effects[d] < 0), key=lambda d: effects[d])[:2]

    def fmt(driver: str) -> str:
        # A missing value was median-imputed + flagged, so its effect is
        # "what stocks with this value missing usually trade at" (e.g. no ROE
        # because equity is negative), not the company's actual figure.
        missing = driver != "sector" and pd.isna(row.get(driver))
        mechanical = "(분모 효과)" if driver in DENOMINATOR_DRIVERS.get(key, ()) else ""
        return f"{FEATURE_LABELS_KO[driver]}{'(값 없음)' if missing else ''} {np.expm1(effects[driver]):+.0%}{mechanical}"

    parts = []
    if ups:
        parts.append("높인 요인 " + ", ".join(map(fmt, ups)))
    if downs:
        parts.append("낮춘 요인 " + ", ".join(map(fmt, downs)))
    return " / ".join(parts)


def multiple_status(row: pd.Series, key: str) -> tuple[str, str]:
    """Why a multiple was or wasn't compared for this stock: (status, Korean
    reason). status: evaluated | excluded_sector | not_available | too_high |
    too_low | no_peers (shared by explain and llm_context)."""
    spec = FAIR_VALUE_TARGETS[key]
    actual, gap = row.get(spec["column"]), row.get(f"{key}_gap")
    if pd.notna(gap):
        return "evaluated", ""
    if row.get("sector") in spec.get("exclude_sectors", ()):
        return "excluded_sector", "금융업은 매출·EBITDA·현금흐름의 의미가 달라 비교하지 않음"
    if pd.isna(actual) or actual <= 0:
        return "not_available", ("적자라 계산 불가" if key == "pe" and row.get("loss_flag")
                                 else "값 없음(적자·자본잠식 또는 데이터 누락)")
    if actual > spec["max"]:
        base = {
            "pe": "이익이 너무 작아",
            "pb": "자본이 너무 작아(자사주 매입 등)",
            "ps": "매출 대비 가격이 너무 높아",
            "ev_ebitda": "EBITDA가 너무 작아",
            "pfcf": "잉여현금흐름이 너무 작아",
            "pe_norm": "3년 평균 이익이 너무 작아",
        }.get(key, "기준 범위를 벗어나")
        return "too_high", f"{base} 배수로 비교하기 어려움"
    if actual < spec["min"]:
        return "too_low", "비정상적으로 작은 값 — 데이터 오류 가능성"
    return "no_peers", "같은 시점 비교 종목 부족"


def explain(row: pd.Series) -> str:
    """One paragraph per stock: actual vs. fair multiple for each multiple
    that could be evaluated, the drivers behind each fair multiple, and why
    a multiple was skipped if it was."""
    lines = []
    for key, spec in FAIR_VALUE_TARGETS.items():
        if spec["column"] not in row.index:  # panel built before this multiple existed
            continue
        actual, fair, gap = row.get(spec["column"]), row.get(f"fair_{key}"), row.get(f"{key}_gap")
        status, why = multiple_status(row, key)
        if status == "evaluated":
            excluded = " [적자라 참고용]" if row.get("loss_flag") and key not in LOSS_VIEW_KEYS else ""
            if not spec.get("in_verdict", True):
                excluded = " [참고용, 종합 판단 제외]"
            particle = "를" if spec["label"].endswith(("EBITDA", "FCF")) else "을"  # 에이/에프 end in a vowel
            lines.append(
                f"{spec['label']} {actual:.1f}배 (적정 {fair:.1f}배, {np.expm1(gap):+.0%}){excluded} — 적정 "
                f"{spec['label']}{particle} {_driver_text(row, key)}"
            )
        elif status == "excluded_sector":
            continue  # one summary line below instead of one per multiple
        elif status == "too_high":
            lines.append(f"{spec['label']} {actual:.0f}배: {why}")
        elif status == "too_low":
            lines.append(f"{spec['label']} {actual:.2f}배: {why}")
        else:
            lines.append(f"{spec['label']}: {why}")
    skipped = [spec["label"] for spec in FAIR_VALUE_TARGETS.values() if row.get("sector") in spec.get("exclude_sectors", ())]
    if skipped:
        lines.append(f"{'·'.join(skipped)}: 금융업은 매출·EBITDA·현금흐름의 의미가 달라 비교하지 않음 (PER·PBR로만 판단)")
    expectation = _expectation_text(row)
    if expectation:
        lines.append(expectation)
    detail = row.get("label_detail")
    note = ""
    if isinstance(detail, str) and ONE_OFF_NOTE in detail:
        note += " — 최근 12개월에 일회성 손익이 있어 실제 성장이 부풀거나 꺾였을 수 있음"
    if isinstance(detail, str) and BASE_EFFECT_NOTE in detail:
        note += " — 3년 전 이익이 지금의 1/8도 안 돼 성장률이 작은 기저 때문에 커 보일 수 있음"
    if detail == NO_GROWTH_HISTORY:
        lines.append(f"→ {row['valuation_label']} · {detail}: 3년 이익 성장률을 계산할 수 없어(지금 적자, "
                     "또는 3년 미만 이력) 이유를 나누지 않음")
    elif isinstance(detail, str) and detail.startswith(TURNAROUND):
        lines.append(f"→ {row['valuation_label']} · {detail}: 3년 전 적자에서 지금 흑자로 돌아서 성장률 대신 "
                     f"전환 자체를 실적으로 봄{note}")
    elif isinstance(detail, str) and detail:
        lines.append(f"→ {row['valuation_label']} · {detail}: 중간 종목 대비 최근 3년 실제 초과 성장 "
                     f"{row['realized_excess_growth']:+.1%} vs 가격에 반영된 초과 성장 {row['implied_excess_growth']:+.1%}{note}")
    if row.get("loss_flag") and pd.notna(row.get("loss_valuation_gap")):
        n = int(row["loss_n_gaps"])
        agree = int(round(row["loss_gap_agreement"] * n))
        side = "싸다" if row["loss_valuation_gap"] < 0 else "비싸다"
        lines.append(f"최근 12개월 적자 — PER을 쓸 수 없어 적자 기업끼리 {row['loss_valuation_basis']}로 비교: "
                     f"{n}개 관점 중 {agree}개가 '{side}' 쪽 (평균 괴리 {np.expm1(row['loss_valuation_gap']):+.0%}). "
                     "적자가 일회성 손상 때문인지 구조적 부진인지는 재무 지표만으로 구분할 수 없음")
        if row.get("financial_risk_withheld"):
            lines.append("판단 보류(재무 위험): 위 비교로는 싼 쪽이지만, 빚이 많아 부도·재무 위험이 가격에 반영됐을 수 있고 "
                         "모델은 그 위험을 측정하지 못해 '저평가'로 판단하지 않음")
    elif row.get("loss_flag"):
        lines.append(
            "최근 12개월 적자 — 일회성 손상인지 구조적 부진인지 재무 지표만으로 구분할 수 없어 판단을 보류함 "
            "(위 배수 괴리는 참고용)"
        )
    elif pd.notna(row.get("valuation_gap")) and row.get("n_gaps", 0) > 1:
        n = int(row["n_gaps"])
        agree = int(round(row["gap_agreement"] * n))
        side = "싸다" if row["valuation_gap"] < 0 else "비싸다"
        lines.append(f"종합: {n}개 관점 중 {agree}개가 '{side}' 쪽 (평균 괴리 {np.expm1(row['valuation_gap']):+.0%})")
    elif pd.notna(row.get("valuation_gap")) and row.get("n_gaps", 0) == 1:
        lines.append(f"종합: {row.get('valuation_basis')} 한 가지 관점으로만 판단 (괴리 {np.expm1(row['valuation_gap']):+.0%}) "
                     "— 다른 배수로 교차 확인할 수 없음")
    if row.get("new_listing_withheld"):
        lines.append("판단 보류(신규 상장): 상장 후 1년이 안 됐고 배수 하나로만 비교돼, 데이터가 부족해 판단하지 않음")
    return "\n".join(lines)


def screen(panel: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline on every row: fair value -> labels -> all flags.
    Accepts a raw panel (features.py output); fair value is recomputed here
    so a saved panel never carries a stale model."""
    df, _ = add_fair_value(panel)
    df = add_labels(df)
    df = flag_meme_stock(df)
    df = flag_value_trap(df)
    df = flag_report_lag(df)
    df = flag_fundamental_break(df)
    df = flag_single_view(df)
    df = flag_financial_risk(df)
    df = add_expectations(df)
    df = add_label_detail(df)  # after the break flag, which it reads
    return classify_valuation_transition(df)


REPORT_COLUMNS = [
    "ticker", "sector", "size_group", "as_of", "valuation_label", "label_detail", "valuation_view",
    "cheapness_rank", "valuation_gap_pct", "valuation_basis",
    "n_gaps", "gap_agreement",
    "trailing_pe", "fair_pe", "price_to_book", "fair_pb", "price_to_sales", "fair_ps",
    "ev_to_ebitda", "fair_ev_ebitda", "price_to_fcf", "fair_pfcf", "normalized_pe", "fair_pe_norm",
    "pe_gap", "pb_gap", "ps_gap", "ev_ebitda_gap", "pfcf_gap", "pe_norm_gap",
    "dividend_yield", "dividend_years_no_cut",
    "implied_excess_growth", "realized_excess_growth", "earnings_cagr_3y", "earnings_cagr_3y_median",
    "earnings_turnaround_3y",
    "sector_valuation_rank", "quality_score",
    "momentum_score", "loss_flag", "loss_cheapness_rank", "loss_peer_rank", "loss_valuation_gap_pct", "loss_valuation_basis",
    "loss_n_gaps", "loss_gap_agreement", "meme_flag", "value_trap_flag", "transition_flag", "transition_type",
    "report_lag_flag", "price_move_since_report", "fundamental_break_flag", "single_view_flag",
    "financial_risk_flag", "financial_risk_withheld", "financial_risk_reason", "heavy_debt_flag", "heavy_debt_reason",
    "new_listing_withheld",
    "meme_reason", "value_trap_reason", "transition_reason", "report_lag_reason", "fundamental_break_reason",
    "single_view_reason", "explanation",
]


def report_at(screened: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per ticker at `as_of` (default: latest), cheapest first."""
    target = pd.Timestamp(as_of) if as_of is not None else screened["as_of"].max()
    rows = screened[screened["as_of"] == target].copy()
    if rows.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    rows["explanation"] = rows.apply(explain, axis=1)
    # report gaps as % (actual / fair - 1); the model works in log units
    rows["valuation_gap_pct"] = np.expm1(rows["valuation_gap"])
    if "loss_valuation_gap" in rows.columns:
        rows["loss_valuation_gap_pct"] = np.expm1(rows["loss_valuation_gap"])
    for key in FAIR_VALUE_TARGETS:
        if f"{key}_gap" in rows.columns:
            rows[f"{key}_gap"] = np.expm1(rows[f"{key}_gap"])
    rows = rows.reindex(columns=REPORT_COLUMNS)  # a panel built before PSR has no price_to_sales
    return rows[REPORT_COLUMNS].sort_values("cheapness_rank", ascending=False, na_position="last").reset_index(drop=True)
