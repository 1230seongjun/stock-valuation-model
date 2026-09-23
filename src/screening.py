"""
Screening report: turns fair_value.py's output into a label per stock, a
plain-language explanation, and a few "look at this by eye" flags.

This is a DESCRIPTIVE tool: "at this date, this stock trades X% below/above
the multiple its fundamentals would normally command". It is not a return
forecast — the return-prediction version of this project found nothing
(0/60 significant after FDR, 2026-09-18), and fair_value.gap_return_test is
where the gap itself gets tested against later returns.

LABEL HISTORY (why it looks like this):
  - until 2026-09-23 AM: composite_score (mean of 11 sector percentiles)
    >= 70 / <= 30. Broken by design: an average of 11 percentiles clusters
    near 50, so ~93% of rows were "중립"; and it mixed quality/momentum into
    a "valuation" label (XRX was 고평가 with one of the cheapest PBRs in its
    sector).
  - 2026-09-23 midday: sector rank of PER/PBR/dividend-yield percentiles
    only. Fixed both problems, but still called every high-growth/high-ROE
    stock expensive.
  - now: cheapness_rank = percentile of -valuation_gap (fair_value.py)
    across all stocks at the same as_of. Top 20% = 저평가, bottom 20% =
    고평가 (config thresholds). The naive sector rank is still reported as
    sector_valuation_rank, so "expensive vs. sector but fair for its
    fundamentals" is visible side by side.

FLAGS (heuristics to prompt a second look, not verdicts):
  - meme_flag: |5-day move| > 15% AND last-day volume > 3 std above its 63d
    mean. Only sees the 5 days before each as_of, so between quarterly
    snapshots it misses events (e.g. GME in late Jan 2021); on the "today"
    snapshot it works as intended.
  - value_trap_flag: 저평가 for 4 consecutive snapshots AND revenue growth in
    the bottom 40% of its sector — cheap for a long time, possibly for a
    reason the model can't see.
  - transition_flag/type: snapshot where the label flipped 저평가 -> 고평가,
    attributed to price vs. EPS movement (classify_valuation_transition).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    CHEAP_THRESHOLD,
    EXPENSIVE_THRESHOLD,
    FAIR_VALUE_FEATURES,
    FAIR_VALUE_TARGETS,
    MOMENTUM_INDICATORS,
    QUALITY_INDICATORS,
    VALUATION_INDICATORS,
)
from fair_value import FEATURE_LABELS_KO, add_fair_value, loss_flag

PRICE_SPIKE_THRESHOLD = 0.15
VOLUME_ZSCORE_THRESHOLD = 3.0

VALUE_TRAP_LOOKBACK_PERIODS = 4
VALUE_TRAP_WEAK_GROWTH_PERCENTILE = 40.0

# |log change| between consecutive snapshots that counts as "this moved".
TRANSITION_PRICE_THRESHOLD = 0.15
TRANSITION_EARNINGS_THRESHOLD = 0.15

# Drivers smaller than this (log units, ~5%) are left out of explanations.
MIN_DRIVER_EFFECT = 0.05


def _pct_rank(values: pd.Series, by: pd.Series) -> pd.Series:
    """Percentile (0-100] of each value within its group; NaN if the value
    is missing or the group has fewer than 3 values."""
    grouped = values.groupby(by)
    counts = grouped.transform("count")
    return (100.0 * grouped.rank(method="max") / counts).where(counts >= 3)


def valuation_label(cheapness_rank: float) -> str:
    if pd.isna(cheapness_rank):
        return "데이터 부족"
    if cheapness_rank >= CHEAP_THRESHOLD:
        return "저평가"
    if cheapness_rank <= EXPENSIVE_THRESHOLD:
        return "고평가"
    return "중립"


def add_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """cheapness_rank + valuation_label from valuation_gap (loss-makers,
    fair_value.loss_flag, get "판단 보류(적자)" and no rank), plus context
    columns: sector_valuation_rank (naive multiple-vs-sector view),
    quality_score, momentum_score (means of sector percentiles)."""
    df = panel.copy()
    if "loss_flag" not in df.columns:
        df["loss_flag"] = loss_flag(df)

    rankable = df["valuation_gap"].where(~df["loss_flag"])
    df["cheapness_rank"] = _pct_rank(-rankable, df["as_of"])
    df["valuation_label"] = df["cheapness_rank"].apply(valuation_label)
    df.loc[df["loss_flag"], "valuation_label"] = "판단 보류(적자)"

    naive = df[[f"{k}_pct" for k in VALUATION_INDICATORS]].mean(axis=1, skipna=True)
    df["sector_valuation_rank"] = _pct_rank(naive, [df["as_of"], df["sector"]])
    df["quality_score"] = df[[f"{k}_pct" for k in QUALITY_INDICATORS]].mean(axis=1, skipna=True)
    df["momentum_score"] = df[[f"{k}_pct" for k in MOMENTUM_INDICATORS]].mean(axis=1, skipna=True)
    return df


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
    cheap = (df["valuation_label"] == "저평가").astype(int)
    streak = cheap.groupby(df["ticker"]).transform(lambda s: s.rolling(lookback_periods).sum())
    weak_growth = df["revenue_growth_yoy_pct"] <= VALUE_TRAP_WEAK_GROWTH_PERCENTILE
    df["value_trap_flag"] = ((streak == lookback_periods) & weak_growth).fillna(False).astype(bool)
    df["value_trap_reason"] = np.where(
        df["value_trap_flag"],
        f"최근 {lookback_periods}개 시점 연속 저평가 + 매출성장률은 섹터 하위 "
        f"{VALUE_TRAP_WEAK_GROWTH_PERCENTILE:.0f}% 이내 — 모델이 못 보는 이유로 계속 싼 것일 수 있음(밸류트랩 후보)",
        "",
    )
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
            if not (labels[i - 1] == "저평가" and labels[i] == "고평가"):
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


def _driver_text(row: pd.Series, key: str) -> str:
    """'ROE +22%, 매출성장률 +9% / 변동성 -12%' — how each fundamental moved
    this stock's fair multiple vs. the average stock at that date."""
    effects = {
        driver: row.get(f"{key}_contrib_{driver}", np.nan) for driver in [*FAIR_VALUE_FEATURES, "sector"]
    }
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
        return f"{FEATURE_LABELS_KO[driver]}{'(값 없음)' if missing else ''} {np.expm1(effects[driver]):+.0%}"

    parts = []
    if ups:
        parts.append("높인 요인 " + ", ".join(map(fmt, ups)))
    if downs:
        parts.append("낮춘 요인 " + ", ".join(map(fmt, downs)))
    return " / ".join(parts)


def explain(row: pd.Series) -> str:
    """One paragraph per stock: actual vs. fair multiple for each multiple
    that could be evaluated, the drivers behind each fair multiple, and why
    a multiple was skipped if it was."""
    lines = []
    for key, spec in FAIR_VALUE_TARGETS.items():
        actual, fair, gap = row.get(spec["column"]), row.get(f"fair_{key}"), row.get(f"{key}_gap")
        if pd.notna(gap):
            lines.append(
                f"{spec['label']} {actual:.1f}배 (적정 {fair:.1f}배, {np.expm1(gap):+.0%}) — 적정 {spec['label']}을 "
                f"{_driver_text(row, key)}"
            )
        elif pd.isna(actual) or actual <= 0:
            why = "적자라 계산 불가" if key == "pe" and row.get("loss_flag") else "값 없음(적자·자본잠식 또는 데이터 누락)"
            lines.append(f"{spec['label']}: {why}")
        elif actual > spec["max"]:
            base = "이익이 너무 작아" if key == "pe" else "자본이 너무 작아(자사주 매입 등)"
            lines.append(f"{spec['label']} {actual:.0f}배: {base} 배수로 비교하기 어려움")
        elif actual < spec["min"]:
            lines.append(f"{spec['label']} {actual:.2f}배: 비정상적으로 작은 값 — 데이터 오류 가능성")
        else:
            lines.append(f"{spec['label']}: 같은 시점 비교 종목 부족")
    if row.get("loss_flag"):
        lines.append("최근 12개월 적자 — 적정 배수 추정을 신뢰하기 어려워 판단을 보류함")
    return "\n".join(lines)


def screen(panel: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline on every row: fair value -> labels -> all flags.
    Accepts a raw panel (features.py output); fair value is recomputed here
    so a saved panel never carries a stale model."""
    df, _ = add_fair_value(panel)
    df = add_labels(df)
    df = flag_meme_stock(df)
    df = flag_value_trap(df)
    return classify_valuation_transition(df)


REPORT_COLUMNS = [
    "ticker", "sector", "as_of", "valuation_label", "cheapness_rank", "valuation_gap_pct", "valuation_basis",
    "trailing_pe", "fair_pe", "price_to_book", "fair_pb", "sector_valuation_rank", "quality_score",
    "momentum_score", "loss_flag", "meme_flag", "value_trap_flag", "transition_flag", "transition_type",
    "meme_reason", "value_trap_reason", "transition_reason", "explanation",
]


def report_at(screened: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per ticker at `as_of` (default: latest), cheapest first."""
    target = pd.Timestamp(as_of) if as_of is not None else screened["as_of"].max()
    rows = screened[screened["as_of"] == target].copy()
    if rows.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)
    rows["valuation_gap_pct"] = np.expm1(rows["valuation_gap"])
    rows["explanation"] = rows.apply(explain, axis=1)
    return rows[REPORT_COLUMNS].sort_values("cheapness_rank", ascending=False, na_position="last").reset_index(drop=True)


def screen_latest(panel: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    return report_at(screen(panel), as_of)


def screen_ticker(panel: pd.DataFrame, ticker: str, as_of: pd.Timestamp | None = None) -> dict:
    report = screen_latest(panel, as_of)
    match = report[report["ticker"] == ticker.upper()]
    if match.empty:
        return {"error": f"{ticker}에 대한 데이터가 없습니다 (유니버스에 없거나 해당 시점 데이터 부족)"}
    return match.iloc[0].to_dict()
