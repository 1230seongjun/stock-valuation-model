"""
Relative valuation screening — the actual deliverable after the 2026-09-18
pivot away from "predict returns" toward "rank stocks by how cheap/expensive
they look relative to their own sector right now."

WHY THIS EXISTS (read this before using the output): the regression models
(models.py's Ridge/XGBoost/RandomForest) and the factor-validity tests
(factor_validation.py) were built to answer "does composite_score predict
FUTURE returns" — and on 281 tickers / 16 years, the honest answer is no,
not at a level that survives multiple-testing correction (0/60 significant
after FDR, see chat history / technical-spec.md's research story). This
module does NOT try to predict anything. It answers a narrower, more
defensible question: "as of today, where does this stock's valuation sit
relative to its sector peers' history?" — a descriptive ranking, not a
forecast. Don't let the confident-looking labels below imply a return
prediction that the earlier work explicitly could not establish.

Two extra safeguards on top of the plain composite_score ranking, matching
the 2026-09-18 discussion (밈주식이 갑자기 고평가되는 경우 /
저평가인데 평생 못 오르는 밸류트랩 둘 다 걸러야 한다는 요구):

1. meme_flag — a sudden, large price move (features._anomaly_signals_asof's
   price_spike_5d) alongside abnormal volume (volume_zscore_63d) means the
   CURRENT snapshot's percentile score is unreliable — the stock may look
   "cheap" or "expensive" simply because its price just detached from
   whatever normal fundamentals-driven range it was in. This flags the row
   as low-confidence rather than silently including it as a normal read.

2. value_trap_flag — a stock sitting in the bottom of its sector's
   valuation percentile (i.e. scoring as "attractively cheap") for many
   consecutive quarters AND showing weak/declining revenue growth relative
   to sector peers is a candidate value trap: cheap because the market
   correctly expects continued underperformance, not because it's
   mispriced. This is a heuristic, not a proof — see the docstring on
   flag_value_trap for exactly what it checks and its limits.

Neither flag is a claim that this pipeline can tell a real mispricing from
a real value trap or a real meme-stock top — nobody can do that reliably,
including professional analysts (see the 2026-09-18 chat discussion on why
this specific ask can't be "solved," only mitigated). They exist to
surface the cases worth a second look by eye, not to auto-resolve them.

3. transition_flag / transition_type — added 2026-09-23, after a full
   codebase re-audit turned up no bugs behind the earlier weak factor-
   validity results (see chat history). A "저평가 -> 고평가" flip in
   valuation_label looks the same regardless of WHY it happened, but the
   two common causes mean close to opposite things:
     - price rose a lot while earnings held up -> the market re-rated the
       stock (could be a legitimate catch-up, could mean it's now actually
       expensive going forward)
     - price didn't move (or fell) but EPS dropped -> PER/PBR rose
       mechanically because the denominator shrank, not because anyone
       decided this is a hot stock. This is a weaker, more cautionary
       signal than the first case, and lumping both under "고평가" hides
       that difference.
   classify_valuation_transition() decomposes this using each ticker's own
   raw price/eps change between the flip's two snapshots — see its
   docstring for the exact rule and its real limitations (snapshot-to-
   snapshot only, can't see sector-relative-only moves, composite_score
   mixes in technical/momentum indicators so "고평가" here isn't a pure
   value-multiple statement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- valuation_label thresholds on composite_score (0-100 scale, higher =
# more attractive on the equal-weighted composite — see features.py) ----
CHEAP_THRESHOLD = 70.0
EXPENSIVE_THRESHOLD = 30.0

# ---- meme_flag thresholds ----
PRICE_SPIKE_THRESHOLD = 0.15  # |5-trading-day cumulative log return| > 15%
VOLUME_ZSCORE_THRESHOLD = 3.0  # most recent day's volume > 3 std devs above its own 63d mean

# ---- value_trap_flag thresholds ----
VALUE_TRAP_LOOKBACK_PERIODS = 4  # consecutive as_of snapshots (≈1y at quarterly rebalance)
# composite_score direction: HIGHER = cheaper/more attractive (lower_is_better
# raw indicators like PER/PBR are inverted before averaging — see
# features.add_percentile_scores). So "persistently cheap" means
# composite_score AT OR ABOVE this, same cutoff as valuation_label's
# CHEAP_THRESHOLD for consistency — a value-trap candidate is exactly a
# stock that would otherwise get labeled "저평가" every single quarter.
VALUE_TRAP_CHEAP_PERCENTILE = CHEAP_THRESHOLD
VALUE_TRAP_WEAK_GROWTH_PERCENTILE = 40.0  # revenue_growth_yoy_pct below this = weak vs. sector peers

# ---- transition_type thresholds (see classify_valuation_transition) ----
# Both are |log change| cutoffs between one snapshot and the next (quarterly
# at the default REBALANCE_FREQ) — 15% matches meme_flag's PRICE_SPIKE_
# THRESHOLD for the price side, kept equal for the EPS side too since
# there's no principled reason a swing in earnings should need a different
# bar than the same-size swing in price to count as "the thing that moved."
TRANSITION_PRICE_THRESHOLD = 0.15
TRANSITION_EARNINGS_THRESHOLD = 0.15


def valuation_label(composite_score: float) -> str:
    """Plain-language bucket for a composite_score value. NOT a return
    forecast — see module docstring."""
    if pd.isna(composite_score):
        return "데이터 부족"
    if composite_score >= CHEAP_THRESHOLD:
        return "저평가"
    if composite_score <= EXPENSIVE_THRESHOLD:
        return "고평가"
    return "중립"


def flag_meme_stock(panel: pd.DataFrame) -> pd.DataFrame:
    """Adds meme_flag (bool) + meme_reason (str) columns. True when BOTH a
    large short-term price move AND an abnormal volume spike happened as of
    that row's as_of date — either alone is common (earnings-day moves,
    seasonal volume patterns); both together is the "something detached
    from normal trading happened here" pattern. Requires
    features._anomaly_signals_asof's columns to be present."""
    df = panel.copy()
    has_signals = {"price_spike_5d", "volume_zscore_63d"}.issubset(df.columns)
    if not has_signals:
        df["meme_flag"] = False
        df["meme_reason"] = ""
        return df

    spike = df["price_spike_5d"].abs() > PRICE_SPIKE_THRESHOLD
    vol_spike = df["volume_zscore_63d"] > VOLUME_ZSCORE_THRESHOLD
    df["meme_flag"] = (spike & vol_spike).fillna(False)

    def _reason(row: pd.Series) -> str:
        if not row["meme_flag"]:
            return ""
        direction = "급등" if row["price_spike_5d"] > 0 else "급락"
        return (
            f"최근 5거래일 {direction} ({row['price_spike_5d']:+.1%}) + "
            f"거래량 이상치(63일 평균 대비 {row['volume_zscore_63d']:.1f}표준편차) "
            "— 밸류에이션 스코어 신뢰도 낮음"
        )

    df["meme_reason"] = df.apply(_reason, axis=1)
    return df


def flag_value_trap(
    panel: pd.DataFrame,
    lookback_periods: int = VALUE_TRAP_LOOKBACK_PERIODS,
    cheap_percentile: float = VALUE_TRAP_CHEAP_PERCENTILE,
    weak_growth_percentile: float = VALUE_TRAP_WEAK_GROWTH_PERCENTILE,
) -> pd.DataFrame:
    """Adds value_trap_flag (bool) + value_trap_reason (str) columns.

    Heuristic, not a statistical test (unlike factor_validation.py) —
    "cheap for a long time + weak relative growth" is a commonly cited
    value-trap pattern, not something this pipeline has validated as
    predictive. Specifically: for each ticker, look at its last
    `lookback_periods` as_of rows (sorted by date, using only data already
    in `panel` — no future rows are read, so this respects the same
    point-in-time discipline as everything else); flag True only if EVERY
    one of those rows had composite_score >= cheap_percentile (i.e. would
    have been labeled "저평가" every single quarter — composite_score's
    direction is HIGHER = cheaper, see valuation_label) AND the most recent
    row's revenue_growth_yoy_pct <= weak_growth_percentile. A stock that
    just became cheap this quarter (interesting — might be a fresh
    opportunity) is deliberately NOT flagged; only sustained cheapness with
    weak relative growth is.

    Needs at least `lookback_periods` rows of history for a ticker to flag
    it at all — a ticker with a shorter history (recent IPO, e.g. TSLA/META
    early in their listed life) simply can't be evaluated for "sustained"
    cheapness yet and is left unflagged, not defaulted to True or False by
    assumption.
    """
    df = panel.sort_values(["ticker", "as_of"]).copy()
    df["value_trap_flag"] = False
    df["value_trap_reason"] = ""

    if "composite_score" not in df.columns:
        return df

    for ticker, group in df.groupby("ticker"):
        if len(group) < lookback_periods:
            continue
        window = group.tail(lookback_periods)
        persistently_cheap = (window["composite_score"] >= cheap_percentile).all()
        if not persistently_cheap:
            continue
        latest_growth = window["revenue_growth_yoy_pct"].iloc[-1]
        weak_growth = pd.notna(latest_growth) and latest_growth <= weak_growth_percentile
        if not weak_growth:
            continue

        latest_idx = window.index[-1]
        df.loc[latest_idx, "value_trap_flag"] = True
        df.loc[latest_idx, "value_trap_reason"] = (
            f"composite_score가 최근 {lookback_periods}개 분기 연속 상위 "
            f"{cheap_percentile:.0f}퍼센타일 이상(=계속 저평가) + 매출성장률은 섹터 내 하위 "
            f"{weak_growth_percentile:.0f}퍼센타일 — 구조적 저평가(밸류트랩) 가능성, "
            "단순 저평가와 구분해서 볼 것"
        )
    return df


def classify_valuation_transition(
    panel: pd.DataFrame,
    price_threshold: float = TRANSITION_PRICE_THRESHOLD,
    earnings_threshold: float = TRANSITION_EARNINGS_THRESHOLD,
) -> pd.DataFrame:
    """Adds transition_flag (bool) + transition_type (str) + transition_reason
    (str) columns, flagging the snapshot where a ticker's valuation_label
    flipped directly from "저평가" to "고평가" against the immediately
    preceding snapshot in `panel`, and attributing WHY using that ticker's
    own raw price/eps change over the same gap (module docstring has the
    motivation — the same label flip can mean opposite things).

    Method: for each ticker, sort by as_of and walk consecutive snapshot
    pairs. Where valuation_label goes 저평가 -> 고평가, compute:
      price_log_change  = ln(price_t / price_{t-1})
      eps_log_change    = ln(eps_t / eps_{t-1})
    (each left NaN if either input is missing/non-positive — ln is undefined
    for a loss-making eps or a non-positive price, and undefined beats a
    wrong sign) and classify:
      - price_log_change > +price_threshold AND eps didn't drop past
        -earnings_threshold -> "주가 상승형" (재평가/모멘텀: 이익은 버티는데
        주가만 뛰어서 벨류에이션이 비싸짐)
      - eps_log_change < -earnings_threshold AND price didn't rise past
        +price_threshold -> "실적 악화형" (주가는 그대로인데 EPS가 꺾여서
        PER 등 배수가 기계적으로 올라간 것 — 시장의 재평가라기보다 이익
        훼손 쪽에 가까움)
      - both conditions true at once -> "복합형" (두 효과가 겹침 — 재평가로
        읽으면 안 됨)
      - price_log_change < -price_threshold AND eps held up -> "가격
        급락형": composite_score는 value+quality+technical 11개 지표
        평균이라(config.ALL_INDICATORS), 주가 급락은 PER/PBR을 오히려
        싸게 만들면서도 momentum계열 지표(ma50_vs_ma200,
        pct_from_52w_high)를 크게 깎아 종합점수를 끌어내릴 수 있다 —
        이 경우 "고평가"는 밸류에이션이 아니라 추세 훼손을 반영한 것일 수
        있으므로 반드시 별도로 표시한다.
      - none of the above clears its threshold (or price/eps data is
        missing) -> "불분명" — 흔한 원인 두 개고, 데이터가 있는데도
        전부 threshold 밑이면 이 종목 자체보다 같은 섹터 내 다른 종목들의
        밸류에이션이 움직여 상대 퍼센타일이 바뀌었을 가능성이 크다는
        점을 reason에 적어 둔다 (percentile은 섹터+as_of 횡단면 상대
        지표라 이 함수가 보는 종목 자체의 price/eps만으로는 못 잡는 경로).

    LIMITATIONS (heuristic, not causal — same spirit as flag_value_trap):
    - snapshot-to-snapshot만 봄. 두 스냅샷 사이에 저평가↔고평가를 여러 번
      오갔어도 마지막 전환 한 번만 잡힘 (value_trap과 동일한 한계).
    - price/eps 데이터가 하나라도 없거나 부호가 바뀌는 구간(적자→흑자 등)은
      로그 변화율 정의가 안 돼 "불분명"으로 빠짐 — 실제로 원인이 없다는
      뜻이 아니라 이 방법으로는 못 잡는다는 뜻.
    - "고평가"가 정말 value 배수(PER/PBR) 때문인지 momentum 지표 때문인지는
      "가격 급락형" 케이스에서만 명시적으로 구분함 — 그 외 케이스에서도
      technical 지표가 일부 기여했을 수 있다는 점은 여전히 남아 있음
      (config.py의 VALUE_COMPOSITE_INDICATORS 진단용 컬럼 참고).
    """
    df = panel.sort_values(["ticker", "as_of"]).copy()
    df["valuation_label"] = df["composite_score"].apply(valuation_label)
    df["transition_flag"] = False
    df["transition_type"] = ""
    df["transition_reason"] = ""

    if not {"price", "eps"}.issubset(df.columns):
        return df

    for ticker, group in df.groupby("ticker"):
        if len(group) < 2:
            continue
        idx = group.index.to_numpy()
        labels = group["valuation_label"].to_numpy()
        prices = group["price"].to_numpy(dtype=float)
        eps_vals = group["eps"].to_numpy(dtype=float)

        for i in range(1, len(group)):
            if not (labels[i - 1] == "저평가" and labels[i] == "고평가"):
                continue

            price_prev, price_cur = prices[i - 1], prices[i]
            eps_prev, eps_cur = eps_vals[i - 1], eps_vals[i]

            price_log_change = np.nan
            if pd.notna(price_prev) and pd.notna(price_cur) and price_prev > 0 and price_cur > 0:
                price_log_change = float(np.log(price_cur / price_prev))

            eps_log_change = np.nan
            if pd.notna(eps_prev) and pd.notna(eps_cur) and eps_prev > 0 and eps_cur > 0:
                eps_log_change = float(np.log(eps_cur / eps_prev))

            price_rose = pd.notna(price_log_change) and price_log_change > price_threshold
            price_crashed = pd.notna(price_log_change) and price_log_change < -price_threshold
            earnings_fell = pd.notna(eps_log_change) and eps_log_change < -earnings_threshold

            price_str = f"{price_log_change:+.1%}" if pd.notna(price_log_change) else "데이터 부족"
            eps_str = f"{eps_log_change:+.1%}" if pd.notna(eps_log_change) else "데이터 부족"

            if price_rose and earnings_fell:
                t_type = "복합형"
                reason = (
                    f"주가 {price_str} 상승과 EPS {eps_str} 하락이 동시에 발생 — "
                    "재평가(re-rating)와 실적 악화가 겹쳐 있어 단순히 '시장이 비싸게 본다'로 "
                    "읽으면 안 됨"
                )
            elif price_rose:
                t_type = "주가 상승형"
                reason = f"주가가 {price_str} 오르면서 저평가 구간을 벗어남 (EPS는 {eps_str})"
            elif earnings_fell:
                t_type = "실적 악화형"
                reason = (
                    f"EPS가 {eps_str} 하락하며 PER 등 배수가 기계적으로 상승 (주가는 {price_str}) — "
                    "시장의 재평가라기보다 이익 훼손 쪽에 가까울 수 있음"
                )
            elif price_crashed:
                t_type = "가격 급락형"
                reason = (
                    f"주가가 {price_str} 급락 (EPS는 {eps_str}) — composite_score는 value뿐 아니라 "
                    "모멘텀/기술적 지표도 포함하므로, '고평가' 전환이 밸류에이션이 아니라 "
                    "추세 훼손을 반영한 것일 수 있음"
                )
            else:
                t_type = "불분명"
                reason = (
                    f"이 종목 자체의 주가({price_str})·EPS({eps_str}) 변화는 뚜렷하지 않음 — "
                    "섹터 내 다른 종목들의 밸류에이션 변화로 상대 퍼센타일만 바뀌었을 가능성 "
                    "(또는 데이터 부족)"
                )

            row_idx = idx[i]
            df.loc[row_idx, "transition_flag"] = True
            df.loc[row_idx, "transition_type"] = t_type
            df.loc[row_idx, "transition_reason"] = reason

    return df


def screen_latest(panel: pd.DataFrame, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Main entry point: one row per ticker at the most recent as_of date in
    `panel` (or a specific `as_of` if given), sorted by composite_score
    descending (most "attractive" first), with valuation_label + both
    anomaly flags applied.

    Typical use (in Colab, after run_real_data.py already saved the panel):
        import pandas as pd
        panel = pd.read_parquet("real_data_output/panel.parquet")
        import screening
        report = screening.screen_latest(panel)
        report[report["meme_flag"] | report["value_trap_flag"]]  # cases needing a second look
        report[report["transition_flag"]]  # 저평가->고평가 전환 원인(주가 상승형/실적 악화형/...)
    """
    target_date = as_of or panel["as_of"].max()

    flagged = flag_meme_stock(panel)
    flagged = flag_value_trap(flagged)
    # classify_valuation_transition needs the WHOLE panel (it walks each
    # ticker's full as_of history to find the flip), so it's run last, on
    # `flagged` (which already carries meme/value-trap columns) rather than
    # on the already-filtered `latest` below. It also (re)adds
    # valuation_label — same values flag_meme_stock/flag_value_trap don't
    # touch, so this doesn't overwrite anything from those two.
    flagged = classify_valuation_transition(flagged)

    latest = flagged[flagged["as_of"] == target_date].copy()

    cols = [
        "ticker", "sector", "as_of", "composite_score", "valuation_label",
        "meme_flag", "meme_reason", "value_trap_flag", "value_trap_reason",
        "transition_flag", "transition_type", "transition_reason",
    ]
    cols = [c for c in cols if c in latest.columns]
    return latest[cols].sort_values("composite_score", ascending=False).reset_index(drop=True)


def screen_ticker(panel: pd.DataFrame, ticker: str, as_of: pd.Timestamp | None = None) -> dict:
    """Single-ticker lookup (the original design-overview.md "온디맨드 1종목
    조회" use case) — same logic as screen_latest, filtered to one ticker,
    returned as a dict instead of a DataFrame row for easier printing."""
    report = screen_latest(panel, as_of=as_of)
    match = report[report["ticker"] == ticker.upper()]
    if match.empty:
        return {"error": f"{ticker}에 대한 데이터가 없습니다 (유니버스에 없거나 해당 시점 데이터 부족)"}
    return match.iloc[0].to_dict()


if __name__ == "__main__":
    import sys

    panel_path = sys.argv[1] if len(sys.argv) > 1 else "real_data_output/panel.parquet"
    panel = pd.read_parquet(panel_path)
    report = screen_latest(panel)
    print(f"Screening report as of {report['as_of'].iloc[0].date() if len(report) else 'N/A'} "
          f"({len(report)} tickers)\n")
    drop_cols = [c for c in ("meme_reason", "value_trap_reason", "transition_reason") if c in report.columns]
    print(report.drop(columns=drop_cols).to_string(index=False))

    flagged = report[report["meme_flag"] | report["value_trap_flag"]]
    if not flagged.empty:
        print(f"\n{len(flagged)} tickers flagged for a second look:")
        for _, row in flagged.iterrows():
            reason = row["meme_reason"] or row["value_trap_reason"]
            print(f"  {row['ticker']:6s} ({row['sector']}): {reason}")

    if "transition_flag" in report.columns:
        transitioned = report[report["transition_flag"]]
        if not transitioned.empty:
            print(f"\n{len(transitioned)} tickers flipped 저평가->고평가 this snapshot:")
            for _, row in transitioned.iterrows():
                print(f"  {row['ticker']:6s} ({row['sector']}) [{row['transition_type']}]: {row['transition_reason']}")
