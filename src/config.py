"""
Shared config for the valuation-factor expected-return pipeline.

Mirrors decisions recorded in technical-spec.md:
- Indicators: value + quality (fundamentals) + a small technical/volume set.
  RSI/MACD/Bollinger explicitly excluded (horizon mismatch, overfitting risk
  on a small universe) — see technical-spec.md section 2.
- Horizons: 1/3/6/12 month forward return.
- Models: Ridge (primary, interpretable) vs XGBoost/LightGBM (nonlinearity
  check) vs RandomForest (overfitting cross-check). No neural nets.
- Train/Val/Test date boundaries + rebalance frequency: decided with the
  user 2026-09-14, alongside the ticker universe (universe.py).
  Widened the same day (16y/4y/2y, from an original 4y/1y/~3y) after the
  first real run found 0/52 statistically significant factor tests — more
  usable quarters gives Fama-MacBeth/IC more periods to work with, which is
  the other lever (besides more tickers) for real statistical power.
"""
import pandas as pd

HORIZONS_MONTHS = [1, 3, 6, 12]

# Train/Validation/Test split — fixed calendar dates, not fractions of
# however many as_of dates happen to exist (technical-spec.md section 5:
# never a random split, and the boundary itself should be a decided value,
# not an accident of how much data got collected).
#   Train: 2004-01-01 ~ 2019-12-31  (16 years)
#   Val:   2020-01-01 ~ 2023-12-31  (4 years)
#   Test:  2024-01-01 ~ whenever the pipeline is actually run ("현재", ~2 years so far)
# pipeline.time_split only needs train_end/val_end (it takes "everything
# after val_end" as test), but TEST_START/VAL_START are kept here too since
# they're what actually got decided, not just derived.
#
# NOTE: this reaches back much further than the original 2018 start, so
# run_real_data.py's yfinance price fetch has to request enough history to
# actually cover TRAIN_START (a handful of tickers in universe.py, e.g.
# GOOGL/ABBV, IPO'd or spun off after this date anyway — see universe.py's
# docstring, that's an accepted, handled gap, not a bug to fix here).
TRAIN_START = pd.Timestamp("2004-01-01")
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_START = pd.Timestamp("2020-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

# Quarterly rebalance / as_of snapshot frequency. Matches pandas' own alias
# so it plugs straight into pd.date_range(..., freq=REBALANCE_FREQ).
REBALANCE_FREQ = "QS"

# metric_key -> "higher_is_better" | "lower_is_better"
# (direction used when turning a raw value into a percentile score)
FUNDAMENTAL_INDICATORS = {
    "trailing_pe": "lower_is_better",       # value
    "price_to_book": "lower_is_better",     # value
    "dividend_yield": "higher_is_better",   # value
    "return_on_equity": "higher_is_better", # quality
    "debt_to_equity": "lower_is_better",    # quality
    "operating_margin": "higher_is_better", # quality
    "revenue_growth_yoy": "higher_is_better",  # growth
}

TECHNICAL_INDICATORS = {
    "ma50_vs_ma200": "higher_is_better",     # trend: 50d MA / 200d MA - 1
    "pct_from_52w_high": "lower_is_better",  # how far below 52w high (smaller gap = stronger)
    "volatility_63d": "lower_is_better",     # realized vol, lower = less risky
    "relative_volume": "higher_is_better",   # recent volume / its own 63d average
}

# subset of ALL_INDICATORS used to build a second, "value-only" composite
# score (features.add_percentile_scores' value_composite_score column).
#
# Added 2026-09-18 after the first full real-data run (66 tickers, 2004-2019
# Train): composite_score's mean IC came back NEGATIVE at all 4 horizons,
# but the per-indicator univariate Fama-MacBeth coefficients showed this was
# NOT uniform across indicators — trailing_pe_pct/price_to_book_pct (value)
# and ma50_vs_ma200_pct (momentum) had the theoretically expected POSITIVE
# sign at longer horizons, while operating_margin_pct (most significant
# single indicator, p=0.0049 at 12m pre-FDR) and dividend_yield_pct
# (p=0.0091 at 12m pre-FDR) were consistently NEGATIVE. Since composite_score
# equal-weights all 11 indicators, the quality/dividend group may be
# canceling out the value/momentum group's signal. This subset isolates
# just the correctly-signed group to check whether IC/quantile_spread look
# different (better OR still insignificant) without that cancellation —
# purely a diagnostic composite, not a replacement for composite_score.
VALUE_COMPOSITE_INDICATORS = ["trailing_pe", "price_to_book", "ma50_vs_ma200"]

ALL_INDICATORS = {**FUNDAMENTAL_INDICATORS, **TECHNICAL_INDICATORS}

RANDOM_SEED = 42
