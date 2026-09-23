"""
Shared settings for the fair-value (적정 밸류에이션) screening pipeline.

History that explains several choices below (details in README.md):
- 2026-09-14~18: built as a "predict forward returns from valuation
  factors" pipeline. On 281 tickers x 16 years, 0/60 factor tests survived
  FDR correction and no model beat a sector-average baseline, so that goal
  was dropped (2026-09-18).
- 2026-09-23: rebuilt around a fair-value model instead — learn how the
  market prices fundamentals (ROE, margins, growth, leverage, payout, risk)
  into PER/PBR at each point in time, and call a stock cheap/expensive
  relative to the multiple its OWN fundamentals would justify (fair_value.py).
"""
import pandas as pd

# ---- Time split --------------------------------------------------------
# Fixed calendar boundaries, never a random split and never fractions of
# however many dates happen to exist. The fair-value model itself only ever
# looks at one as_of cross-section at a time (no future rows), but every
# modelling choice (features, alpha grid, thresholds) is made by looking at
# Train/Val results only; Test is for the final report.
#   Train: 2004-01-01 ~ 2019-12-31  (16 years)
#   Val:   2020-01-01 ~ 2023-12-31  (4 years)
#   Test:  2024-01-01 ~ today
TRAIN_START = pd.Timestamp("2004-01-01")
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END = pd.Timestamp("2023-12-31")

# Quarterly as_of snapshots (pandas alias, plugs into pd.date_range). The
# build step also appends today's date so screening has a current snapshot.
REBALANCE_FREQ = "QS"

# Forward-return horizons. Not used by the fair-value model — only by
# fair_value.gap_return_test, which checks whether the model's "cheap" call
# has any relationship with later returns (a hypothesis test, not a feature).
HORIZONS_MONTHS = [1, 3, 6, 12]

RANDOM_SEED = 42

# ---- Indicators ----------------------------------------------------------
# metric_key -> direction used when turning a raw value into a sector
# percentile (features.add_percentile_scores). These percentiles are CONTEXT
# columns in the screening report; the fair-value model uses raw values.
FUNDAMENTAL_INDICATORS = {
    "trailing_pe": "lower_is_better",
    "price_to_book": "lower_is_better",
    "dividend_yield": "higher_is_better",
    "return_on_equity": "higher_is_better",
    "debt_to_equity": "lower_is_better",
    "operating_margin": "higher_is_better",
    "revenue_growth_yoy": "higher_is_better",
}

TECHNICAL_INDICATORS = {
    "ma50_vs_ma200": "higher_is_better",     # 50d MA / 200d MA - 1
    "pct_from_52w_high": "lower_is_better",  # (52w high - price) / 52w high
    "volatility_63d": "lower_is_better",     # annualized realized vol
    "relative_volume": "higher_is_better",   # 21d avg volume / 63d avg volume
}

ALL_INDICATORS = {**FUNDAMENTAL_INDICATORS, **TECHNICAL_INDICATORS}

# Context groups shown next to the fair-value label in the screening report.
# sector_valuation_rank is the naive "multiple vs. sector peers" view (what
# the label was before the fair-value model) — kept so the report can show
# "expensive vs. sector, but fair given its fundamentals" side by side.
VALUATION_INDICATORS = ["trailing_pe", "price_to_book", "dividend_yield"]
QUALITY_INDICATORS = ["return_on_equity", "debt_to_equity", "operating_margin", "revenue_growth_yoy"]
MOMENTUM_INDICATORS = ["ma50_vs_ma200", "pct_from_52w_high"]

# ---- Fair-value model (fair_value.py) -------------------------------------
# Multiples the model learns to explain. Rows outside [min, max] are neither
# trained on nor given a gap for that multiple:
#   - missing / non-positive: losses or negative equity (undefined multiple)
#   - above max: earnings or book so close to zero that the ratio stops
#     describing valuation (PER 15,000; PBR 40 after years of buybacks —
#     CLX/CL/KMB/HD/AAPL spend long stretches above PBR 20)
#   - below min: in this dataset only ever data errors (BNY's Finnhub series
#     reads PER 0.04 / PBR 0.006 for its whole history)
FAIR_VALUE_TARGETS = {
    "pe": {"column": "trailing_pe", "min": 1.0, "max": 100.0, "label": "PER"},
    "pb": {"column": "price_to_book", "min": 0.1, "max": 20.0, "label": "PBR"},
}

# Finnhub's quarterly peTTM / pb are computed at the fiscal period-end price,
# but a snapshot is 45-135 days later. When True, features.build_raw_panel
# rescales both by price(as_of) / price(period end) so they reflect the
# snapshot's own price (the reported values are kept as *_reported).
# `python src/main.py verify-multiples` checks this assumption against
# Finnhub's current-price metrics.
RESCALE_MULTIPLES_TO_AS_OF_PRICE = True

# Fundamentals that should justify a higher/lower multiple. Price-derived
# signals (momentum, 52w-high distance) are deliberately excluded: a recent
# price drop lowers the multiple AND moves those signals, so the model would
# "explain away" exactly the cheapness we want to measure. volatility_63d is
# the one exception, as the only available risk proxy (discount rate).
# 2026-09-23 prototype on the real panel: adding eps growth YoY or widening
# the training window to 4 quarters changed out-of-fold R^2 by < 0.01, and
# gradient boosting did not beat Ridge, so this stays small and linear.
FAIR_VALUE_FEATURES = [
    "return_on_equity",
    "operating_margin",
    "revenue_growth_yoy",
    "debt_to_equity",
    "payout_ratio_ttm",
    "volatility_63d",
]

FAIR_VALUE_CV_FOLDS = 5             # out-of-fold by ticker within each as_of
FAIR_VALUE_WINSOR_QUANTILE = 0.02   # clip each feature to [2%, 98%] per as_of
FAIR_VALUE_MIN_ROWS = 50            # skip an as_of with fewer usable rows
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

# ---- Screening labels (screening.py) ---------------------------------------
# cheapness_rank = percentile (0-100) of -valuation_gap across all stocks at
# the same as_of; higher = cheaper relative to its own fair value.
CHEAP_THRESHOLD = 80.0
EXPENSIVE_THRESHOLD = 20.0
