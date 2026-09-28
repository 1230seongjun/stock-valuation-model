"""
Shared settings for the fair-value (적정 밸류에이션) screening pipeline.

History that explains several choices below (details in README.md):
- 2026-09-14~18: built as a "predict forward returns from valuation
  factors" pipeline. On 281 tickers x 16 years, 0/60 factor tests survived
  FDR correction and no model beat a sector-average baseline, so that goal
  was dropped (2026-09-18).
- 2026-09-23: rebuilt around a fair-value model instead — learn how the
  market prices fundamentals (ROE, margins, growth, leverage, payout, risk)
  into PER/PBR (+PSR from 2026-09-28) at each point in time, and call a stock
  cheap/expensive relative to the multiples its OWN fundamentals would
  justify (fair_value.py).
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
    "price_to_sales": "lower_is_better",
    "ev_to_ebitda": "lower_is_better",
    "price_to_fcf": "lower_is_better",
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
VALUATION_INDICATORS = ["trailing_pe", "price_to_book", "price_to_sales", "ev_to_ebitda", "price_to_fcf", "dividend_yield"]
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
# PSR (added 2026-09-28): its max is loose because fast-growing software
# legitimately trades at 20-30x sales. It stays defined for loss-makers, but
# judging them on it alone was tried and reverted (fair_value.loss_flag).
# EV/EBITDA and P/FCF (added 2026-09-28): EV/EBITDA is capital-structure
# neutral and less exposed to one-off gains/charges than PER (HON's spin-off
# gain inflated net income, not EBITDA); P/FCF checks earnings against cash.
# EV/Sales (~PSR) and P/TBV (~PBR, 66 quarters, negative for goodwill-heavy
# firms) were left out so no single angle is counted twice.
FAIR_VALUE_TARGETS = {
    "pe": {"column": "trailing_pe", "min": 1.0, "max": 100.0, "label": "PER"},
    "pb": {"column": "price_to_book", "min": 0.1, "max": 20.0, "label": "PBR"},
    "ps": {"column": "price_to_sales", "min": 0.05, "max": 40.0, "label": "PSR"},
    "ev_ebitda": {"column": "ev_to_ebitda", "min": 1.0, "max": 60.0, "label": "EV/EBITDA"},
    "pfcf": {"column": "price_to_fcf", "min": 1.0, "max": 100.0, "label": "P/FCF"},
}

# Finnhub's quarterly multiples are computed at the fiscal period-end price,
# but a snapshot is 45-135 days later. When True, features.build_raw_panel
# rescales them to the snapshot's own price (reported values kept as
# *_reported): price multiples by price(as_of) / price(period end); EV/EBITDA
# by rescaling only the market-cap part of EV (net debt doesn't move with the
# price). Confirmed against yfinance's current values: 2026-09-23 PER/PBR
# (AAPL rescaled 38.4 / 46.0 vs. 38.9 / 46.2; unrescaled 32.1 / 38.5),
# 2026-09-28 PSR (10.60 vs. 10.66). `python src/main.py verify-multiples`
# re-checks all of them.
RESCALE_MULTIPLES_TO_AS_OF_PRICE = True

# Fundamentals that should justify a higher/lower multiple. Price-derived
# signals (momentum, 52w-high distance) are deliberately excluded: a recent
# price drop lowers the multiple AND moves those signals, so the model would
# "explain away" exactly the cheapness we want to measure. volatility_63d is
# the one exception, as the only available risk proxy (discount rate).
# 2026-09-23 prototype on the real panel: adding eps growth YoY or widening
# the training window to 4 quarters changed out-of-fold R^2 by < 0.01.
# Model family, re-checked 2026-09-28 with XGBoost on the exact same setup
# (bounded multiples, same folds; out-of-fold R^2 train/val/test):
#   PBR  Ridge 0.47/0.52/0.55   XGB depth-2 0.54/0.54/0.57
#   PER  Ridge 0.27/0.31/0.28   XGB depth-2 0.35/0.30/0.31
# XGB gains mostly in Train and ~0 on Val (the period model choices are made
# on), with in-sample R^2 0.84 vs 0.54 out-of-fold (overfitting). Ridge stays:
# same Val performance, stable economically-signed coefficients. Revisit if
# the feature set grows (nonlinear models gain more with more features).
FAIR_VALUE_FEATURES = [
    "return_on_equity",
    "operating_margin",
    "revenue_growth_yoy",
    "debt_to_equity",
    "payout_ratio_ttm",
    "volatility_63d",
]

# Added 2026-09-28 (Finnhub statement ratios + trend + dividend history),
# NOT in the model until `python src/main.py compare-features` shows they
# help on Train/Val. All are free of the snapshot's price, except that none
# of the dividend ones is dividend_yield (a price drop raises the yield, the
# same problem as momentum). dividend_growth_3y / dividend_years_no_cut
# capture the premium the market pays for a reliable, growing dividend.
FAIR_VALUE_FEATURE_CANDIDATES = [
    "roic",
    "gross_margin",
    "fcf_margin",
    "net_debt_to_capital",
    "current_ratio",
    "asset_turnover",
    "sga_to_sales",
    "revenue_cagr_3y",
    "op_margin_volatility",
    "dividend_growth_3y",
    "dividend_years_no_cut",
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
