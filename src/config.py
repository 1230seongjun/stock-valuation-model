"""
Shared settings of the fair-value (적정 밸류에이션) screening pipeline.
Why each choice was made, with the numbers, is in README.md (실험 이력) and the
git history; comments here only say what a setting does.
"""
import pandas as pd

# ---- Time split: fixed dates, never random -------------------------------
# Choices (features, alphas, thresholds) are made on Train/Val only; Test is
# for the final report. The model itself fits one as_of cross-section at a time.
TRAIN_START = pd.Timestamp("2004-01-01")
TRAIN_END = pd.Timestamp("2019-12-31")
VAL_END = pd.Timestamp("2023-12-31")
# Design frozen 2026-10-06. Train/Val/Test have all been looked at while choosing
# features, so snapshots from this date on are the untouched hold-out ("sealed"):
# no design choice may use them, and they are evaluated once a year at most.
# 2026-10-08 (all rejected, README 실험 이력): standardized gap ranking, loss-maker
# features, net margin, a direct market-cap model, buyback/M&A features, a
# rule-based financial-health grade. Point-in-time S&P 500 membership: gains vanish
# once the own multiple and market-cap rank two years earlier are controlled for.
# Pre-registered for the first sealed evaluation (2026-10-08, judged once, not re-tried
# before then): ix_size_margin = (log_revenue - that date's median log_revenue) x
# (operating_margin's percentile that date - 0.5), entered untransformed into every
# in-verdict multiple. Explains the mega-cap premium if, on the sealed dates, (a) every
# in-verdict multiple's R^2 moves by >= -0.002 and (b) the top-10-by-market-cap premium
# (median gap of the top 10 minus that of large caps outside the top 50) is at most half
# the current model's. A model change needs also +0.01 R^2 on some multiple and the user's
# decision. (On Val it cut the premium 34% -> 21%, PSR R^2 +0.005 Train / +0.010 Val.)
SEALED_TEST_START = pd.Timestamp("2027-01-01")

REBALANCE_FREQ = "QS"            # quarterly snapshots (+ today's date)
HORIZONS_MONTHS = [1, 3, 6, 12]  # forward returns: labels for gap_return_test only, never features
N_JOBS = -1                      # joblib processes (-1 = all cores); results don't depend on it

# ---- Indicators (sector percentiles = report context, not model inputs) ---
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
VALUATION_INDICATORS = ["trailing_pe", "price_to_book", "price_to_sales", "ev_to_ebitda", "price_to_fcf", "dividend_yield"]
QUALITY_INDICATORS = ["return_on_equity", "debt_to_equity", "operating_margin", "revenue_growth_yoy"]
MOMENTUM_INDICATORS = ["ma50_vs_ma200", "pct_from_52w_high"]

# ---- Universe ---------------------------------------------------------------
UNIVERSE_INDEXES = ("sp500", "sp400", "sp600")  # on top of the 282-name core list (universe.py)
INDUSTRY_MIN_TICKERS = 20              # a GICS sub-industry with fewer members falls back to "<sector> 기타"

# ---- Fair-value model (fair_value.py) -------------------------------------
# Multiples explained by the model. Outside [min, max] a row is neither trained
# on nor given a gap (losses, near-zero denominators, data errors).
# exclude_sectors: Financials' sales / EBITDA / FCF mean something else.
# extra_features: used for that multiple only, on top of FAIR_VALUE_FEATURES.
# in_verdict=False: shown in the report, left out of the combined verdict.
FAIR_VALUE_TARGETS = {
    "pe": {"column": "trailing_pe", "min": 1.0, "max": 100.0, "label": "PER",
           "extra_features": ("log_book_value", "roe_avg_3y", "accruals")},
    "pb": {"column": "price_to_book", "min": 0.1, "max": 20.0, "label": "PBR",
           "extra_features": ("net_debt_to_capital", "log_book_value", "roe_avg_3y", "log_revenue",
                              "eps_volatility_3y")},
    "ps": {"column": "price_to_sales", "min": 0.05, "max": 40.0, "label": "PSR",
           "exclude_sectors": ("Financials",), "extra_features": ("asset_turnover", "log_revenue", "industry")},
    "ev_ebitda": {"column": "ev_to_ebitda", "min": 1.0, "max": 60.0, "label": "EV/EBITDA",
                  "exclude_sectors": ("Financials",), "extra_features": ("asset_turnover",)},
    "pfcf": {"column": "price_to_fcf", "min": 1.0, "max": 100.0, "label": "P/FCF",
             "exclude_sectors": ("Financials",),
             "extra_features": ("fcf_margin", "cash_conversion_3y", "log_revenue")},
    "pe_norm": {"column": "normalized_pe", "min": 1.0, "max": 100.0, "label": "정규화 PER",
                "in_verdict": False, "extra_features": ("log_book_value", "roe_avg_3y")},
}
# Features treated as missing for some sectors (a lender's operating cash flow mixes in loans).
FAIR_VALUE_FEATURE_EXCLUDE_SECTORS = {"accruals": ("Financials",)}
# A candidate is adopted only if it raises out-of-fold R^2 by this much on Train AND Val.
FAIR_VALUE_MIN_GAIN = 0.01
# Finnhub's multiples are at the fiscal period-end price; rescale them to the snapshot's price.
RESCALE_MULTIPLES_TO_AS_OF_PRICE = True

# Common features. Nothing price-derived (momentum, 52w high, dividend yield,
# market cap): a price drop would be "explained" as deserved cheapness.
# volatility_63d is the one exception, as the only risk proxy.
FAIR_VALUE_FEATURES = [
    "return_on_equity",
    "operating_margin",
    "revenue_growth_yoy",
    "debt_to_equity",
    "payout_ratio_ttm",
    "volatility_63d",
    "gross_margin",
]
# Tested by `python src/main.py compare-features`, not in the model.
FAIR_VALUE_FEATURE_CANDIDATES = [
    "log_revenue", "log_book_value", "op_margin_avg_3y", "roe_avg_3y", "roic", "fcf_margin",
    "net_debt_to_capital", "asset_turnover", "revenue_cagr_3y", "op_margin_volatility",
    "dividend_growth_3y", "dividend_years_no_cut",
    "eps_cagr_3y", "growth_consistency_3y", "eps_volatility_3y", "roe_volatility_3y",
    "gross_margin_volatility_3y", "loss_share_3y", "cash_conversion_3y",
    "roe_spike", "eps_spike",
    "revenue_growth_accel", "revenue_growth_ttm", "revenue_growth_ttm_accel",
    "op_margin_change_1y", "gross_margin_change_1y",
    "current_ratio", "sga_to_sales", "industry",
]

FAIR_VALUE_CV_FOLDS = 5             # out-of-fold by ticker within each as_of
FAIR_VALUE_WINSOR_QUANTILE = 0.02   # used by the "winsor" transform only
FAIR_VALUE_FEATURE_TRANSFORM = "rank"  # each feature -> its percentile within the as_of ("winsor" for comparison)
FAIR_VALUE_MIN_ROWS = 50            # skip an as_of with fewer usable rows
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

# ---- Priced-in expectations (screening.add_expectations / add_label_detail) --
IMPLIED_GROWTH_YEARS = 10   # (1 + excess)^years = PER / median PER
LABEL_DETAIL_BAND = 0.02    # delivered vs. priced-in excess growth this close -> "요구 성장 ≈ 과거 성장"
BASE_EFFECT_CAGR = 1.0      # 3-year earnings growth above this per year -> "(기저 효과 가능)"

# ---- Screening labels (screening.py) ---------------------------------------
# cheapness_rank = percentile (0-100) of -gap among the date's stocks; five 20% bands.
CHEAP_THRESHOLD = 80.0
EXPENSIVE_THRESHOLD = 20.0
VERY_CHEAP_LABEL, VERY_EXPENSIVE_LABEL = "큰 할인", "큰 프리미엄"
LABEL_BANDS = [(80.0, VERY_CHEAP_LABEL), (60.0, "할인"), (40.0, "중립"), (20.0, "프리미엄")]  # rank >= edge; <= 20: 큰 프리미엄
LABELS = [VERY_CHEAP_LABEL, "할인", "중립", "프리미엄", VERY_EXPENSIVE_LABEL]

# Heavy debt = negative equity, or (outside Financials, whose debt is their
# business) net debt / capital above the limit or in the date's top share among
# non-Financials. The model can't see default risk, so its fair multiple is too
# high there: a loss-maker's cheap verdict is withheld, expensive ones keep their
# label; profitable stocks get a warning only.
FINANCIAL_RISK_NET_DEBT_TO_CAPITAL = 1.0
FINANCIAL_RISK_TOP_SHARE = 0.10
FINANCIAL_RISK_LABEL = "판단 보류(재무 위험)"
# A verdict on one multiple within this many days of a stock's first snapshot is withheld.
NEW_LISTING_DAYS = 365
NEW_LISTING_LABEL = "판단 보류(신규 상장)"
# Earnings-deterioration risk (deterioration.py): P(trailing EPS down this much or a loss a year later).
# A 큰 할인 / 할인 label with a risk at or above the threshold is withheld: a shrinking business priced low
# is not a discount.
DETERIORATION_DROP = 0.10
# For a loss-maker, "deteriorated" = still a loss a year later and the loss not even halved.
DETERIORATION_LOSS_IMPROVEMENT = 0.5
DETERIORATION_THRESHOLD = 0.5
DETERIORATION_MIN_TRAIN_ROWS = 5000
DETERIORATION_LABEL = "판단 보류(실적 악화 위험)"
