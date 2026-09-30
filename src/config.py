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

# CPU processes for the per-ticker panel build and the per-date fair-value
# fits (joblib; -1 = all cores). Each unit is independent, so results are
# identical to a single-process run, only faster. Set 1 to debug.
N_JOBS = -1

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

# ---- Universe --------------------------------------------------------------
# Index pages (universe.INDEX_PAGES) added on top of the 281-name core list.
# () = core only. See universe.py for the survivorship caveat.
UNIVERSE_INDEXES = ("sp400", "sp600")
# universe.industry_groups: a GICS sub-industry is its own group only with at
# least this many universe members; the rest fall back to "<sector> 기타".
# 2026-09-30, gain from the industry one-hot (train/val) by threshold:
#   PSR   min 10 (44 groups) +.013/+.014   min 20 (13) +.014/+.015   min 30 (3) +.002/+.010
#   PER   -.013/+.008   -.004/+.007    P/FCF  -.026/+.005   -.007/+.018
#   PBR   +.005/+.011   +.008/+.012    EV/EBITDA +.001/+.028  -.004/+.016
# Only PSR clears the bar (at 10 and 20; 20 is better and sparser). Other
# multiples gain in Val/Test but lose in Train: early dates have fewer
# members per group, and the sub-industry is today's GICS label, which fits
# older years worse.
INDUSTRY_MIN_TICKERS = 20

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
# exclude_sectors (2026-09-29): for banks, brokers and insurers "sales"
# (interest + fee income), EBITDA (interest is the business) and FCF (loans
# and deposits run through operating cash flow) don't mean what they do for
# other companies. On the 2026-09-29 panel GS came out PSR +277% / P/FCF
# +395% "고평가" while its PER/PBR gaps were -22% / -4%; AXP PSR +283%.
# Financials are judged on PER and PBR only.
# extra_features (2026-09-29): features used for that multiple only, on top
# of FAIR_VALUE_FEATURES — a candidate that clearly helps one multiple can be
# irrelevant to another (FCF margin says a lot about P/FCF, little about
# PBR). Same rule as for FAIR_VALUE_FEATURES, per multiple, with a minimum
# gain so tiny noise-level wins aren't collected: out-of-fold R^2 up by at
# least FAIR_VALUE_MIN_GAIN on Train AND Val, second-round compare-features
# of 2026-09-29 (base = the 9 FAIR_VALUE_FEATURES, Financials excluded):
#   P/FCF  + fcf_margin            +.046/+.026
#   PSR    + asset_turnover        +.078/+.088
#   PBR    + asset_turnover        +.015/+.033
#   PBR    + net_debt_to_capital   +.012/+.026
# Just short: asset_turnover for EV/EBITDA (+.008/+.005), roic for PER
# (+.009/+.016) and PBR (+.007/+.015).
# 2026-09-29, extended universe (1,261 tickers, rank transform), each
# candidate alone on top of the model above (train/val):
#   PER    + log_book_value  +.019/+.021   + roe_avg_3y      +.014/+.022
#   PBR    + log_book_value  +.028/+.038   + roe_avg_3y      +.012/+.028
#   PSR    + log_revenue     +.015/+.015
#   EV/EBITDA + asset_turnover +.015/+.012
#   P/FCF  + asset_turnover  +.015/+.017
# log_book_value / log_revenue are company size without the price. They
# were adopted (user decision, 2026-09-29) hoping they would take the
# small-cap discount out of the gap (before: 26% of small caps but 9% of
# large caps 저평가). They did NOT: their coefficients are negative (PBR on
# log_book_value -0.27, PSR on log_revenue -0.15, same sign on 100% of
# dates) — mostly the denominator effect (more book, lower P/B at a given
# market value), so small caps get a HIGHER fair multiple. Mean
# valuation_gap on 2026-09-29 stayed small -0.12 / mid +0.01 / large +0.20
# (log); labels 저평가 small 139 / large 21. They stay because they explain
# the multiples (the R^2 rule), not because they neutralize size. pe_norm
# takes PER's extras so the PER vs. normalized PER comparison (evaluate 2b)
# stays like for like.
# Second round (all of the above in, each feature dropped once; R^2 lost
# train/val, bar = FAIR_VALUE_MIN_GAIN on both):
#   log_book_value  PER .021/.023  PBR .029/.039
#   roe_avg_3y      PER .017/.023  PBR .013/.030
#   asset_turnover  PSR .104/.072  EV/EBITDA .015/.012  P/FCF .014/.017
#                   PBR -.001/.002 -> taken out of PBR (chosen under winsor,
#                   no longer pulls its weight with rank + the size features)
#   net_debt_to_capital PBR .012/.016   log_revenue PSR .015/.014
#   fcf_margin      P/FCF .089/.052
# and one new pass on top: log_revenue for PBR +.015/+.013. Third round
# (PBR as below, dropped once): log_book_value .043/.056, net_debt_to_capital
# .018/.019, log_revenue .014/.015, roe_avg_3y .012/.028; asset_turnover
# added back +.001/-.000.
# 2026-09-30 durability candidates (features._add_durability), each alone
# (train/val): passed only eps_volatility_3y for PBR +.017/+.016 and
# cash_conversion_3y for P/FCF +.038/+.029 (drop-one after adding: .017/.016,
# .038/.029). Neither is durability as hoped: eps_volatility_3y's own
# coefficient flips sign (same sign on 47% of dates) — its gain comes from
# the missing flag, i.e. a 3-year mean EPS <= 0; cash_conversion_3y is -0.13
# (P/FCF = PER x net income / FCF, a denominator effect). Near misses:
# growth_consistency_3y (PER +.005/+.013, PSR +.003/+.020). The quality
# premium (WMT, COST, AAPL gaps +151/+132/+111%) stays in the gap —
# screening.add_expectations shows what growth it prices in instead.
# 2026-09-30 evening: log_revenue passed for P/FCF (+.016/+.012, drop-one
# .016/.013) once cash_conversion_3y was in; asset_turnover then fell just
# under the bar there (dropped .0099/.0100, re-added +.0099/+.0100) and was
# taken out of P/FCF like out of PBR before.
# ROE check (the PER coefficient on ROE is -0.31, AAPL's fair PER -35% from
# it): roe_spike (TTM ROE - 3-year mean) and eps_spike (TTM EPS / 3-year
# average) did not pass for PER (-.001/+.003, +.005/+.007) and only moved
# AAPL's ROE effect to -30/-31%. Spearman(ROE, log PER) per date is -0.23
# even among stable earners (|eps_spike - 1| < 10%; -0.52 among spiky ones),
# and ROA in place of ROE docks AAPL harder (-42%, R^2 .378/.419). Read with
# care: ROA carries the same identity (PER = (market cap / assets) / ROA),
# so neither test separates the arithmetic from any "profitability fades"
# pricing — the negative coefficient mixes both, and one-off spikes are only
# part of it. (Corrected 2026-09-30; an earlier note read ROA as proof of an
# economic effect.) (ROE + ROA together: PER +.023/+.010 — ROA is not a
# registered candidate; it would deepen the AAPL penalty to fair PER 18.1.)
# The common FAIR_VALUE_FEATURES were dropped once too: operating_margin,
# current_ratio and sga_to_sales no longer clear the bar for any multiple
# (each <= .006) — they overlap with the features added since. Left in for
# now (removing them is a separate decision; Ridge keeps them harmless).
# in_verdict=False: shown in the report, left out of valuation_gap (the
# combined verdict). pe_norm (PER on 3-year average EPS, 2026-09-29) is a
# candidate replacement for PER that one-off quarters distort less; whether
# it replaces PER is decided from the reference runs, not assumed.
# 2026-09-29 (all splits, 20,112 labelled rows): mean |gap| PER 0.29 vs.
# normalized 0.34 on rows without a fundamental break, 0.46 vs. 0.42 on the
# 23% with one; per-date Spearman between the two gaps 0.65. Normalized PER
# is only better where the TTM is broken, so it does not replace PER.
# Out-of-fold R^2 train/val 0.18/0.11 vs. PER's 0.29/0.33.
# Extended universe + current features (2026-09-29, 59,961 labelled rows):
# 0.43 vs. 0.43 with a break (26%), PER 0.32 vs. 0.35 without; R^2
# 0.25/0.27 vs. PER's 0.38/0.44. Not better even where the TTM is broken.
FAIR_VALUE_TARGETS = {
    "pe": {"column": "trailing_pe", "min": 1.0, "max": 100.0, "label": "PER",
           "extra_features": ("log_book_value", "roe_avg_3y")},
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
FAIR_VALUE_MIN_GAIN = 0.01
# Tried and rejected 2026-09-30 (judged on R^2, Spearman, stability and
# top-20% overlap, not R^2 alone):
#   - within-(date, sector) percentiles of the common features, added
#     (all within +-.005) or instead of the global ones (PER -.034/-.033):
#     sector one-hots on date-ranked features already carry it.
#   - combined gap from standardized residuals (gap / 1.4826 MAD per date and
#     multiple): mean of z keeps 95% of the cheap 20%, median of z 83%, but
#     neither moves stability (val rank corr .892 -> .892/.888) or the
#     fundamental-break share in the tails (31%); per-multiple error scales
#     are already alike (.37-.49 log).
#   - trajectory candidates (revenue growth acceleration, TTM revenue growth
#     and its acceleration, 1-year operating / gross margin change): each
#     alone within +-.005 on every multiple (below).
#   - linear calibration of the out-of-fold prediction (y = a + b * pred on
#     Train): b = .97-.99, MAE unchanged, 98% of labels the same.
#   - quadratic calibration (same, fitted on Train out-of-fold predictions):
#     only PSR's outer-5% bias shrinks consistently (-.11/-.21 -> -.07/+.04 in
#     all splits); MAE -.002 at most, EV/EBITDA Val worse. Not worth a rule.
# Group ablation (val R^2 lost): profitability PER .140, PBR .117, PSR .073;
# growth <= .011 anywhere. All candidates at once: +.02 to +.06 on every
# multiple incl. Train — many small signals the one-at-a-time rule can't see.
# Not pursued (2026-09-30): Val has been used for many selections already, and
# searching bundles for small gains would fit Val rather than find structure.
# Growth left in the residual: per-date out-of-fold Ridge of each residual on
# 9 growth variables, R^2 Train <= 0, Test <= .009 (Val up to .039, PSR only).
# Learning curve (random ticker subsets, same model): Val R^2 75% -> 100% of
# tickers +.002 (PSR) to +.019 (P/FCF) — flattening, sample size is not the
# main limit; more dates add no training rows (each date is fitted alone).
# Large errors (|residual| > .8, criteria fixed in advance: RR >= 1.5 in Val
# AND Test, >= 5% of rows): per-share sales jump 1.8/2.1, loss 1.9/2.1, ROE
# bottom 20% 1.5/1.6; turnaround, top growth, 1-2 multiples ~1.5 (borderline);
# one-off EPS 1.4/1.5; small caps 1.2. 39% of large errors carry no flag.
# Confidence score (logistic on flags + high-error sectors, fitted on Train):
# low-confidence group's large-error rate 1.7-2.0x the rest in Val/Test, AUC
# .62 — below the pre-set 2x, so no confidence rating; flags stay warnings.
# After the freeze (baseline-2026-09-30), scratch-only tests, criteria fixed in
# advance, all rejected (2026-10-01):
#   - market risk (diagnostic): beta + idiosyncratic vol + turnover-based
#     illiquidity add ~+.02 R^2; Amihud's +.11 is market cap leaking in (rank
#     corr with log market cap -.95; market cap alone +.19).
#   - PER on SEC-XBRL earnings net of impairments, disposal/deconsolidation
#     gains, debt-extinguishment and discontinued ops (filed-date point in
#     time, dates >= 2012): HON fixed (PER 7.8 -> 16.9) but one-off-flagged
#     large-error rate +8-9% and PER R^2 -.015 to -.030.
#   - R&D/sales, SBC/sales, capex/D&A (one bundle): +.02 to +.07 on every
#     multiple, almost all from SBC. SBC follows the share price (10% return ->
#     +1.9% SBC next year, t 9.4) and adds <= .008 once the own multiples of 2
#     years ago are in: price contamination can't be ruled out, nothing left
#     beyond past valuation. R&D alone mainly EV/EBITDA (+.02, denominator).

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
#
# 2026-09-29 compare-features (92 snapshots, each candidate added alone to
# the 6 original features): only gross_margin, current_ratio and sga_to_sales
# raised out-of-fold R^2 on Train AND Val for all five multiples, so they
# were moved in. Change in R^2 train/val:
#                 PER          PBR          PSR          EV/EBITDA    P/FCF
#   gross_margin  +.020/+.014  +.014/+.006  +.139/+.118  +.044/+.039  +.007/+.011
#   current_ratio +.002/+.004  +.004/+.003  +.027/+.035  +.033/+.039  +.020/+.022
#   sga_to_sales  +.024/+.025  +.021/+.022  +.062/+.067  +.036/+.042  +.005/+.006
# The rest stay candidates and get re-tested on top of this set (their
# overlap with these three is unknown until then).
# 2026-09-30 clean-up, the adoption rule run backwards: a common feature
# goes when dropping it costs < FAIR_VALUE_MIN_GAIN on train or val for
# EVERY verdict multiple. operating_margin, gross_margin, current_ratio and
# sga_to_sales each failed alone, but they overlap (gross - SG&A ~ operating
# margin), so removal was tested jointly (R^2 lost train/val):
#   all four: >= bar in all five multiples (PBR .044/.037, P/FCF .045/.024)
#   current_ratio + sga_to_sales: under the bar everywhere (max EV/EBITDA
#     .009/.007) -> both removed
#   + operating_margin: EV/EBITDA .012/.011 -> operating_margin stays
#   + gross_margin instead: all five >= bar -> gross_margin stays
FAIR_VALUE_FEATURES = [
    "return_on_equity",
    "operating_margin",
    "revenue_growth_yoy",
    "debt_to_equity",
    "payout_ratio_ttm",
    "volatility_63d",
    "gross_margin",
]

# Added 2026-09-28 (Finnhub statement ratios + trend + dividend history),
# NOT in the model until `python src/main.py compare-features` shows they
# help on Train/Val. All are free of the snapshot's price, except that none
# of the dividend ones is dividend_yield (a price drop raises the yield, the
# same problem as momentum). dividend_growth_3y / dividend_years_no_cut
# capture the premium the market pays for a reliable, growing dividend.
# 2026-09-29 first round, alone on top of the original 6 (train/val):
#   asset_turnover  PSR +.146/+.157 but PER -.006/+.001, P/FCF -.005/+.001
#   fcf_margin      helps 4 of 5; PER -.002/-.008
#   roic            PBR/PER up, others ~0 or slightly down
#   revenue_cagr_3y, dividend_growth_3y, dividend_years_no_cut: ~0 or down
#   net_debt_to_capital, op_margin_volatility: mixed
# Second round: fcf_margin, asset_turnover and net_debt_to_capital went into
# single multiples (FAIR_VALUE_TARGETS extra_features); they stay here for
# the others. op_margin_avg_3y / roe_avg_3y (3-year means, less exposed to a
# one-off quarter) added 2026-09-29.
# log_revenue / log_book_value (2026-09-29, with the mid/small-cap extension):
# company size without the snapshot price — market cap is price-derived and
# would explain away cheapness like momentum does (features._size_features).
FAIR_VALUE_FEATURE_CANDIDATES = [
    "log_revenue",
    "log_book_value",
    "op_margin_avg_3y",
    "roe_avg_3y",
    "roic",
    "fcf_margin",
    "net_debt_to_capital",
    "asset_turnover",
    "revenue_cagr_3y",
    "op_margin_volatility",
    "dividend_growth_3y",
    "dividend_years_no_cut",
    # durability / multi-year growth (2026-09-30, features._add_durability):
    # what the market may pay for beyond this year's numbers
    "eps_cagr_3y",
    "growth_consistency_3y",
    "eps_volatility_3y",
    "roe_volatility_3y",
    "gross_margin_volatility_3y",
    "loss_share_3y",
    "cash_conversion_3y",
    # this year's earnings vs. the company's own 3-year norm (2026-09-30):
    # can the model tell a one-off ROE/EPS jump from a durably high ROE?
    "roe_spike",
    "eps_spike",
    # trajectory (2026-09-30, features._add_trajectory): improving or
    # deteriorating growth and margins, the past-data proxy for expectations
    "revenue_growth_accel",
    "revenue_growth_ttm",
    "revenue_growth_ttm_accel",
    "op_margin_change_1y",
    "gross_margin_change_1y",
    # removed from FAIR_VALUE_FEATURES 2026-09-30 (see there), candidates again
    "current_ratio",
    "sga_to_sales",
    # GICS sub-industry group (one-hot on top of sector, 2026-09-30,
    # universe.industry_groups / INDUSTRY_MIN_TICKERS)
    "industry",
]

FAIR_VALUE_CV_FOLDS = 5             # out-of-fold by ticker within each as_of
FAIR_VALUE_WINSOR_QUANTILE = 0.02   # clip each feature to [2%, 98%] per as_of
# "winsor": raw values clipped as above. "rank": each feature replaced by its
# percentile within the as_of cross-section, so one extreme value (AAPL's
# buyback-inflated ROE drove its fair PSR +108% on 2026-09-29) can move a
# fair multiple no more than the most extreme rank. compare-features runs
# the other one as a comparison row.
# 2026-09-29 (92 snapshots, change in out-of-fold R^2 train/val, rank vs.
# winsor): PER +.048/+.045, PBR +.038/+.030, PSR +.012/+.018 (all past
# FAIR_VALUE_MIN_GAIN), P/FCF +.009/+.011, EV/EBITDA -.004/+.015 (~0),
# normalized PER +.006/+.004. Switched to rank. extra_features were chosen
# under winsor; the next compare-features re-checks them under rank.
FAIR_VALUE_FEATURE_TRANSFORM = "rank"
FAIR_VALUE_MIN_ROWS = 50            # skip an as_of with fewer usable rows
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]

# ---- Priced-in expectations (screening.add_expectations) -----------------
# "How much future is in the price": the yearly EPS growth above the median
# stock's that a PER needs over this many years, if the stock is then valued
# like the median stock (same PER) and both are discounted alike:
#   (1 + excess)^years = PER / median PER   (same as_of, PER bounds)
# A description, not a model input: it is built from the price on purpose.
IMPLIED_GROWTH_YEARS = 10
# screening.add_label_detail: delivered and priced-in excess growth closer
# than this (per year) count as "요구 성장 ≈ 과거 성장" — a 3-year EPS CAGR is too
# noisy to split on a hair (2026-09-30: AAPL +6.9% vs +6.9%, COST +7.1% vs
# +8.8%, CRI -10.8% vs -11.8%). A judgment call, not tuned on any split.
LABEL_DETAIL_BAND = 0.02
# 3-year earnings growth above this per year (8x in 3 years) is marked "기저
# 효과 가능": from a tiny base the rate says little (2026-09-30: BROS +586%/yr
# read "과거 성장 > 요구 성장" against +14% priced in). Real booms are marked too (NVDA
# +152%) — it says "check the base", not "wrong". A judgment call.
BASE_EFFECT_CAGR = 1.0

# ---- Screening labels (screening.py) ---------------------------------------
# cheapness_rank = percentile (0-100) of -valuation_gap across all stocks at
# the same as_of; higher = cheaper relative to its own fair value.
CHEAP_THRESHOLD = 80.0
EXPENSIVE_THRESHOLD = 20.0
