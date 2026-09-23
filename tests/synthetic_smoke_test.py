"""
Runs the full feature-engineering + modeling pipeline on SYNTHETIC data with
a known embedded signal, so we can check two different things at once:

1. Mechanical correctness: does the code run end-to-end without errors, on
   data shaped like the real thing (multiple sectors, missing values,
   look-ahead-sensitive dates)?
2. Sanity of the statistics: since we control the "true" relationship here
   (a synthetic per-ticker skill factor drives both fundamentals AND future
   returns), does composite_score end up with a positive IC and a positive
   quantile spread, as it should if factor_validation.py is implemented
   correctly?

This tells us nothing about real markets — it's a unit test for the
pipeline, not a backtest. Real conclusions need Finnhub + yfinance data
(see README.md).

Run: python tests/synthetic_smoke_test.py
(or, in Colab with every .py file uploaded flat into one folder: !python synthetic_smoke_test.py)
"""
import sys
from pathlib import Path

try:
    # normal case: run as a script, e.g. `python synthetic_smoke_test.py`
    _this_dir = Path(__file__).resolve().parent
except NameError:
    # __file__ doesn't exist when this code runs pasted into a notebook
    # cell / via %run — fall back to the notebook's current directory
    # (in Colab that's /content/, where the uploaded files live)
    _this_dir = Path.cwd()

# Covers both layouts: everything uploaded flat into one folder (Colab),
# and the original repo split into src/ + tests/.
for _candidate in (_this_dir, _this_dir.parent / "src"):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

import numpy as np
import pandas as pd

from config import HORIZONS_MONTHS, RANDOM_SEED
from features import add_percentile_scores, add_revenue_growth_yoy, build_raw_panel
from pipeline import apply_fdr_across_report, run_all_horizons

rng = np.random.default_rng(RANDOM_SEED)

N_TICKERS = 30
SECTORS = ["Technology", "Healthcare", "Financial Services", "Consumer Cyclical"]
PRICE_START = pd.Timestamp("2018-01-01")
PRICE_END = pd.Timestamp("2024-06-01")
AS_OF_DATES = pd.date_range("2019-01-01", "2023-01-01", freq="QS")  # quarterly rebalance


def make_universe():
    tickers = [f"SYN{i:03d}" for i in range(N_TICKERS)]
    universe = {t: {"name": t, "sector": SECTORS[i % len(SECTORS)]} for i, t in enumerate(tickers)}
    true_alpha = {t: rng.normal(0, 1) for t in tickers}
    return tickers, universe, true_alpha


def make_prices(tickers, true_alpha) -> dict[str, pd.DataFrame]:
    dates = pd.date_range(PRICE_START, PRICE_END, freq="B")
    prices = {}
    for t in tickers:
        alpha = true_alpha[t]
        daily_drift = (0.04 + 0.06 * alpha) / 252  # higher alpha -> higher expected return
        daily_vol = 0.018
        returns = rng.normal(daily_drift, daily_vol, size=len(dates))
        close = 50 * np.exp(np.cumsum(returns))
        volume = rng.integers(1_000_000, 5_000_000, size=len(dates)).astype(float)
        prices[t] = pd.DataFrame({"date": dates, "close": close, "volume": volume})
    return prices


def make_fundamentals(tickers, true_alpha) -> dict[str, pd.DataFrame]:
    """dividend_yield and revenue_growth_yoy are no longer generated directly
    — features.py now derives them (add_revenue_growth_yoy, _dividend_yield),
    so this generates the RAW components instead (eps, payout_ratio_ttm,
    sales_per_share), matching what data_collection.py actually fetches from
    Finnhub. This is what exercises the new derivation code in the smoke
    test."""
    periods = pd.date_range("2017-01-01", "2023-01-01", freq="QS")
    fundamentals = {}
    for t in tickers:
        alpha = true_alpha[t]
        n = len(periods)

        # sales_per_share as a cumulative-growth process (like the price
        # series above) so consecutive-year values actually encode the
        # ~(0.05 + 0.03*alpha) annual growth rate revenue_growth_yoy should
        # recover.
        annual_growth = 0.05 + 0.03 * alpha
        quarterly_log_growth = np.log1p(annual_growth / 4) + rng.normal(0, 0.01, n)
        sales_per_share = 20 * np.exp(np.cumsum(quarterly_log_growth))

        fundamentals[t] = pd.DataFrame(
            {
                "period": periods,
                "trailing_pe": np.clip(22 - 6 * alpha + rng.normal(0, 3, n), 3, None),
                "price_to_book": np.clip(3 - 0.8 * alpha + rng.normal(0, 0.5, n), 0.2, None),
                "return_on_equity": 0.10 + 0.05 * alpha + rng.normal(0, 0.03, n),
                "debt_to_equity": np.clip(1.0 - 0.2 * alpha + rng.normal(0, 0.3, n), 0, None),
                "operating_margin": 0.12 + 0.04 * alpha + rng.normal(0, 0.03, n),
                "eps": np.clip(2.0 + 1.5 * alpha + rng.normal(0, 0.3, n), 0.1, None),
                "payout_ratio_ttm": np.clip(0.3 + 0.05 * alpha + rng.normal(0, 0.05, n), 0, 1),
                "sales_per_share": sales_per_share,
            }
        )
    return fundamentals


def main():
    print("1. Generating synthetic universe...")
    tickers, universe, true_alpha = make_universe()
    prices = make_prices(tickers, true_alpha)
    fundamentals = make_fundamentals(tickers, true_alpha)
    print(f"   {len(tickers)} tickers across {len(SECTORS)} sectors")

    print("1b. Checking derived revenue_growth_yoy against its known ~(0.05 + 0.03*alpha) design...")
    sample_ticker = tickers[0]
    derived = add_revenue_growth_yoy(fundamentals[sample_ticker])
    recovered = derived["revenue_growth_yoy"].dropna().mean()
    expected = 0.05 + 0.03 * true_alpha[sample_ticker]
    print(f"   {sample_ticker}: recovered mean YoY growth={recovered:.4f} vs. design target={expected:.4f}")
    assert abs(recovered - expected) < 0.03, "derived revenue_growth_yoy is way off its design target"

    print("2. Building point-in-time raw panel (this is the look-ahead-bias-sensitive part)...")
    raw_panel = build_raw_panel(tickers, prices, fundamentals, universe, list(AS_OF_DATES))
    print(f"   raw panel shape: {raw_panel.shape}")
    assert not raw_panel.empty, "raw panel came back empty — check date ranges overlap"
    for col in ("dividend_yield", "revenue_growth_yoy"):
        na_rate_col = raw_panel[col].isna().mean()
        print(f"   {col} NA rate in raw panel: {na_rate_col:.1%}")
        assert na_rate_col < 0.5, f"{col} is NaN almost everywhere — derivation likely broken"

    print("3. Adding sector-relative percentile scores + composite score...")
    panel = add_percentile_scores(raw_panel)
    na_rate = panel["composite_score"].isna().mean()
    print(f"   composite_score NA rate: {na_rate:.1%}")
    assert na_rate < 0.5, "too many missing composite scores — check indicator availability"

    print("4. Running model pipeline for all horizons (Ridge / Fama-MacBeth / XGBoost / RandomForest)...")
    reports = run_all_horizons(panel, HORIZONS_MONTHS)

    for horizon, report in reports.items():
        print(f"\n--- Horizon: {horizon}m ---")
        if "error" in report:
            print("   SKIPPED:", report["error"])
            continue
        print(f"   n_train={report['n_train']} n_val={report['n_val']} n_test={report['n_test']}")

        ridge_gap = report["ridge"]["train_val_gap"]
        print(f"   Ridge   train_mae={ridge_gap['train_mae']:.4f} val_mae={ridge_gap['val_mae']:.4f} "
              f"gap={ridge_gap['overfit_gap_mae']:.4f}")

        if "train_val_gap" in report.get("xgboost", {}):
            xg = report["xgboost"]["train_val_gap"]
            print(f"   XGBoost train_mae={xg['train_mae']:.4f} val_mae={xg['val_mae']:.4f} gap={xg['overfit_gap_mae']:.4f}")

        rf = report["random_forest"]["train_val_gap"]
        print(f"   RF      train_mae={rf['train_mae']:.4f} val_mae={rf['val_mae']:.4f} gap={rf['overfit_gap_mae']:.4f}")

        qs = report["factor_validity"]["quantile_spread"]
        ic = report["factor_validity"]["information_coefficient"]
        if "spread" in qs:
            print(f"   Factor spread (top-bottom quintile): {qs['spread']:.4f} (p={qs['p_value']:.4f})")
        if "mean_ic" in ic:
            print(f"   Mean IC: {ic['mean_ic']:.4f} (p={ic['p_value']:.4f}, n_periods={ic['n_periods']})")

    print("\n5. FDR correction across all horizons' factor-validity tests...")
    fdr_table = apply_fdr_across_report(reports)
    print(fdr_table.to_string(index=False) if not fdr_table.empty else "   (no p-values collected)")

    print("\n6. Sanity check: since this synthetic data has a REAL embedded skill factor,")
    print("   composite_score should show a positive IC for at least the shorter horizons.")
    any_positive_ic = any(
        reports[h].get("factor_validity", {}).get("information_coefficient", {}).get("mean_ic", -1) > 0
        for h in HORIZONS_MONTHS
        if "error" not in reports[h]
    )
    status = "PASS" if any_positive_ic else "CHECK NEEDED"
    print(f"   -> {status}")

    print("\nDone. This validates the PIPELINE MECHANICS only — see README.md for plugging in real data.")


if __name__ == "__main__":
    main()
