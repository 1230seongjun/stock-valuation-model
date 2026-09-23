"""
Real-data counterpart to tests/synthetic_smoke_test.py: collects the actual
ticker universe (universe.py, 283 tickers across 11 sectors as of
2026-09-18 — see universe.py's docstring for why sector sizes vary) from
yfinance + Finnhub, builds the point-in-time panel, and runs the full model
pipeline on the Train/Val/Test calendar boundaries decided in config.py
(16y/4y/2y).

tests/synthetic_smoke_test.py only proves the pipeline MECHANICS work on
fake data. This is the first time any of it touches real market data.

Usage in Colab (every .py file uploaded flat into one folder):
    !pip install -q pandas numpy scipy scikit-learn statsmodels xgboost pyarrow yfinance finnhub-python

    FINNHUB_API_KEY = "your_key_here"
    import run_real_data
    reports = run_real_data.main(api_key=FINNHUB_API_KEY)
    # Re-running the cell above again is now fast — price/fundamentals are
    # cached to ./data_cache/ (data_collection.py) after the first run, so
    # only the model/feature-engineering part re-executes. This cache lives
    # in Colab's local disk, so it's gone after a runtime restart/
    # disconnect (mount Google Drive and pass cache_dir="/content/drive/
    # MyDrive/.../data_cache" instead if you want it to survive that).
    # Force a real re-fetch (e.g. to pick up the last few days'/quarter's
    # new data) with:
    #   reports = run_real_data.main(api_key=FINNHUB_API_KEY, force_refresh=True)

Or as a script:
    export FINNHUB_API_KEY=your_key_here
    python run_real_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    _this_dir = Path(__file__).resolve().parent
except NameError:  # pasted into a notebook cell — __file__ doesn't exist
    _this_dir = Path.cwd()

for _candidate in (_this_dir, _this_dir.parent / "src"):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

import pandas as pd

from baselines import compare_baselines
from config import HORIZONS_MONTHS, REBALANCE_FREQ, TRAIN_END, TRAIN_START, VAL_END
from data_collection import (
    DEFAULT_CACHE_DIR,
    fetch_price_history_cached,
    fetch_universe_fundamentals_cached,
)
from features import add_percentile_scores, build_raw_panel
from pipeline import apply_fdr_across_report, run_all_horizons
from universe import TICKERS, UNIVERSE

OUTPUT_DIR = _this_dir / "real_data_output"


def collect(
    api_key: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Fetch price history (yfinance) + point-in-time fundamentals (Finnhub)
    for every ticker in universe.TICKERS — reading from `cache_dir` first
    (data_collection.py's on-disk cache, added 2026-09-18) and only hitting
    the network for whatever isn't cached yet. On a full cache hit this
    finishes in well under a minute instead of ~6+ minutes; on a first run
    (empty cache) it's the same speed as before, since everything still has
    to be fetched once.

    Pass force_refresh=True to ignore the cache and re-fetch everything —
    do this before a run whose conclusions you actually care about, since
    cached data can lag "today" (see data_collection.py's caching note).

    period="max" (not a fixed "10y" anymore): config.TRAIN_START now reaches
    back to 2004, so a fixed 10-year window would silently truncate most of
    Train's price/technical-indicator history. "max" costs nothing extra
    (build_raw_panel only ever looks at data up to each as_of date anyway)
    and means this doesn't go stale again if TRAIN_START moves further back
    later."""
    print(f"Fetching price history for {len(TICKERS)} tickers (yfinance, cache_dir={cache_dir})...")
    prices = {}
    for i, ticker in enumerate(TICKERS, 1):
        prices[ticker] = fetch_price_history_cached(
            ticker, cache_dir=cache_dir, period="max", force_refresh=force_refresh
        )
        print(f"  [{i}/{len(TICKERS)}] {ticker}: {len(prices[ticker])} rows")

    print(f"\nFetching point-in-time fundamentals for {len(TICKERS)} tickers (Finnhub, cache_dir={cache_dir})...")
    fundamentals = fetch_universe_fundamentals_cached(
        TICKERS, cache_dir=cache_dir, api_key=api_key, force_refresh=force_refresh
    )
    for ticker, df in fundamentals.items():
        status = f"{len(df)} quarterly rows" if not df.empty else "EMPTY — check ticker/API key"
        print(f"  {ticker}: {status}")

    return prices, fundamentals


def data_quality_report(prices: dict[str, pd.DataFrame], fundamentals: dict[str, pd.DataFrame]) -> list[str]:
    """Cheap sanity checks (technical-spec.md 4.1: 'existence + plausibility
    of required columns right after collection') before any of this feeds
    the model. Doesn't drop or fix anything — just surfaces what to look at
    by eye. Returns the list of tickers with at least one issue."""
    print("\n--- Data quality report ---")
    flagged = []
    for ticker in TICKERS:
        price_df = prices.get(ticker)
        fund_df = fundamentals.get(ticker)
        issues = []

        if price_df is None or price_df.empty:
            issues.append("no price history")
        elif (price_df["close"] <= 0).any():
            issues.append("non-positive close price present")

        if fund_df is None or fund_df.empty:
            issues.append("no fundamentals")
        elif "trailing_pe" in fund_df.columns and fund_df["trailing_pe"].isna().all():
            issues.append("trailing_pe entirely missing")

        if issues:
            flagged.append(ticker)
            print(f"  {ticker}: {', '.join(issues)}")

    if not flagged:
        print("  no issues found")
    return flagged


def main(
    api_key: str | None = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> dict:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError('Finnhub API 키가 필요합니다 — run_real_data.main(api_key="...")로 직접 전달하세요.')

    prices, fundamentals = collect(api_key, cache_dir=cache_dir, force_refresh=force_refresh)
    data_quality_report(prices, fundamentals)

    as_of_dates = list(pd.date_range(TRAIN_START, pd.Timestamp.today().normalize(), freq=REBALANCE_FREQ))
    print(f"\nBuilding point-in-time panel for {len(as_of_dates)} as_of dates ({REBALANCE_FREQ}, "
          f"{as_of_dates[0].date()} ~ {as_of_dates[-1].date()})...")
    raw_panel = build_raw_panel(TICKERS, prices, fundamentals, UNIVERSE, as_of_dates)
    print(f"  raw panel shape: {raw_panel.shape}")
    if raw_panel.empty:
        raise RuntimeError("raw panel came back empty — check that price/fundamentals collection actually worked")

    panel = add_percentile_scores(raw_panel)
    print(f"  composite_score NA rate: {panel['composite_score'].isna().mean():.1%}")
    if "value_composite_score" in panel.columns:
        print(f"  value_composite_score NA rate: {panel['value_composite_score'].isna().mean():.1%}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    panel_path = OUTPUT_DIR / "panel.parquet"
    panel.to_parquet(panel_path)
    print(f"  saved panel to {panel_path} (pd.read_parquet it later to skip re-collecting from yfinance/Finnhub)")

    print(f"\nRunning model pipeline (Train ends {TRAIN_END.date()}, Val ends {VAL_END.date()}, "
          f"Test is everything after that)...")
    reports = run_all_horizons(panel, HORIZONS_MONTHS, train_end=TRAIN_END, val_end=VAL_END)

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

        vo = report.get("factor_validity_value_only", {})
        if vo:
            print("   -- value_composite_score diagnostic (PE/PBR/momentum only, excl. operating_margin/dividend_yield) --")
            qs_vo, ic_vo = vo.get("quantile_spread", {}), vo.get("information_coefficient", {})
            if "spread" in qs_vo:
                print(f"      Factor spread (top-bottom quintile): {qs_vo['spread']:.4f} (p={qs_vo['p_value']:.4f})")
            if "mean_ic" in ic_vo:
                print(f"      Mean IC: {ic_vo['mean_ic']:.4f} (p={ic_vo['p_value']:.4f}, n_periods={ic_vo['n_periods']})")

        fmb_rows = report.get("fama_macbeth_univariate", {}).get("significance", [])
        if fmb_rows:
            print("   Univariate Fama-MacBeth (per indicator, sorted by p-value):")
            for row in fmb_rows:
                if pd.isna(row["p_value"]):
                    continue
                print(f"     {row['feature']:22s} mean_coef={row['mean_coef']:+.4f} "
                      f"t={row['t_stat']:+.2f} p={row['p_value']:.4f} (n_periods={row['n_periods']})")

        print("   Baseline comparison (Test MAE — lower is better). Real bar is vs_sector_avg, not vs_zero:")
        print("     vs_zero can be beaten just by capturing broad market drift (US equities trend up on")
        print("     average) — vs_sector_avg only rewards separating stocks WITHIN a sector by valuation.")
        baseline_table = compare_baselines(panel, horizon, TRAIN_END, VAL_END)
        if "note" in baseline_table.columns and baseline_table["baseline"].iloc[0] == "all":
            print(f"     SKIPPED: {baseline_table['note'].iloc[0]}")
        else:
            for _, row in baseline_table.iterrows():
                note = f" ({row['note']})" if "note" in row and pd.notna(row.get("note")) else ""
                print(f"     {row['baseline']:16s} test_mae={row['test_mae']:.4f} test_rmse={row['test_rmse']:.4f} "
                      f"vs_sector_avg={row['improvement_vs_sector_average']:+.4f} "
                      f"vs_zero={row['improvement_vs_zero_return']:+.4f}{note}")

    print("\nFDR correction across all horizons' factor-validity + per-indicator tests...")
    fdr_table = apply_fdr_across_report(reports)
    if fdr_table.empty:
        print("  (no p-values collected)")
    else:
        n_sig = int(fdr_table["significant_after_fdr"].sum())
        print(f"  {n_sig}/{len(fdr_table)} tests significant after FDR correction")
        if n_sig:
            print(fdr_table[fdr_table["significant_after_fdr"]].to_string(index=False))
        else:
            print("  none survived FDR correction — full table below for reference:")
            print(fdr_table.to_string(index=False))

    print(f"\nDone. Panel saved at {panel_path} — per-sector error breakdown is a separate, not-yet-built step "
          f"(discussed 2026-09-14: pooled vs. per-sector split decision comes from that).")
    return reports


if __name__ == "__main__":
    main()
