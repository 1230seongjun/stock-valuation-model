"""
Entry point.

    python src/main.py build     [--force-refresh]   # collect data -> real_data_output/panel.parquet
    python src/main.py evaluate                      # fair-value model quality + gap-vs-return test
    python src/main.py screen    [--ticker AAPL] [--as-of 2026-07-01]
    python src/main.py verify-multiples              # check the PER/PBR rescaling assumption (6 API calls)

build needs FINNHUB_API_KEY (env var or --api-key). evaluate/screen only need
the saved panel, and recompute the fair-value model from it every time, so a
saved panel never carries a stale model.

Colab (every .py uploaded flat into /content):
    import main
    main.build(api_key="...")      # first run ~6 min, then cached in ./data_cache
    main.verify_multiples(api_key="...")
    main.evaluate()
    report = main.screen()         # or main.screen(ticker="AAPL")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from config import TRAIN_START
from data import DEFAULT_CACHE_DIR, collect, data_quality_report
from fair_value import add_fair_value, coefficient_summary, gap_return_test, summarize_diagnostics
from features import add_percentile_scores, build_as_of_dates, build_raw_panel
from screening import report_at, screen as screen_panel
from universe import TICKERS, UNIVERSE

OUTPUT_DIR = Path("real_data_output")
PANEL_PATH = OUTPUT_DIR / "panel.parquet"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def build(api_key: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR, force_refresh: bool = False,
          panel_path: str | Path = PANEL_PATH) -> pd.DataFrame:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    prices, fundamentals = collect(TICKERS, api_key=api_key, cache_dir=cache_dir, force_refresh=force_refresh)
    data_quality_report(prices, fundamentals)

    as_of_dates = build_as_of_dates(TRAIN_START)
    print(f"\nBuilding point-in-time panel: {len(as_of_dates)} snapshots "
          f"({as_of_dates[0].date()} ~ {as_of_dates[-1].date()})...")
    panel = add_percentile_scores(build_raw_panel(TICKERS, prices, fundamentals, UNIVERSE, as_of_dates))
    if panel.empty:
        raise RuntimeError("panel is empty — check that collection worked")

    print(f"  {len(panel)} rows, {panel['ticker'].nunique()} tickers")
    for col in ("trailing_pe", "price_to_book", "return_on_equity", "revenue_growth_yoy", "payout_ratio_ttm", "dividend_yield"):
        print(f"  {col:20s} missing {panel[col].isna().mean():.1%}")

    Path(panel_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path)
    print(f"  saved {panel_path}")
    return panel


def _load(panel_path: str | Path) -> pd.DataFrame:
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다 — 먼저 build를 실행하세요")
    return pd.read_parquet(path)


def evaluate(panel_path: str | Path = PANEL_PATH) -> dict[str, pd.DataFrame]:
    panel, diagnostics = add_fair_value(_load(panel_path))

    summary = summarize_diagnostics(diagnostics)
    print("=== 1. Fair-value model: how well fundamentals explain log multiples (out-of-fold, mean per date) ===")
    print("    r2_model must beat r2_sector_median, i.e. explain more than 'which sector is it'.")
    print(summary.round(3).to_string())

    coefs = coefficient_summary(diagnostics)
    print("\n=== 2. What the market paid for (standardized Ridge coefficient, all dates) ===")
    print("    same_sign_share = share of dates with the same sign as the mean")
    print(coefs.pivot(index="feature", columns="target", values=["mean_coef", "same_sign_share"]).round(2).to_string())

    tests = gap_return_test(panel)
    print("\n=== 3. Hypothesis test: did stocks called cheap later outperform? (not a model metric) ===")
    print("    ic = Spearman(cheapness, forward return) per date; spread = cheapest - most expensive quintile.")
    print("    Newey-West errors for overlapping horizons, BH-FDR across the whole table.")
    if tests.empty:
        print("    (not enough forward-return data)")
    else:
        print(tests.round(4).to_string(index=False))
        n_sig = int(tests["significant_after_fdr"].sum())
        print(f"    {n_sig}/{len(tests)} significant after FDR")
    return {"summary": summary, "coefficients": coefs, "gap_return_test": tests}


def screen(panel_path: str | Path = PANEL_PATH, as_of: str | None = None, ticker: str | None = None,
           save: bool = True) -> pd.DataFrame:
    report = report_at(screen_panel(_load(panel_path)), as_of)
    if report.empty:
        raise ValueError(f"no rows for as_of={as_of}")
    date = report["as_of"].iloc[0].date()

    if ticker:
        match = report[report["ticker"] == ticker.upper()]
        if match.empty:
            print(f"{ticker}: {date} 기준 데이터 없음")
        else:
            row = match.iloc[0]
            rank = f"{row['cheapness_rank']:.0f}" if pd.notna(row["cheapness_rank"]) else "-"
            print(f"{row['ticker']} ({row['sector']}) {date}: {row['valuation_label']} (저평가 순위 {rank}/100)")
            print(row["explanation"])
            for col in ("meme_reason", "value_trap_reason", "transition_reason"):
                if row[col]:
                    print(f"※ {row[col]}")
        return report

    counts = report["valuation_label"].value_counts()
    print(f"Screening report {date} ({len(report)} tickers): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    table = report.assign(
        gap=report["valuation_gap_pct"].map(lambda v: f"{v:+.0%}" if pd.notna(v) else ""),
        pe=report["trailing_pe"].round(1), fair_pe=report["fair_pe"].round(1),
        pb=report["price_to_book"].round(2), fair_pb=report["fair_pb"].round(2),
        rank=report["cheapness_rank"].round(0), sector_rank=report["sector_valuation_rank"].round(0),
        quality=report["quality_score"].round(0),
    )
    cols = ["ticker", "sector", "valuation_label", "rank", "gap", "valuation_basis", "pe", "fair_pe", "pb", "fair_pb",
            "sector_rank", "quality", "loss_flag"]
    print("\n-- 저평가 상위 15 --")
    print(table[cols].head(15).to_string(index=False))
    print("\n-- 고평가 상위 15 --")
    print(table[table["valuation_label"] == "고평가"][cols].tail(15).iloc[::-1].to_string(index=False))

    for flag, reason, title in [("meme_flag", "meme_reason", "급등락·거래량 이상"),
                                ("value_trap_flag", "value_trap_reason", "밸류트랩 후보"),
                                ("transition_flag", "transition_reason", "저평가→고평가 전환")]:
        hits = report[report[flag]]
        if not hits.empty:
            print(f"\n-- {title} ({len(hits)}) --")
            for _, row in hits.iterrows():
                print(f"  {row['ticker']:6s} {row[reason]}")

    if save:
        out = OUTPUT_DIR / f"screening_{date}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nsaved {out}")
    return report


VERIFY_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "KO", "TSLA"]


def verify_multiples(tickers: list[str] = VERIFY_TICKERS, api_key: str | None = None,
                     panel_path: str | Path = PANEL_PATH) -> pd.DataFrame:
    """Checks config.RESCALE_MULTIPLES_TO_AS_OF_PRICE: compares the latest
    snapshot's rescaled and as-reported PER/PBR with Finnhub's current-price
    metrics. Run right after `build` so the latest snapshot is today.
    If "rescaled" sits close to Finnhub and "reported" doesn't, the
    period-end-price assumption holds. A newer quarter reported < 45 days ago
    (not yet in our snapshot, by design) can also cause a gap."""
    from data import fetch_current_metrics

    panel = _load(panel_path)
    latest = panel[panel["as_of"] == panel["as_of"].max()].set_index("ticker")
    rows = []
    for ticker in tickers:
        metric = fetch_current_metrics(ticker, api_key)
        pe_now = next((metric[k] for k in ("peTTM", "peExclExtraTTM", "peBasicExclExtraTTM") if metric.get(k)), None)
        pb_now = next((metric[k] for k in ("pbQuarterly", "pb", "pbAnnual") if metric.get(k)), None)
        row = latest.loc[ticker] if ticker in latest.index else None
        rows.append({
            "ticker": ticker,
            "finnhub_pe_now": pe_now,
            "pe_rescaled": None if row is None else row.get("trailing_pe"),
            "pe_reported": None if row is None else row.get("trailing_pe_reported"),
            "finnhub_pb_now": pb_now,
            "pb_rescaled": None if row is None else row.get("price_to_book"),
            "pb_reported": None if row is None else row.get("price_to_book_reported"),
        })
    result = pd.DataFrame(rows)
    print(f"Latest snapshot in panel: {latest.index.size} tickers as of {panel['as_of'].max().date()}")
    print(result.round(2).to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-value stock screening")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="collect data and build the point-in-time panel")
    b.add_argument("--api-key")
    b.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    b.add_argument("--force-refresh", action="store_true")
    e = sub.add_parser("evaluate", help="fair-value model quality + gap-vs-return test")
    s = sub.add_parser("screen", help="screening report")
    s.add_argument("--as-of")
    s.add_argument("--ticker")
    v = sub.add_parser("verify-multiples", help="check the PER/PBR price-rescaling assumption against Finnhub")
    v.add_argument("--api-key")
    v.add_argument("--tickers", nargs="+", default=VERIFY_TICKERS)
    for p in (b, e, s, v):
        p.add_argument("--panel", default=str(PANEL_PATH))
    args = parser.parse_args()

    if args.command == "build":
        build(api_key=args.api_key, cache_dir=args.cache_dir, force_refresh=args.force_refresh, panel_path=args.panel)
    elif args.command == "evaluate":
        evaluate(args.panel)
    elif args.command == "verify-multiples":
        verify_multiples(args.tickers, api_key=args.api_key, panel_path=args.panel)
    else:
        screen(args.panel, as_of=args.as_of, ticker=args.ticker)


if __name__ == "__main__":
    main()
