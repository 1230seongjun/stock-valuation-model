"""
Entry point.

    python src/main.py build     [--force-refresh]   # collect data -> real_data_output/panel.parquet
    python src/main.py verify-multiples              # check the multiples' price rescaling against yfinance
    python src/main.py compare-features              # which candidate features help (Train/Val)
    python src/main.py evaluate                      # fair-value model quality + gap-vs-return test
    python src/main.py screen    [--ticker AAPL] [--as-of 2026-07-01]

build needs FINNHUB_API_KEY (env var or --api-key). The other commands only
need the saved panel, and recompute the fair-value model from it every time,
so a saved panel never carries a stale model.

Colab (every .py uploaded flat into /content):
    import main
    main.build(api_key="...")      # first run ~6 min, then cached in ./data_cache
    main.verify_multiples()
    main.compare_features()
    main.evaluate()
    report = main.screen()         # or main.screen(ticker="AAPL")
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from config import FAIR_VALUE_FEATURE_CANDIDATES, FAIR_VALUE_FEATURES, FAIR_VALUE_TARGETS, TRAIN_START
from data import DEFAULT_CACHE_DIR, collect, data_quality_report
from fair_value import add_fair_value, coefficient_summary, compare_feature_sets, gap_return_test, summarize_diagnostics
from features import add_percentile_scores, build_as_of_dates, build_raw_panel
from screening import report_at, screen as screen_panel
from universe import TICKERS, UNIVERSE

OUTPUT_DIR = Path("real_data_output")
PANEL_PATH = OUTPUT_DIR / "panel.parquet"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


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

    print(f"  {len(panel)} rows, {panel['ticker'].nunique()} tickers — share of rows missing:")
    multiples = [spec["column"] for spec in FAIR_VALUE_TARGETS.values()]
    for col in [*multiples, *FAIR_VALUE_FEATURES, *FAIR_VALUE_FEATURE_CANDIDATES, "dividend_yield"]:
        missing = panel[col].isna().mean()
        warn = "  <- almost empty: check the Finnhub field name (data.FINNHUB_FIELD_MAP)" if missing > 0.95 else ""
        print(f"    {col:24s} {missing:6.1%}{warn}")
    print(f"    dividend payers in latest snapshot: "
          f"{(panel.loc[panel['as_of'] == panel['as_of'].max(), 'dividend_yield'] > 0).mean():.0%}")

    Path(panel_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path)
    print(f"  saved {panel_path}")
    return panel


def _load(panel_path: str | Path) -> pd.DataFrame:
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다 — 먼저 build를 실행하세요")
    return pd.read_parquet(path)


def compare_features(panel_path: str | Path = PANEL_PATH) -> pd.DataFrame:
    """Out-of-fold R^2 of each multiple for: the current features, current +
    each candidate alone, and current + all candidates. Keep a candidate only
    if it helps on train AND val (test is for the final report)."""
    panel = _load(panel_path)
    candidates = [c for c in FAIR_VALUE_FEATURE_CANDIDATES if c in panel.columns]
    sets = {"current": FAIR_VALUE_FEATURES}
    sets.update({f"+{c}": [*FAIR_VALUE_FEATURES, c] for c in candidates})
    sets["+all"] = [*FAIR_VALUE_FEATURES, *candidates]
    print(f"Comparing {len(sets)} feature sets on {panel['as_of'].nunique()} snapshots (takes a few minutes)...")
    table = compare_feature_sets(panel, sets)
    delta = table.drop(columns=["sector_median", "current"]).sub(table["current"], axis=0)
    print("\n=== out-of-fold R^2 (mean per date) ===")
    print(table.round(3).T.to_string())
    print("\n=== change vs. current features (positive = helps) ===")
    print(delta.round(3).T.to_string())
    return table


def evaluate(panel_path: str | Path = PANEL_PATH) -> dict[str, pd.DataFrame]:
    panel, diagnostics = add_fair_value(_load(panel_path))

    summary = summarize_diagnostics(diagnostics)
    print("=== 1. Fair-value model: how well fundamentals explain log multiples (out-of-fold, mean per date) ===")
    print("    r2_model must beat r2_sector_median, i.e. explain more than 'which sector is it'.")
    print(summary.round(3).to_string())

    labelled = panel[panel["valuation_gap"].notna() & (panel["n_gaps"] > 1)]
    print("\n=== 2. Combined verdict: do the multiples agree? ===")
    print(f"    mean share of views on the same side as the combined gap: {labelled['gap_agreement'].mean():.0%}")
    print(f"    rows where every view agrees: {(labelled['gap_agreement'] == 1).mean():.0%}")

    coefs = coefficient_summary(diagnostics)
    print("\n=== 3. What the market paid for (standardized Ridge coefficient, all dates) ===")
    print("    same_sign_share = share of dates with the same sign as the mean")
    print(coefs.pivot(index="feature", columns="target", values=["mean_coef", "same_sign_share"]).round(2).to_string())

    tests = gap_return_test(panel)
    print("\n=== 4. Hypothesis test: did stocks called cheap later outperform? (not a model metric) ===")
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
            for col in ("meme_reason", "value_trap_reason", "transition_reason", "report_lag_reason"):
                if row[col]:
                    print(f"※ {row[col]}")
        return report

    counts = report["valuation_label"].value_counts()
    print(f"Screening report {date} ({len(report)} tickers): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    pct = lambda s: s.map(lambda v: f"{v:+.0%}" if pd.notna(v) else "")
    table = report.assign(
        rank=report["cheapness_rank"].round(0),
        gap=pct(report["valuation_gap_pct"]),
        agree=[f"{a * n:.0f}/{n:.0f}" if pd.notna(a) else "" for a, n in zip(report["gap_agreement"], report["n_gaps"])],
        **{spec["label"]: pct(report[f"{key}_gap"]) for key, spec in FAIR_VALUE_TARGETS.items()},
        quality=report["quality_score"].round(0),
    )
    cols = ["ticker", "sector", "valuation_label", "rank", "gap", "agree",
            *[spec["label"] for spec in FAIR_VALUE_TARGETS.values()], "quality"]
    print("    (gap = combined; agree = views on the same side; per-multiple columns = actual vs. fair)")
    print("\n-- 저평가 상위 15 --")
    print(table[cols].head(15).to_string(index=False))
    print("\n-- 고평가 상위 15 --")
    print(table[table["valuation_label"] == "고평가"][cols].tail(15).iloc[::-1].to_string(index=False))

    for flag, reason, title in [("meme_flag", "meme_reason", "급등락·거래량 이상"),
                                ("value_trap_flag", "value_trap_reason", "밸류트랩 후보"),
                                ("transition_flag", "transition_reason", "저평가→고평가 전환"),
                                ("report_lag_flag", "report_lag_reason", "재무 기준일 이후 주가 급변")]:
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


# HON: suspected spin-off mismatch on 2026-09-29 (screening.flag_report_lag).
VERIFY_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "KO", "TSLA", "HON"]


def _yf_trailing_fcf(ticker) -> float | None:
    """Operating cash flow - capex over the last 4 reported quarters, from
    yfinance's cash-flow statement. Not info["freeCashflow"]: that is
    Yahoo's *levered* FCF (after interest, a different definition), which on
    2026-09-29 put MSFT's P/FCF at 229 and KO's at 72 against ~57 / ~26."""
    try:
        cf = ticker.quarterly_cashflow
        fcf = cf.loc["Free Cash Flow"].dropna().iloc[:4]
    except Exception:  # noqa: BLE001 — missing statement or row: no reference value
        return None
    return float(fcf.sum()) if len(fcf) == 4 else None


def verify_multiples(tickers: list[str] = VERIFY_TICKERS, panel_path: str | Path = PANEL_PATH) -> pd.DataFrame:
    """Checks config.RESCALE_MULTIPLES_TO_AS_OF_PRICE: the latest snapshot's
    rescaled multiples next to yfinance's current ones (and the as-reported
    Finnhub values). Run right after `build` so the latest snapshot is today.
    "rescaled" close to yfinance and "reported" not = the rescaling holds.

    yfinance, not Finnhub, is the reference: on 2026-09-23 Finnhub's own
    "current" pbQuarterly metric turned out to equal its period-end series
    value exactly (AAPL 38.49 vs. yfinance 46.16), so it can't tell stale from
    current. Gaps can also come from a quarter reported < 45 days ago (not in
    our snapshot yet, by design) or from yfinance's own definitions (its
    EV/EBITDA and FCF differ from Finnhub's in detail — look for the same
    ballpark and the right direction of the rescaling, not equality)."""
    import yfinance as yf

    panel = _load(panel_path)
    as_of = panel["as_of"].max()
    latest = panel[panel["as_of"] == as_of].set_index("ticker")
    rows = []
    for ticker in tickers:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        row = latest.loc[ticker] if ticker in latest.index else pd.Series(dtype=float)
        mcap, fcf = info.get("marketCap"), _yf_trailing_fcf(yf_ticker)
        yf_values = {
            "trailing_pe": info.get("trailingPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            "price_to_fcf": mcap / fcf if mcap and fcf and fcf > 0 else None,
        }
        for col, yf_value in yf_values.items():
            rows.append({"ticker": ticker, "multiple": col, "yfinance": yf_value,
                         "rescaled": row.get(col), "reported": row.get(f"{col}_reported")})
    result = pd.DataFrame(rows)
    print(f"Latest snapshot: {len(latest)} tickers as of {as_of.date()}")
    if latest["price"].isna().all():
        print("  WARNING: no prices in the latest snapshot — rescaling could not run (rebuild with the current code)")
    print(result.pivot(index="ticker", columns="multiple", values=["yfinance", "rescaled", "reported"])
          .swaplevel(axis=1).sort_index(axis=1).round(2).to_string())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-value stock screening")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="collect data and build the point-in-time panel")
    b.add_argument("--api-key")
    b.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    b.add_argument("--force-refresh", action="store_true")
    v = sub.add_parser("verify-multiples", help="check the multiples' price rescaling against yfinance")
    v.add_argument("--tickers", nargs="+", default=VERIFY_TICKERS)
    c = sub.add_parser("compare-features", help="out-of-fold R^2 with each candidate feature added")
    e = sub.add_parser("evaluate", help="fair-value model quality + gap-vs-return test")
    s = sub.add_parser("screen", help="screening report")
    s.add_argument("--as-of")
    s.add_argument("--ticker")
    for p in (b, v, c, e, s):
        p.add_argument("--panel", default=str(PANEL_PATH))
    args = parser.parse_args()

    if args.command == "build":
        build(api_key=args.api_key, cache_dir=args.cache_dir, force_refresh=args.force_refresh, panel_path=args.panel)
    elif args.command == "verify-multiples":
        verify_multiples(args.tickers, panel_path=args.panel)
    elif args.command == "compare-features":
        compare_features(args.panel)
    elif args.command == "evaluate":
        evaluate(args.panel)
    else:
        screen(args.panel, as_of=args.as_of, ticker=args.ticker)


if __name__ == "__main__":
    main()
