"""
Entry point.

    python src/main.py build     [--force-refresh]   # collect data -> real_data_output/panel.parquet
    python src/main.py verify-multiples              # check the multiples' price rescaling against yfinance
    python src/main.py compare-features [--drop-check]  # which candidate features help (Train/Val)
    python src/main.py evaluate                      # fair-value model quality + gap-vs-return test
    python src/main.py screen    [--ticker AAPL] [--as-of 2026-07-01]   # + market_context_<date>.json
    python src/main.py export    [--as-of 2026-07-01]   # per-stock JSON for an LLM -> real_data_output/llm/<date>/
    python src/main.py explain   --tickers AAPL NVDA | --submit | --collect   # Korean explanations (Claude API)
    python src/main.py publish   [--fundamentals-max-age 14]   # scheduled job: screen everything -> real_data_output/site/

build needs FINNHUB_API_KEY (env var or --api-key). The other commands only
need the saved panel, and recompute the fair-value model from it every time,
so a saved panel never carries a stale model.

Colab: clone the repo's dev branch and add src/ to sys.path — the full
cell-by-cell setup (Drive cache, module reload after git pull) is in README.md.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    FAIR_VALUE_FEATURE_CANDIDATES,
    FAIR_VALUE_FEATURE_TRANSFORM,
    FAIR_VALUE_FEATURES,
    FAIR_VALUE_MIN_GAIN,
    FAIR_VALUE_TARGETS,
    IMPLIED_GROWTH_YEARS,
    INDUSTRY_MIN_TICKERS,
    TRAIN_START,
    UNIVERSE_INDEXES,
)
from data import (
    DEFAULT_CACHE_DIR,
    collect,
    collect_fiscal_year_ends,
    collect_short_interest,
    collect_wiki_views,
    data_quality_report,
    load_sec_facts,
)
from fair_value import (
    add_fair_value,
    coefficient_summary,
    compare_feature_sets,
    gap_return_test,
    r2_by_group,
    split_of,
    stability_summary,
    summarize_diagnostics,
    target_features,
)
from features import add_percentile_scores, add_sentiment_features, build_as_of_dates, build_raw_panel
from llm_context import export_date
from market_context import context_lines, load_market_data, market_context
from screening import (
    BASE_EFFECT_NOTE,
    DETAIL_NOTES,
    ONE_OFF_NOTE,
    flag_fundamental_break,
    report_at,
    screen as screen_panel,
)
from universe import industry_groups, load_sub_industries, load_universe

OUTPUT_DIR = Path("real_data_output")
PANEL_PATH = OUTPUT_DIR / "panel.parquet"
SITE_DIR = OUTPUT_DIR / "site"  # what the web/app reads (publish)
# screen prints this many rows per flag (most extreme labels first); the CSV
# has all of them (2026-09-29: 393 fundamental-break lines otherwise).
MAX_FLAGGED_SHOWN = 15

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)


def _universe(cache_dir: str | Path) -> dict[str, dict[str, str]]:
    universe = load_universe(cache_dir, UNIVERSE_INDEXES)
    groups = industry_groups(universe, load_sub_industries(cache_dir), INDUSTRY_MIN_TICKERS)
    return {t: {**meta, "industry": groups[t]} for t, meta in universe.items()}


def _with_sentiment(panel: pd.DataFrame, tickers: list[str], cache_dir: str | Path) -> pd.DataFrame:
    """Short interest and Wikipedia attention (descriptive only, features.add_sentiment_features),
    recomputed for every row from the cached sources. A source that can't be
    fetched leaves its column empty; the rest of the run is unaffected."""
    sources = {}
    for name, collect_fn in (("short_interest", collect_short_interest), ("wiki_views", collect_wiki_views)):
        try:
            sources[name] = collect_fn(tickers, cache_dir=cache_dir)
        except Exception as exc:  # network / API: the verdicts don't depend on it
            print(f"  {name}: not available ({type(exc).__name__}: {exc})")
    base = panel.drop(columns=["short_ratio", "wiki_views_3m"], errors="ignore")
    return add_sentiment_features(base, **sources)


def build(api_key: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR, force_refresh: bool = False,
          panel_path: str | Path = PANEL_PATH) -> pd.DataFrame:
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    universe = _universe(cache_dir)
    tickers = list(universe)
    sizes = pd.Series({t: m["size"] for t, m in universe.items()}).value_counts()
    print(f"Universe: {len(tickers)} tickers ({', '.join(f'{k} {v}' for k, v in sizes.items())})")
    prices, fundamentals = collect(tickers, api_key=api_key, cache_dir=cache_dir, force_refresh=force_refresh)
    fiscal_year_ends = collect_fiscal_year_ends(tickers, api_key=api_key, cache_dir=cache_dir)
    sec_facts = load_sec_facts(tickers, cache_dir=cache_dir)
    data_quality_report(prices, fundamentals)

    as_of_dates = build_as_of_dates(TRAIN_START)
    print(f"\nBuilding point-in-time panel: {len(as_of_dates)} snapshots "
          f"({as_of_dates[0].date()} ~ {as_of_dates[-1].date()})...")
    panel = add_percentile_scores(build_raw_panel(tickers, prices, fundamentals, universe, as_of_dates,
                                                  fiscal_year_ends=fiscal_year_ends, sec_facts=sec_facts))
    panel = _with_sentiment(panel, tickers, cache_dir)
    if panel.empty:
        raise RuntimeError("panel is empty — check that collection worked")

    latest = panel[panel["as_of"] == panel["as_of"].max()]
    print(f"  {len(panel)} rows, {panel['ticker'].nunique()} tickers; latest snapshot by size: "
          + ", ".join(f"{k} {v}" for k, v in latest["size_group"].value_counts().items()))
    print("  share of rows missing:")
    multiples = [spec["column"] for spec in FAIR_VALUE_TARGETS.values()]
    extras = sorted({f for spec in FAIR_VALUE_TARGETS.values() for f in spec.get("extra_features", ())} - {"industry"})
    for col in dict.fromkeys([*multiples, *FAIR_VALUE_FEATURES, *extras, *FAIR_VALUE_FEATURE_CANDIDATES, "dividend_yield"]):
        missing = panel[col].isna().mean()
        warn = "  <- almost empty: check the Finnhub field name (data.FINNHUB_FIELD_MAP)" if missing > 0.95 else ""
        print(f"    {col:24s} {missing:6.1%}{warn}")
    print(f"    dividend payers in latest snapshot: "
          f"{(panel.loc[panel['as_of'] == panel['as_of'].max(), 'dividend_yield'] > 0).mean():.0%}")
    _check_eps_basis(panel)

    Path(panel_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path)
    print(f"  saved {panel_path}")
    return panel


def _check_eps_basis(panel: pd.DataFrame) -> None:
    """normalized_pe is built from Finnhub's quarterly EPS and yfinance's
    quoted price. If both are on the same share basis, price / (sum of the
    last 4 quarterly EPS) matches Finnhub's own peTTM (rescaled to the same
    price); a stock split one side doesn't adjust for would show up here."""
    ok = (panel["eps_ttm"] > 0) & (panel["trailing_pe"] > 0)
    ratio = (panel.loc[ok, "price"] / panel.loc[ok, "eps_ttm"]) / panel.loc[ok, "trailing_pe"]
    if ratio.empty:
        return
    close = ((ratio - 1).abs() <= 0.15).mean()
    print(f"  EPS basis check: price / TTM EPS vs. Finnhub PER — median ratio {ratio.median():.2f}, "
          f"{close:.0%} of rows within 15%"
          + ("  <- check splits / share basis before trusting normalized_pe" if close < 0.8 else ""))


def _load(panel_path: str | Path) -> pd.DataFrame:
    path = Path(panel_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다 — 먼저 build를 실행하세요")
    return pd.read_parquet(path)


def compare_features(panel_path: str | Path = PANEL_PATH, drop_check: bool = False) -> pd.DataFrame:
    """Out-of-fold R^2 of each multiple for: the current features (each
    multiple with its extra_features), current + each candidate alone,
    current minus each feature in use, current + all candidates, and the
    current features under the other transform (winsor/rank, config
    FAIR_VALUE_FEATURE_TRANSFORM).
    Keep a candidate (for all multiples, or as one multiple's extra) only if
    it gains >= FAIR_VALUE_MIN_GAIN on train AND val; the "-feature" rows
    re-check the features already in use the same way (dropping one should
    cost at least that much where it is used — the second round after
    adopting several overlapping candidates at once; only with drop_check,
    since they double the run time). Test is for the final report."""
    panel = _load(panel_path)
    candidates = [c for c in FAIR_VALUE_FEATURE_CANDIDATES if c in panel.columns]
    in_use = {key: target_features(spec, FAIR_VALUE_FEATURES) for key, spec in FAIR_VALUE_TARGETS.items()}
    used = list(dict.fromkeys(f for feats in in_use.values() for f in feats))
    sets: dict = {"current": FAIR_VALUE_FEATURES}
    sets.update({f"+{c}": [*FAIR_VALUE_FEATURES, c] for c in candidates})
    if drop_check:
        sets.update({f"-{f}": {"drop": (f,)} for f in used})
    sets["+all"] = [*FAIR_VALUE_FEATURES, *candidates]
    other = "winsor" if FAIR_VALUE_FEATURE_TRANSFORM == "rank" else "rank"
    sets[f"{other}_transform"] = (FAIR_VALUE_FEATURES, other)
    print(f"Comparing {len(sets)} feature sets on {panel['as_of'].nunique()} snapshots (takes a few minutes)...")
    table = compare_feature_sets(panel, sets)
    delta = table.drop(columns=["sector_median", "current"]).sub(table["current"], axis=0)
    print("\n=== out-of-fold R^2 (mean per date) ===")
    print(table.round(3).T.to_string())
    print("\n=== change vs. current features (+candidate: positive = helps; -feature: negative = it was helping) ===")
    # 4 decimals: the bar is 0.01, and 0.0099 printed as 0.010 read as a pass (P/FCF asset_turnover, 2026-09-30)
    print(delta.round(4).T.to_string())

    def split_delta(name: str, key: str) -> tuple[float, float] | None:
        if key not in delta.index.get_level_values(0) or name not in delta.columns:
            return None
        return delta.loc[(key, "train"), name], delta.loc[(key, "val"), name]

    print(f"\n=== candidates that meet the bar (>= +{FAIR_VALUE_MIN_GAIN} on train AND val), per multiple ===")
    hits = []
    for name in (c for c in delta.columns if not c.startswith("-")):
        for key in FAIR_VALUE_TARGETS:
            d = split_delta(name, key)
            if d and d[0] >= FAIR_VALUE_MIN_GAIN and d[1] >= FAIR_VALUE_MIN_GAIN:
                hits.append(f"    {name:24s} {FAIR_VALUE_TARGETS[key]['label']:10s} {d[0]:+.3f} / {d[1]:+.3f}")
    print("\n".join(hits) if hits else "    (none)")

    if not drop_check:
        print("\n(features in use not re-checked: run with --drop-check after adopting candidates)")
        return table
    print(f"\n=== features in use: R^2 lost when dropped (train / val; bar {FAIR_VALUE_MIN_GAIN} on both) ===")
    for feat in used:
        cells = []
        for key, spec in FAIR_VALUE_TARGETS.items():
            d = split_delta(f"-{feat}", key)
            if feat not in in_use[key] or d is None:
                continue
            ok = -d[0] >= FAIR_VALUE_MIN_GAIN and -d[1] >= FAIR_VALUE_MIN_GAIN
            cells.append(f"{spec['label']} {-d[0]:+.3f}/{-d[1]:+.3f}{'' if ok else ' (below bar)'}")
        print(f"    {feat:24s} " + "; ".join(cells))
    return table


def evaluate(panel_path: str | Path = PANEL_PATH) -> dict[str, pd.DataFrame]:
    panel, diagnostics = add_fair_value(_load(panel_path))

    summary = summarize_diagnostics(diagnostics)
    print("=== 1. Fair-value model: how well fundamentals explain log multiples (out-of-fold, mean per date) ===")
    print("    r2_model must beat r2_sector_median, i.e. explain more than 'which sector is it'.")
    print(summary.round(3).to_string())

    print("\n=== 1c. Spread across dates and size of the error ===")
    print("    r2 per date: median [25%-75%] (the mean above can hide a few bad dates);")
    print("    error = |actual / fair - 1| per stock, out-of-fold: median and 90th percentile")
    rows = []
    for (key, split), g in diagnostics.groupby(["target", "split"]):
        err = np.expm1(panel.loc[panel["as_of"].map(split_of) == split, f"{key}_gap"].abs().dropna())
        rows.append({"target": key, "split": split,
                     "r2_median": g["r2_model"].median(), "r2_q25": g["r2_model"].quantile(0.25),
                     "r2_q75": g["r2_model"].quantile(0.75),
                     "error_median": err.median(), "error_p90": err.quantile(0.9)})
    spread = pd.DataFrame(rows).set_index(["target", "split"]).reindex(["train", "val", "test"], level="split")
    print(spread.round(3).to_string())

    labelled = panel[panel["valuation_gap"].notna() & (panel["n_gaps"] > 1)]
    print("\n=== 2. Combined verdict: do the multiples agree? ===")
    print(f"    mean share of views on the same side as the combined gap: {labelled['gap_agreement'].mean():.0%}")
    print(f"    rows where every view agrees: {(labelled['gap_agreement'] == 1).mean():.0%}")

    coefs = coefficient_summary(diagnostics)
    print("\n=== 3. What the market paid for (standardized Ridge coefficient, all dates) ===")
    print("    same_sign_share = share of dates with the same sign as the mean")
    print(coefs.pivot(index="feature", columns="target", values=["mean_coef", "same_sign_share"]).round(2).to_string())
    print("    spread across dates: median [25% ~ 75%]")
    spread_text = coefs.assign(
        text=[f"{m:+.2f} [{a:+.2f}~{b:+.2f}]" for m, a, b in zip(coefs["median_coef"], coefs["q25_coef"], coefs["q75_coef"])])
    print(spread_text.pivot(index="feature", columns="target", values="text").fillna("").to_string())

    _report_ttm_reliability(panel)

    print("\n=== 2c. Stability between consecutive snapshots (same tickers) ===")
    print("    rank_corr = Spearman of the combined gap; label_changed = share whose 할인/중립/프리미엄 changed;")
    print("    cheap_rich_flip = share that jumped straight between 할인 and 프리미엄")
    print(stability_summary(panel).round(3).to_string())

    by_size = "size_group" in panel.columns and panel["size_group"].nunique() > 1
    if by_size:
        print("\n=== 1b. Fit within each size group (out-of-fold R^2 vs. the group's own mean, mean per date) ===")
        print(r2_by_group(panel)["r2"].unstack("group").round(3).to_string())

    tests = gap_return_test(panel, by="size_group" if by_size else None)
    print("\n=== 4. Hypothesis test: did stocks called cheap later outperform? (not a model metric) ===")
    print("    ic = Spearman(cheapness, forward return) per date; spread = cheapest - most expensive quintile.")
    print("    Newey-West errors for overlapping horizons, BH-FDR across the whole table.")
    if tests.empty:
        print("    (not enough forward-return data)")
    else:
        if by_size:
            print("    group = within that size group only; survivorship bias: delisted small caps are missing (universe.py)")
        print(tests.round(4).to_string(index=False))
        n_sig = int(tests["significant_after_fdr"].sum())
        print(f"    {n_sig}/{len(tests)} significant after FDR")
        if by_size:
            per = tests.groupby("group")["significant_after_fdr"].agg(["sum", "size"])
            print("    by group: " + ", ".join(f"{g} {int(r['sum'])}/{int(r['size'])}" for g, r in per.iterrows()))
    return {"summary": summary, "coefficients": coefs, "gap_return_test": tests}


def _report_ttm_reliability(panel: pd.DataFrame) -> None:
    """Evidence for two open decisions (2026-09-29): should a
    fundamental_break_flag row get a verdict at all, and should the
    normalized PER (3-year EPS) replace PER? Tails = cheapest / most
    expensive 20% by valuation_gap within each as_of, like the labels."""
    print("\n=== 2b. Trailing-12-month reliability ===")
    if "eps_one_off_period" not in panel.columns:
        print("    (panel built before the break checks — rebuild)")
        return
    df = flag_fundamental_break(panel[panel["valuation_gap"].notna()])
    rank = df.groupby("as_of")["valuation_gap"].rank(pct=True)
    tail = pd.Series("middle", index=df.index).mask(rank <= 0.2, "cheap 20%").mask(rank > 0.8, "expensive 20%")
    share = df.groupby(tail)["fundamental_break_flag"].mean()
    print("    share of labelled rows with a fundamental break (one-off EPS / per-share break in the TTM window):")
    print("    " + ", ".join(f"{k} {v:.1%}" for k, v in share.items())
          + f"  (all {df['fundamental_break_flag'].mean():.1%})")
    if "pe_norm_gap" in df.columns:
        both = df.dropna(subset=["pe_gap", "pe_norm_gap"])
        for name, g in (("flagged", both[both["fundamental_break_flag"]]), ("not flagged", both[~both["fundamental_break_flag"]])):
            if len(g):
                print(f"    {name:12s} n={len(g):5d}  mean |gap| PER {g['pe_gap'].abs().mean():.2f}"
                      f" vs normalized PER {g['pe_norm_gap'].abs().mean():.2f}  (log units)")
        corr = both.groupby("as_of").apply(lambda g: g["pe_gap"].corr(g["pe_norm_gap"], method="spearman")).mean()
        print(f"    mean per-date Spearman(PER gap, normalized PER gap): {corr:.2f}")


def _market_context(as_of: pd.Timestamp, save: bool, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> dict | None:
    """market_context for the report date: printed as a short block and saved
    as real_data_output/market_context_<date>.json (for an LLM explanation).
    A download failure without a cache only prints a note."""
    import json

    try:
        ctx = market_context(load_market_data(cache_dir), as_of)
    except Exception as exc:  # network / file format: the stock report still runs
        print(f"\n-- 시장 전체 맥락: 계산 못 함 ({type(exc).__name__}: {exc}) --")
        return None
    if ctx is None:
        print(f"\n-- 시장 전체 맥락: {pd.Timestamp(as_of).date()} 직전 달의 데이터가 아직 없음 --")
        return None
    print("\n-- 시장 전체 맥락 (서술용, 예측 아님; 종목 라벨과 무관) --")
    for line in context_lines(ctx):
        print("  " + line)
    if save:
        out = OUTPUT_DIR / f"market_context_{pd.Timestamp(as_of).date()}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  saved {out}")
    return ctx


def screen(panel_path: str | Path = PANEL_PATH, as_of: str | None = None, ticker: str | None = None,
           save: bool = True) -> pd.DataFrame:
    report = report_at(screen_panel(_load(panel_path)), as_of)
    if report.empty:
        raise ValueError(f"no rows for as_of={as_of}")
    date = report["as_of"].iloc[0].date()
    _market_context(report["as_of"].iloc[0], save)

    if ticker:
        match = report[report["ticker"] == ticker.upper()]
        if match.empty:
            print(f"{ticker}: {date} 기준 데이터 없음")
        else:
            row = match.iloc[0]
            rank = f"{row['cheapness_rank']:.0f}" if pd.notna(row["cheapness_rank"]) else "-"
            print(f"{row['ticker']} ({row['sector']}, {row['size_group']}) {date}: {row['valuation_view']} (할인 순위 {rank}/100)")
            print(row["explanation"])
            for col in ("meme_reason", "value_trap_reason", "transition_reason", "report_lag_reason",
                        "fundamental_break_reason", "single_view_reason"):
                if row[col]:
                    print(f"※ {row[col]}")
        return report

    counts = report["valuation_label"].value_counts()
    print(f"Screening report {date} ({len(report)} tickers): " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    if report["size_group"].nunique() > 1:
        mix = pd.crosstab(report["size_group"], report["valuation_label"])
        print("    labels by size group:\n" + mix.to_string())
    split = report[report["label_detail"].fillna("") != ""]
    if not split.empty:
        # one row per reason; the "(...)" note variants are counted, not listed apart
        detail = split["label_detail"]
        for note in DETAIL_NOTES:
            detail = detail.str.replace(note, "", regex=False)
        view = (split["valuation_label"] + " · " + detail).rename("세부 라벨")
        why = pd.crosstab(view, split["size_group"].rename("규모"))
        why["일회성 가능"] = split["label_detail"].str.contains(ONE_OFF_NOTE, regex=False).groupby(view).sum()
        why["기저 효과 가능"] = split["label_detail"].str.contains(BASE_EFFECT_NOTE, regex=False).groupby(view).sum()
        print("    why (priced-in vs. delivered growth, screening.add_label_detail):\n" + why.to_string())
    pct = lambda s: s.map(lambda v: f"{v:+.0%}" if pd.notna(v) else "")
    table = report.assign(
        rank=report["cheapness_rank"].round(0),
        gap=pct(report["valuation_gap_pct"]),
        agree=[f"{a * n:.0f}/{n:.0f}" if pd.notna(a) else "" for a, n in zip(report["gap_agreement"], report["n_gaps"])],
        **{spec["label"]: pct(report[f"{key}_gap"]) for key, spec in FAIR_VALUE_TARGETS.items()},
        quality=report["quality_score"].round(0),
        priced_in=pct(report["implied_excess_growth"]),
        earn_3y=pct(report["earnings_cagr_3y"]),
    )
    cols = ["ticker", "sector", "size_group", "valuation_view", "rank", "gap", "agree",
            *[spec["label"] for spec in FAIR_VALUE_TARGETS.values()], "priced_in", "earn_3y", "quality"]
    median_growth = report["earnings_cagr_3y_median"].dropna()
    print("    (gap = combined; agree = views on the same side; per-multiple columns = actual vs. fair;\n"
          f"     priced_in = yearly EPS growth above the median stock the PER needs for {IMPLIED_GROWTH_YEARS} years;\n"
          "     earn_3y = realized earnings growth per year, last 3 years (EPS, or net income without EPS)"
          + (f" (median stock {median_growth.iloc[0]:+.0%})" if len(median_growth) else "") + ")")
    print("\n-- 큰 할인 상위 15 --")
    print(table[cols].head(15).to_string(index=False))
    print("\n-- 큰 프리미엄 상위 15 --")
    print(table[table["valuation_label"] == "큰 프리미엄"][cols].tail(15).iloc[::-1].to_string(index=False))
    loss = report[report["loss_cheapness_rank"].notna()].sort_values("loss_cheapness_rank", ascending=False)
    if not loss.empty:
        loss_table = loss.assign(rank=loss["loss_cheapness_rank"].round(0), gap=pct(loss["loss_valuation_gap_pct"]),
                                 basis=loss["loss_valuation_basis"])[["ticker", "sector", "size_group", "valuation_label",
                                                                      "rank", "gap", "basis"]]
        print(f"\n-- 적자 기업 {len(loss)}개 (PER 없음, 주로 PSR·PBR; rank = 같은 배수 기준 전체 종목 중 백분위) --")
        print("  할인 쪽 10:\n" + loss_table.head(10).to_string(index=False))
        print("  프리미엄 쪽 10:\n" + loss_table.tail(10).iloc[::-1].to_string(index=False))

    # flagged rows nearest the label extremes first; the full lists are in the CSV
    extremeness = (report["cheapness_rank"] - 50).abs().fillna(-1)
    for flag, reason, title in [("meme_flag", "meme_reason", "단기 가격·거래량 이상"),
                                ("value_trap_flag", "value_trap_reason", "지속 할인·저성장 경고"),
                                ("transition_flag", "transition_reason", "큰 할인→큰 프리미엄 전환"),
                                ("report_lag_flag", "report_lag_reason", "재무 기준일 이후 주가 급변"),
                                ("fundamental_break_flag", "fundamental_break_reason", "최근 12개월 재무 단절"),
                                ("single_view_flag", "single_view_reason", "한 가지 배수로만 판단"),
                                ("financial_risk_flag", "financial_risk_reason", "적자 + 재무 위험"),
                                ("heavy_debt_flag", "heavy_debt_reason", "빚 많은 흑자 기업")]:
        hits = report[report[flag]]
        if not hits.empty:
            print(f"\n-- {title} ({len(hits)}) --")
            shown = hits.loc[extremeness[hits.index].sort_values(ascending=False, kind="stable").index[:MAX_FLAGGED_SHOWN]]
            for _, row in shown.iterrows():
                print(f"  {row['ticker']:6s} {row['valuation_label']:4s} {row[reason]}")
            if len(hits) > MAX_FLAGGED_SHOWN:
                print(f"  ... +{len(hits) - MAX_FLAGGED_SHOWN} more (CSV column {flag})")

    if save:
        out = OUTPUT_DIR / f"screening_{date}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"\nsaved {out}")
    return report


def export(panel_path: str | Path = PANEL_PATH, as_of: str | None = None, out_dir: str | Path = OUTPUT_DIR / "llm") -> Path:
    """Per-stock JSON for an LLM explanation (llm_context): screens the panel,
    adds the market context for that date, writes <out_dir>/<date>/."""
    screened = screen_panel(_load(panel_path))
    target = pd.Timestamp(as_of) if as_of else screened["as_of"].max()
    market = _market_context(target, save=False)
    folder = export_date(screened, target, out_dir, market)
    n_stocks = len(list(folder.glob("*.json"))) - 1 - (market is not None)  # minus index / market files
    print(f"\nexported {n_stocks} stocks -> {folder}")
    return folder


def explain(as_of: str | None = None, tickers: list[str] | None = None, submit: bool = False, collect: bool = False,
            model: str | None = None, effort: str | None = None, out_dir: str | Path = OUTPUT_DIR / "llm") -> None:
    """Korean explanations of an exported date (llm_explain): --tickers runs
    them one by one now; --submit sends every stock as a Message Batch;
    --collect fetches a finished batch. Needs Claude API credentials in the
    environment (ANTHROPIC_API_KEY or `ant auth login`)."""
    import llm_explain

    dates = sorted(p.name for p in Path(out_dir).iterdir() if p.is_dir()) if Path(out_dir).exists() else []
    folder = Path(out_dir) / (as_of or (dates[-1] if dates else ""))
    if not (folder / "index.json").exists():
        raise SystemExit(f"{folder} has no export — run `python src/main.py export` first")
    kw = {"model": model or llm_explain.MODEL}
    if tickers:
        for t in tickers:
            r = llm_explain.explain_one(folder, t.upper(), effort=effort or llm_explain.EFFORT, **kw)
            e = r["explanation"] or {}
            verdict = "통과" if r["check"]["passed"] else "검수 실패: " + "; ".join(r["check"]["issues"])
            print(f"\n=== {r['ticker']} ({verdict}) ===")
            print(e.get("headline", ""))
            print(e.get("summary", ""))
            for line in e.get("reasons", []):
                print("  - " + line)
            for line in e.get("cautions", []):
                print("  ※ " + line)
            if e.get("market_note"):
                print("  시장: " + e["market_note"])
            print(f"  (tokens in {r['usage']['input_tokens']} + cache {r['usage']['cache_read_input_tokens']}, out {r['usage']['output_tokens']})")
    elif submit:
        print(f"batch submitted: {llm_explain.submit_batch(folder, effort=effort or llm_explain.EFFORT, **kw)}")
    elif collect:
        print(llm_explain.collect_batch(folder, **kw))


def publish(api_key: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR, panel_path: str | Path = PANEL_PATH,
            out_dir: str | Path = SITE_DIR, fundamentals_max_age: float = 14, keep: int = 14) -> Path:
    """The scheduled job behind the web/app (publish.py): fresh prices, Finnhub
    fundamentals only for caches older than `fundamentals_max_age` days (about
    1/14 of the universe a day, well inside 50 calls/min), the snapshots the
    saved panel lacks, every stock screened at once, site files written.
    Needs a panel from `build` first."""
    import time

    from publish import update_panel, write_site

    start = time.time()
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    universe = _universe(cache_dir)
    tickers = list(universe)
    prices, fundamentals = collect(tickers, api_key=api_key, cache_dir=cache_dir, price_max_age_days=0,
                                   fundamentals_max_age_days=fundamentals_max_age)
    panel, added = update_panel(
        _load(panel_path), build_as_of_dates(TRAIN_START), tickers, prices=prices, fundamentals=fundamentals,
        universe=universe, fiscal_year_ends=collect_fiscal_year_ends(tickers, api_key=api_key, cache_dir=cache_dir),
        sec_facts=load_sec_facts(tickers, cache_dir=cache_dir))
    panel = _with_sentiment(panel, tickers, cache_dir)
    print(f"panel: {len(panel)} rows, added {', '.join(str(d.date()) for d in added) or 'nothing'}")
    tmp = Path(panel_path).with_suffix(".tmp")
    panel.to_parquet(tmp)
    tmp.replace(panel_path)

    screened = screen_panel(panel)
    as_of = screened["as_of"].max()
    latest = write_site(screened, out_dir, _market_context(as_of, save=False, cache_dir=cache_dir), keep=keep)
    counts = screened.loc[screened["as_of"] == as_of, "valuation_label"].value_counts()
    print(f"\npublished {as_of.date()} ({int(counts.sum())} stocks) -> {latest}  [{time.time() - start:.0f}s]")
    print("  " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    return latest


# HON: all 5 views -77% on 2026-09-29 — a one-off gain + per-share break in
# Finnhub's data (screening.flag_fundamental_break); prices matched yfinance.
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
    c.add_argument("--drop-check", action="store_true", help="also drop each feature in use once (slower)")
    e = sub.add_parser("evaluate", help="fair-value model quality + gap-vs-return test")
    s = sub.add_parser("screen", help="screening report")
    s.add_argument("--as-of")
    s.add_argument("--ticker")
    x = sub.add_parser("export", help="per-stock JSON for an LLM explanation")
    x.add_argument("--as-of")
    y = sub.add_parser("explain", help="Korean explanations with Claude (needs API credentials)")
    y.add_argument("--as-of")
    y.add_argument("--tickers", nargs="+")
    y.add_argument("--submit", action="store_true")
    y.add_argument("--collect", action="store_true")
    y.add_argument("--model")
    y.add_argument("--effort")
    u = sub.add_parser("publish", help="scheduled job: refresh prices, screen every stock, write site files")
    u.add_argument("--api-key")
    u.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    u.add_argument("--out-dir", default=str(SITE_DIR))
    u.add_argument("--fundamentals-max-age", type=float, default=14, help="re-fetch fundamentals older than this many days")
    u.add_argument("--keep", type=int, default=14, help="date folders to keep")
    for p in (b, v, c, e, s, x, u):
        p.add_argument("--panel", default=str(PANEL_PATH))
    args = parser.parse_args()

    if args.command == "build":
        build(api_key=args.api_key, cache_dir=args.cache_dir, force_refresh=args.force_refresh, panel_path=args.panel)
    elif args.command == "verify-multiples":
        verify_multiples(args.tickers, panel_path=args.panel)
    elif args.command == "compare-features":
        compare_features(args.panel, drop_check=args.drop_check)
    elif args.command == "evaluate":
        evaluate(args.panel)
    elif args.command == "publish":
        publish(args.api_key, args.cache_dir, args.panel, args.out_dir, args.fundamentals_max_age, args.keep)
    elif args.command == "explain":
        explain(args.as_of, args.tickers, args.submit, args.collect, args.model, args.effort)
    elif args.command == "export":
        export(args.panel, as_of=args.as_of)
    else:
        screen(args.panel, as_of=args.as_of, ticker=args.ticker)


if __name__ == "__main__":
    main()
