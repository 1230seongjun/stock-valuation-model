"""
Synthetic-data tests for the whole pipeline. They check the LOGIC (no
look-ahead, the fair-value model recovers a relationship we planted, labels
and flags fire on the cases built for them), not anything about real
markets.

Run:  python tests/test_pipeline.py     (or: pytest tests/)
Colab (all .py flat in /content):  !python test_pipeline.py
"""
import sys
from pathlib import Path

try:
    _this_dir = Path(__file__).resolve().parent
except NameError:  # pasted into a notebook cell
    _this_dir = Path.cwd()
for _candidate in (_this_dir, _this_dir.parent / "src"):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

import numpy as np
import pandas as pd

from config import FAIR_VALUE_FEATURES, HORIZONS_MONTHS
from fair_value import add_fair_value, gap_return_test, summarize_diagnostics
from features import add_percentile_scores, add_revenue_growth_yoy, build_as_of_dates, build_raw_panel
from screening import add_labels, classify_valuation_transition, flag_meme_stock, flag_value_trap, report_at, screen

SECTORS = ["Technology", "Healthcare", "Financials", "Utilities"]
SECTOR_EFFECT = {"Technology": 0.4, "Healthcare": 0.2, "Financials": -0.3, "Utilities": -0.1}


# ---------------------------------------------------------------------------
# synthetic raw data (prices + Finnhub-shaped fundamentals)
# ---------------------------------------------------------------------------
def make_raw_data(n_tickers: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    universe = {t: {"name": t, "sector": SECTORS[i % len(SECTORS)]} for i, t in enumerate(tickers)}
    dates = pd.date_range("2016-01-01", "2024-06-28", freq="B")
    periods = pd.date_range("2015-03-31", "2024-03-31", freq="QE")
    prices, fundamentals = {}, {}
    for t in tickers:
        ret = rng.normal(0.0003, 0.015, len(dates))
        prices[t] = pd.DataFrame({
            "date": dates,
            "close": 50 * np.exp(np.cumsum(ret)),
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
        })
        growth = rng.normal(0.06, 0.04)
        n = len(periods)
        fundamentals[t] = pd.DataFrame({
            "period": periods,
            "trailing_pe": np.clip(18 + rng.normal(0, 4, n), 3, None),
            "price_to_book": np.clip(3 + rng.normal(0, 0.8, n), 0.3, None),
            "return_on_equity": 0.12 + rng.normal(0, 0.05, n),
            "debt_to_equity": np.clip(1 + rng.normal(0, 0.3, n), 0, None),
            "operating_margin": 0.15 + rng.normal(0, 0.04, n),
            "eps": np.clip(2 + rng.normal(0, 0.3, n), 0.1, None),
            "payout_ratio_ttm": np.clip(0.35 + rng.normal(0, 0.1, n), 0, 1),
            "sales_per_share": 20 * np.exp(np.cumsum(np.log1p(growth / 4) + rng.normal(0, 0.005, n))),
        })
    return tickers, universe, prices, fundamentals


def build_synthetic_panel(**kwargs) -> pd.DataFrame:
    tickers, universe, prices, fundamentals = make_raw_data(**kwargs)
    as_of = list(pd.date_range("2018-01-01", "2024-04-01", freq="QS"))
    return add_percentile_scores(build_raw_panel(tickers, prices, fundamentals, universe, as_of))


# ---------------------------------------------------------------------------
# a cross-section where we KNOW the fair multiple and the mispricing
# ---------------------------------------------------------------------------
def make_fair_value_panel(n_tickers: int = 200, n_dates: int = 3, seed: int = 1):
    """log(PE) and log(PB) are a linear function of the features + sector +
    small noise; the first 15 tickers trade 45% below that, the next 15 are
    ~57% above it. The model has never seen the planted offsets."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2020-01-01", periods=n_dates, freq="QS"):
        for i in range(n_tickers):
            sector = SECTORS[i % len(SECTORS)]
            f = {
                "return_on_equity": rng.normal(0.15, 0.06),
                "operating_margin": rng.normal(0.15, 0.05),
                "revenue_growth_yoy": rng.normal(0.06, 0.05),
                "debt_to_equity": abs(rng.normal(1.0, 0.4)),
                "payout_ratio_ttm": float(np.clip(rng.normal(0.4, 0.15), 0, 1)),
                "volatility_63d": abs(rng.normal(0.3, 0.08)),
            }
            fair_log_pe = 2.8 + 4.0 * f["revenue_growth_yoy"] + 0.6 * f["payout_ratio_ttm"] + SECTOR_EFFECT[sector]
            fair_log_pb = 0.5 + 5.0 * f["return_on_equity"] + 0.5 * SECTOR_EFFECT[sector]
            offset = -0.6 if i < 15 else (0.45 if i < 30 else 0.0)
            noise_pe, noise_pb = rng.normal(0, 0.08), rng.normal(0, 0.08)
            rows.append({
                "as_of": d, "ticker": f"T{i:03d}", "sector": sector, **f,
                "trailing_pe": float(np.exp(fair_log_pe + offset + noise_pe)),
                "price_to_book": float(np.exp(fair_log_pb + offset + noise_pb)),
                "eps": 2.0, "price": 100.0,
            })
    return pd.DataFrame(rows)


def _labelled(panel: pd.DataFrame) -> pd.DataFrame:
    """add_labels needs the context percentile columns; fill them neutrally."""
    df = panel.copy()
    for col in ("trailing_pe", "price_to_book", "dividend_yield", "return_on_equity", "debt_to_equity",
                "operating_margin", "revenue_growth_yoy", "ma50_vs_ma200", "pct_from_52w_high"):
        if f"{col}_pct" not in df.columns:
            df[f"{col}_pct"] = 50.0
    return add_labels(df)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_revenue_growth_derivation():
    periods = pd.date_range("2018-03-31", periods=12, freq="QE")
    sales = 10 * (1.08 ** (np.arange(12) / 4))  # exactly +8% per year
    derived = add_revenue_growth_yoy(pd.DataFrame({"period": periods, "sales_per_share": sales}))
    growth = derived["revenue_growth_yoy"].dropna()
    assert len(growth) == 8, "first 4 quarters have no prior-year match"
    assert np.allclose(growth, 0.08, atol=1e-9)


def test_as_of_dates_include_today():
    today = pd.Timestamp("2026-09-23")
    dates = build_as_of_dates(pd.Timestamp("2026-01-01"), today)
    assert dates[-1] == today and dates[-2] == pd.Timestamp("2026-07-01")


def test_no_look_ahead():
    """Rows at or before a cutoff must not change when everything AFTER the
    cutoff (future prices, not-yet-available fundamentals) is scrambled.
    Forward returns are the only columns allowed to differ."""
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=8)
    as_of = list(pd.date_range("2018-01-01", "2024-04-01", freq="QS"))
    cutoff = pd.Timestamp("2021-07-01")
    base = build_raw_panel(tickers, prices, fundamentals, universe, as_of)

    rng = np.random.default_rng(99)
    prices2 = {}
    for t, df in prices.items():
        df = df.copy()
        future = df["date"] > cutoff
        df.loc[future, "close"] *= rng.uniform(0.2, 5.0, future.sum())
        df.loc[future, "volume"] *= rng.uniform(0.1, 10.0, future.sum())
        prices2[t] = df
    fundamentals2 = {}
    for t, df in fundamentals.items():
        df = df.copy()
        unknown = df["period"] + pd.Timedelta(days=45) > cutoff
        for col in df.columns.drop("period"):
            df.loc[unknown, col] *= rng.uniform(0.2, 5.0, unknown.sum())
        fundamentals2[t] = df
    scrambled = build_raw_panel(tickers, prices2, fundamentals2, universe, as_of)

    fwd = [f"fwd_return_{h}m" for h in HORIZONS_MONTHS]
    known_a = base[base["as_of"] <= cutoff].drop(columns=fwd).reset_index(drop=True)
    known_b = scrambled[scrambled["as_of"] <= cutoff].drop(columns=fwd).reset_index(drop=True)
    pd.testing.assert_frame_equal(known_a, known_b)
    later = base["as_of"] > cutoff
    assert not base.loc[later, "price"].equals(scrambled.loc[later, "price"]), "scramble had no effect"


def test_no_rows_before_listing_or_after_delisting():
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=2)
    prices["SYN000"] = prices["SYN000"][prices["SYN000"]["date"] <= "2020-02-14"]  # "delisted"
    prices["SYN001"] = prices["SYN001"][prices["SYN001"]["date"] >= "2019-06-03"]  # "IPO"
    as_of = list(pd.date_range("2019-01-01", "2021-01-01", freq="QS"))
    panel = build_raw_panel(tickers, prices, fundamentals, universe, as_of)
    gone = panel[panel["ticker"] == "SYN000"]["as_of"]
    new = panel[panel["ticker"] == "SYN001"]["as_of"]
    assert gone.max() == pd.Timestamp("2020-01-01"), "no stale price carried past the last trade"
    assert new.min() == pd.Timestamp("2019-07-01"), "no rows before the first trade"


def test_multiples_rescaled_to_as_of_price():
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    close = np.where(dates <= "2020-03-31", 100.0, 150.0)
    prices = {"AAA": pd.DataFrame({"date": dates, "close": close, "volume": 1e6})}
    fundamentals = {"AAA": pd.DataFrame({
        "period": [pd.Timestamp("2020-03-31")], "trailing_pe": [20.0], "price_to_book": [4.0],
        "return_on_equity": [0.2], "debt_to_equity": [1.0], "operating_margin": [0.1],
        "eps": [5.0], "payout_ratio_ttm": [0.3], "sales_per_share": [50.0],
    })}
    panel = build_raw_panel(["AAA"], prices, fundamentals, {"AAA": {"sector": "X"}}, [pd.Timestamp("2020-07-01")])
    row = panel.iloc[0]
    assert row["trailing_pe_reported"] == 20.0 and np.isclose(row["trailing_pe"], 30.0)
    assert np.isclose(row["price_to_book"], 6.0)
    assert np.isclose(row["dividend_yield"], 0.3 * 5.0 / 150.0)


def test_fair_value_recovers_planted_mispricing():
    panel, diag = add_fair_value(make_fair_value_panel())
    summary = summarize_diagnostics(diag)
    for key in ("pe", "pb"):
        r2_model = summary.loc[key, "r2_model"].mean()
        r2_base = summary.loc[key, "r2_sector_median"].mean()
        assert r2_model > r2_base + 0.1, f"{key}: model R2 {r2_model:.2f} should clearly beat sector median {r2_base:.2f}"

    labelled = _labelled(panel)
    cheap = labelled[labelled["ticker"].isin([f"T{i:03d}" for i in range(15)])]
    rich = labelled[labelled["ticker"].isin([f"T{i:03d}" for i in range(15, 30)])]
    assert (cheap["valuation_label"] == "저평가").mean() > 0.8
    assert (rich["valuation_label"] == "고평가").mean() > 0.8
    assert np.isclose(np.expm1(cheap["valuation_gap"]).median(), np.expm1(-0.6), atol=0.1)


def test_fair_value_is_out_of_fold():
    """A stock's own multiple must not move its own fair multiple."""
    panel = make_fair_value_panel(n_dates=1)
    before, _ = add_fair_value(panel)
    changed = panel.copy()
    changed.loc[changed["ticker"] == "T100", "trailing_pe"] *= 1.8
    after, _ = add_fair_value(changed)
    b = before.set_index("ticker").loc["T100"]
    a = after.set_index("ticker").loc["T100"]
    assert np.isclose(a["fair_pe"], b["fair_pe"], rtol=1e-12)
    assert np.isclose(a["pe_gap"], b["pe_gap"] + np.log(1.8))


def test_contributions_add_up():
    panel, _ = add_fair_value(make_fair_value_panel(n_dates=1))
    contrib_cols = [f"pe_contrib_{d}" for d in [*FAIR_VALUE_FEATURES, "sector"]]
    implied_intercept = np.log(panel["fair_pe"]) - panel[contrib_cols].sum(axis=1)
    # the per-fold intercept (training-fold mean of log PE) is nearly constant
    assert implied_intercept.std() < 0.05


def test_out_of_range_multiples_get_no_gap():
    panel = make_fair_value_panel(n_dates=1)
    panel.loc[panel["ticker"] == "T050", "trailing_pe"] = 400.0   # above max
    panel.loc[panel["ticker"] == "T051", "price_to_book"] = 0.01  # below min (data error)
    panel.loc[panel["ticker"] == "T052", ["trailing_pe", "eps"]] = [np.nan, -1.0]  # loss
    out, _ = add_fair_value(panel)
    out = _labelled(out).set_index("ticker")
    assert pd.isna(out.loc["T050", "pe_gap"]) and pd.notna(out.loc["T050", "pb_gap"])
    assert out.loc["T050", "valuation_basis"] == "PBR"
    assert pd.isna(out.loc["T051", "pb_gap"])
    assert out.loc["T052", "loss_flag"] and out.loc["T052", "valuation_label"] == "판단 보류(적자)"
    assert pd.isna(out.loc["T052", "cheapness_rank"])


def test_transition_classification():
    rows = [  # ticker, as_of, cheapness_rank, price, eps
        ("A", "2024-01-01", 90, 100.0, 5.00), ("A", "2024-04-01", 10, 135.0, 5.05),  # 주가 상승형
        ("B", "2024-01-01", 90, 100.0, 5.00), ("B", "2024-04-01", 10, 101.0, 3.60),  # 실적 악화형
        ("C", "2024-01-01", 90, 100.0, 5.00), ("C", "2024-04-01", 10, 135.0, 3.60),  # 복합형
        ("D", "2024-01-01", 90, 100.0, 5.00), ("D", "2024-04-01", 10, 75.0, 5.05),   # 가격 급락형
        ("E", "2024-01-01", 90, 100.0, 5.00), ("E", "2024-04-01", 10, 102.0, 5.10),  # 불분명
        ("F", "2024-01-01", 90, 100.0, 5.00), ("F", "2024-04-01", 50, 103.0, 5.10),  # no flip
    ]
    panel = pd.DataFrame(rows, columns=["ticker", "as_of", "cheapness_rank", "price", "eps"])
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    result = classify_valuation_transition(panel)
    flagged = result[result["transition_flag"]].set_index("ticker")["transition_type"].to_dict()
    assert flagged == {"A": "주가 상승형", "B": "실적 악화형", "C": "복합형", "D": "가격 급락형", "E": "불분명"}


def test_value_trap_and_meme_flags():
    dates = pd.date_range("2023-01-01", periods=5, freq="QS")
    panel = pd.DataFrame({
        "ticker": ["TRAP"] * 5 + ["FRESH"] * 5,
        "as_of": list(dates) * 2,
        "valuation_label": ["저평가"] * 5 + ["중립"] * 4 + ["저평가"],
        "revenue_growth_yoy_pct": [30.0] * 10,
        "price_spike_5d": [0.0] * 9 + [0.25],
        "volume_zscore_63d": [0.0] * 9 + [4.0],
    })
    trapped = flag_value_trap(panel).set_index(["ticker", "as_of"])
    assert trapped.loc[("TRAP", dates[3]), "value_trap_flag"] and trapped.loc[("TRAP", dates[4]), "value_trap_flag"]
    assert not trapped.loc[("TRAP", dates[2]), "value_trap_flag"], "only 3 cheap snapshots so far"
    assert not trapped.loc[("FRESH", dates[-1]), "value_trap_flag"], "newly cheap is not a trap"
    memes = flag_meme_stock(panel)
    assert memes["meme_flag"].sum() == 1 and "급등" in memes.loc[memes["meme_flag"], "meme_reason"].iloc[0]


def test_end_to_end():
    panel = build_synthetic_panel(n_tickers=80)  # >= FAIR_VALUE_MIN_ROWS per date
    screened = screen(panel)
    report = report_at(screened)
    assert len(report) == panel["ticker"].nunique()
    assert report["valuation_label"].isin(["저평가", "중립", "고평가", "데이터 부족", "판단 보류(적자)"]).all()
    assert report["explanation"].str.len().gt(0).all()
    labelled = report["valuation_label"].isin(["저평가", "고평가"])
    assert 0.25 < labelled.mean() < 0.55, "top/bottom 20% should be labelled"
    tests = gap_return_test(screened)
    assert {"split", "horizon_m", "test", "p_value_fdr"}.issubset(tests.columns)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # report every test, then exit non-zero
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failed" if failures else "\nall passed")
    sys.exit(1 if failures else 0)
