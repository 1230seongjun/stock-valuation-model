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
from fair_value import add_fair_value, compare_feature_sets, gap_return_test, r2_by_group, summarize_diagnostics
from features import (
    _dividend_features,
    add_fundamental_trends,
    add_percentile_scores,
    build_as_of_dates,
    build_raw_panel,
)
from screening import (
    add_labels,
    classify_valuation_transition,
    flag_meme_stock,
    flag_report_lag,
    flag_single_view,
    flag_value_trap,
    report_at,
    screen,
)

SECTORS = ["Technology", "Healthcare", "Financials", "Utilities"]
SECTOR_EFFECT = {"Technology": 0.4, "Healthcare": 0.2, "Financials": -0.3, "Utilities": -0.1}
MULTIPLES = ["trailing_pe", "price_to_book", "price_to_sales", "ev_to_ebitda", "price_to_fcf"]


# ---------------------------------------------------------------------------
# synthetic raw data (prices + dividends + Finnhub-shaped fundamentals)
# ---------------------------------------------------------------------------
def make_raw_data(n_tickers: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    universe = {t: {"name": t, "sector": SECTORS[i % len(SECTORS)]} for i, t in enumerate(tickers)}
    dates = pd.date_range("2016-01-01", "2024-06-28", freq="B")
    periods = pd.date_range("2015-03-31", "2024-03-31", freq="QE")
    prices, fundamentals = {}, {}
    for i, t in enumerate(tickers):
        ret = rng.normal(0.0003, 0.015, len(dates))
        close_raw = 50 * np.exp(np.cumsum(ret))
        dividend = np.zeros(len(dates))
        if i % 2 == 0:  # half the universe pays a growing quarterly dividend
            pay_days = np.arange(40, len(dates), 63)
            dividend[pay_days] = 0.30 * 1.02 ** np.arange(len(pay_days))
        prices[t] = pd.DataFrame({
            "date": dates,
            "close": close_raw * np.linspace(0.8, 1.0, len(dates)),  # dividend-adjusted history sits lower
            "close_raw": close_raw,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
            "dividend": dividend,
        })
        growth = rng.normal(0.06, 0.04)
        n = len(periods)
        fundamentals[t] = pd.DataFrame({
            "period": periods,
            "trailing_pe": np.clip(18 + rng.normal(0, 4, n), 3, None),
            "price_to_book": np.clip(3 + rng.normal(0, 0.8, n), 0.3, None),
            "price_to_sales": np.clip(2.5 + rng.normal(0, 0.6, n), 0.2, None),
            "ev_to_ebitda": np.clip(11 + rng.normal(0, 2.5, n), 2, None),
            "price_to_fcf": np.clip(20 + rng.normal(0, 5, n), 3, None),
            "return_on_equity": 0.12 + rng.normal(0, 0.05, n),
            "roic": 0.10 + rng.normal(0, 0.04, n),
            "gross_margin": 0.40 + rng.normal(0, 0.08, n),
            "operating_margin": 0.15 + rng.normal(0, 0.04, n),
            "fcf_margin": 0.10 + rng.normal(0, 0.04, n),
            "debt_to_equity": np.clip(1 + rng.normal(0, 0.3, n), 0, None),
            "net_debt_to_capital": 0.3 + rng.normal(0, 0.1, n),
            "current_ratio": np.clip(1.5 + rng.normal(0, 0.4, n), 0.3, None),
            "asset_turnover": np.clip(0.8 + rng.normal(0, 0.2, n), 0.1, None),
            "sga_to_sales": np.clip(0.2 + rng.normal(0, 0.05, n), 0.01, None),
            "eps": np.clip(2 + rng.normal(0, 0.3, n), 0.1, None),
            "payout_ratio_ttm": np.clip(0.35 + rng.normal(0, 0.1, n), 0, 1),
            "sales_per_share": 5 * np.exp(np.cumsum(np.log1p(growth / 4) + rng.normal(0, 0.005, n))),
            "enterprise_value": np.full(n, 1200.0),
            "book_value": np.full(n, 300.0),
        })
    return tickers, universe, prices, fundamentals


def build_synthetic_panel(**kwargs) -> pd.DataFrame:
    tickers, universe, prices, fundamentals = make_raw_data(**kwargs)
    as_of = list(pd.date_range("2018-01-01", "2024-04-01", freq="QS"))
    return add_percentile_scores(build_raw_panel(tickers, prices, fundamentals, universe, as_of))


# ---------------------------------------------------------------------------
# a cross-section where we KNOW the fair multiples and the mispricing
# ---------------------------------------------------------------------------
def make_fair_value_panel(n_tickers: int = 200, n_dates: int = 3, seed: int = 1):
    """Every log multiple is a linear function of the features + sector +
    small noise; the first 15 tickers trade 45% below that on every multiple,
    the next 15 are ~57% above it. The model never sees the planted offsets."""
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
            s = SECTOR_EFFECT[sector]
            fair = {
                "trailing_pe": 2.8 + 4.0 * f["revenue_growth_yoy"] + 0.6 * f["payout_ratio_ttm"] + s,
                "price_to_book": 0.5 + 5.0 * f["return_on_equity"] + 0.5 * s,
                "price_to_sales": 0.2 + 6.0 * f["operating_margin"] + 3.0 * f["revenue_growth_yoy"] + s,
                "ev_to_ebitda": 2.2 + 3.0 * f["revenue_growth_yoy"] - 0.3 * f["debt_to_equity"] + s,
                "price_to_fcf": 2.9 + 3.0 * f["revenue_growth_yoy"] + 0.5 * f["payout_ratio_ttm"] + s,
            }
            offset = -0.6 if i < 15 else (0.45 if i < 30 else 0.0)
            noise = rng.normal(0, 0.08, len(fair))
            rows.append({
                "as_of": d, "ticker": f"T{i:03d}", "sector": sector, **f,
                **{col: float(np.exp(v + offset + e)) for (col, v), e in zip(fair.items(), noise)},
                "eps": 2.0, "price": 100.0,
            })
    return pd.DataFrame(rows)


def _labelled(panel: pd.DataFrame) -> pd.DataFrame:
    """add_labels needs the context percentile columns; fill them neutrally."""
    df = panel.copy()
    for col in (*MULTIPLES, "dividend_yield", "return_on_equity", "debt_to_equity",
                "operating_margin", "revenue_growth_yoy", "ma50_vs_ma200", "pct_from_52w_high"):
        if f"{col}_pct" not in df.columns:
            df[f"{col}_pct"] = 50.0
    return add_labels(df)


def screen_ready(fitted: pd.DataFrame) -> pd.DataFrame:
    """A make_fair_value_panel after add_fair_value, with every column
    report_at reads (labels, flags) filled in."""
    df = _labelled(fitted)
    df["price_spike_5d"], df["volume_zscore_63d"], df["revenue_growth_yoy_pct"] = 0.0, 0.0, 50.0
    return classify_valuation_transition(flag_report_lag(flag_value_trap(flag_meme_stock(df))))


# ---------------------------------------------------------------------------
# point-in-time panel
# ---------------------------------------------------------------------------
def test_revenue_growth_and_trends():
    periods = pd.date_range("2016-03-31", periods=24, freq="QE")
    sales = 10 * (1.08 ** (np.arange(24) / 4))  # exactly +8% per year
    margin = np.tile([0.10, 0.20], 12)  # alternating -> std ~0.052
    trends = add_fundamental_trends(pd.DataFrame({"period": periods, "sales_per_share": sales, "operating_margin": margin}))
    growth = trends["revenue_growth_yoy"].dropna()
    assert len(growth) == 20 and np.allclose(growth, 0.08, atol=1e-9)
    cagr = trends["revenue_cagr_3y"].dropna()
    assert len(cagr) == 24 - 12 - 3, "needs 4 quarters for TTM plus 3 years of history"
    assert np.allclose(cagr, 0.08, atol=1e-9)
    vol = trends["op_margin_volatility"]
    assert vol.iloc[:7].isna().all() and np.isclose(vol.iloc[-1], np.std(margin[-12:], ddof=1))


def test_fundamental_breaks_hon_2026():
    """HON's Finnhub quarters as fetched on 2026-09-29 (first two quarters
    filled in): a one-off charge in Q4 2025, a one-off gain in Q2 2026 and
    sales/share doubling in Q1 2026. A seasonal retailer's big Q4 is not a
    break."""
    periods = pd.date_range("2023-09-30", periods=12, freq="QE")
    eps = [2.27, 2.60, 2.2281, 2.3601, 2.1602, 1.9624, 2.2234, 2.4497, 2.8569, 0.4619, 2.5721, 17.8343]
    sps = [14.0, 14.6, 13.8669, 14.6393, 14.8723, 14.0027, 15.0714, 16.1523, 16.2930, 15.2803, 28.6435, 30.5053]
    hon = add_fundamental_trends(pd.DataFrame({"period": periods, "eps": eps, "sales_per_share": sps,
                                               "operating_margin": 0.2})).set_index("period")
    q2_26, q1_26, q4_25, q3_25 = (pd.Timestamp(d) for d in ("2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"))
    assert hon.loc[q2_26, "eps_one_off_period"] == q2_26 and np.isclose(hon.loc[q2_26, "eps_one_off_ratio"], 17.8343 / 2.4497)
    assert hon.loc[q2_26, "per_share_break_period"] == q1_26 and np.isclose(hon.loc[q2_26, "per_share_break_ratio"], 28.6435 / 15.0714)
    assert hon.loc[q4_25, "eps_one_off_period"] == q4_25 and pd.isna(hon.loc[q4_25, "per_share_break_period"])
    assert pd.isna(hon.loc[q3_25, "eps_one_off_period"]) and pd.isna(hon.loc[q3_25, "per_share_break_period"])

    retail_periods = pd.date_range("2018-03-31", periods=24, freq="QE")
    growth = 1.02 ** np.arange(24)
    retail = add_fundamental_trends(pd.DataFrame({
        "period": retail_periods, "eps": np.tile([0.3, 0.4, 0.2, 2.0], 6) * growth,
        "sales_per_share": np.tile([8.0, 8.5, 8.0, 14.0], 6) * growth, "operating_margin": 0.05}))
    assert retail["eps_one_off_period"].isna().all() and retail["per_share_break_period"].isna().all()


def test_normalized_pe_uses_3y_average_eps():
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=2)
    as_of = pd.Timestamp("2022-07-01")
    panel = build_raw_panel(tickers, prices, fundamentals, universe, [as_of]).set_index("ticker")
    fund = fundamentals[tickers[0]]
    known = fund[fund["period"] + pd.Timedelta(days=45) <= as_of].tail(12)
    row = panel.loc[tickers[0]]
    assert np.isclose(row["normalized_pe"], row["price"] / (4 * known["eps"].mean()))
    assert np.isclose(row["eps_ttm"], known["eps"].tail(4).sum())


def test_as_of_dates_include_today():
    today = pd.Timestamp("2026-09-23")
    dates = build_as_of_dates(pd.Timestamp("2026-01-01"), today)
    assert dates[-1] == today and dates[-2] == pd.Timestamp("2026-07-01")


def test_no_look_ahead():
    """Rows at or before a cutoff must not change when everything AFTER the
    cutoff (future prices, dividends, not-yet-available fundamentals) is
    scrambled. Forward returns are the only columns allowed to differ."""
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=8)
    as_of = list(pd.date_range("2018-01-01", "2024-04-01", freq="QS"))
    cutoff = pd.Timestamp("2021-07-01")
    base = build_raw_panel(tickers, prices, fundamentals, universe, as_of)

    rng = np.random.default_rng(99)
    prices2 = {}
    for t, df in prices.items():
        df = df.copy()
        future = df["date"] > cutoff
        for col in ("close", "close_raw", "volume"):
            df.loc[future, col] *= rng.uniform(0.2, 5.0, future.sum())
        df.loc[future, "dividend"] = rng.uniform(0, 2.0, future.sum())
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


def test_missing_close_days_are_ignored():
    """yfinance sometimes returns empty closes (always for today's row before
    the close); the snapshot price must come from the last real trade and no
    pandas fill warning may be raised."""
    import warnings

    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=1)
    df = prices["SYN000"]
    last_real = df.loc[df["date"] <= "2021-06-29", "close_raw"].iloc[-1]
    df.loc[df["date"].isin(pd.to_datetime(["2021-07-01", "2021-06-30", "2021-05-12"])), ["close", "close_raw"]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        panel = build_raw_panel(tickers, prices, fundamentals, universe, [pd.Timestamp("2021-07-01")])
    row = panel.iloc[0]
    assert row["price"] == last_real and pd.notna(row["volatility_63d"])


def test_non_positive_prices_are_dropped():
    """A bad 0 close in yfinance's history must not reach volatility or the
    forward-return labels as +-inf."""
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=1)
    df = prices[tickers[0]].copy()
    bad = df["date"].isin(df["date"].iloc[[1500, 1510]])
    df.loc[bad, ["close", "close_raw"]] = 0.0
    panel = build_raw_panel(tickers, {tickers[0]: df}, fundamentals, universe, list(pd.date_range("2019-01-01", "2023-01-01", freq="QS")))
    numeric = panel.select_dtypes("number")
    assert np.isfinite(numeric.to_numpy()[~np.isnan(numeric.to_numpy())]).all()


def test_multiples_rescaled_to_as_of_price():
    """Price multiples scale with the QUOTED price; EV/EBITDA only through its
    market-cap part. The dividend-adjusted close moves differently (100->150
    quoted vs. 90->140 adjusted) and must not be used."""
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    raw = np.where(dates <= "2020-03-31", 100.0, 150.0)
    adjusted = np.where(dates <= "2020-03-31", 90.0, 140.0)
    prices = {"AAA": pd.DataFrame({"date": dates, "close": adjusted, "close_raw": raw, "volume": 1e6, "dividend": 0.0})}
    fundamentals = {"AAA": pd.DataFrame({
        "period": [pd.Timestamp("2020-03-31")], "trailing_pe": [20.0], "price_to_book": [4.0],
        "price_to_sales": [2.0], "price_to_fcf": [25.0], "ev_to_ebitda": [10.0],
        "enterprise_value": [500.0], "book_value": [100.0],  # market cap = 4 * 100 = 400, net debt = 100
        "return_on_equity": [0.2], "debt_to_equity": [1.0], "operating_margin": [0.1],
        "eps": [5.0], "payout_ratio_ttm": [0.3], "sales_per_share": [12.5],
    })}
    panel = build_raw_panel(["AAA"], prices, fundamentals, {"AAA": {"sector": "X"}}, [pd.Timestamp("2020-07-01")])
    row = panel.iloc[0]
    assert row["price"] == 150.0
    assert row["trailing_pe_reported"] == 20.0 and np.isclose(row["trailing_pe"], 30.0)
    assert np.isclose(row["price_to_book"], 6.0)
    assert row["price_to_sales_reported"] == 2.0 and np.isclose(row["price_to_sales"], 3.0)
    assert np.isclose(row["price_to_fcf"], 37.5)
    # EV: 400 * 1.5 + 100 = 700 -> x 700/500 = 1.4
    assert row["ev_to_ebitda_reported"] == 10.0 and np.isclose(row["ev_to_ebitda"], 14.0)


def test_dividend_features():
    """Per-payment logic: drifting ex-dates (5 payments inside one 365-day
    window) must not look like a raise, a one-off special must not move the
    yield, and a cut resets the no-cut clock."""
    dates = list(pd.date_range("2010-02-15", periods=36, freq="91D"))
    amounts = [0.50] * 20 + [0.30] * 16  # cut after 20 payments (~5 years)
    dates.insert(30, dates[29] + pd.Timedelta(days=20))  # special one-off 20 days after a payment
    amounts.insert(30, 2.00)
    dates[-1] = dates[-2] + pd.Timedelta(days=60)  # a drifted ex-date
    payments = pd.DataFrame({"date": pd.to_datetime(dates), "dividend": amounts})

    as_of = dates[-1] + pd.Timedelta(days=5)
    feats = _dividend_features(payments, as_of, price=40.0)
    assert np.isclose(feats["dividend_yield"], 0.30 * 4 / 40.0), feats
    cut_date = dates[20]
    assert np.isclose(feats["dividend_years_no_cut"], (as_of - cut_date).days / 365.25)
    assert pd.notna(feats["dividend_growth_3y"]) and abs(feats["dividend_growth_3y"]) < 1e-9

    before_cut = _dividend_features(payments, dates[19] + pd.Timedelta(days=5), price=40.0)
    assert before_cut["dividend_growth_3y"] == 0 and before_cut["dividend_years_no_cut"] > 4.5

    stopped = _dividend_features(payments, as_of + pd.Timedelta(days=500), price=40.0)
    assert stopped == {"dividend_yield": 0.0, "dividend_growth_3y": stopped["dividend_growth_3y"], "dividend_years_no_cut": 0.0}
    assert pd.isna(stopped["dividend_growth_3y"])


def test_dividend_yield_uses_quoted_price():
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=2)
    panel = build_raw_panel(tickers, prices, fundamentals, universe, [pd.Timestamp("2019-07-01")]).set_index("ticker")
    payer, non_payer = panel.loc["SYN000"], panel.loc["SYN001"]
    assert payer["dividend_yield"] > 0 and payer["dividend_years_no_cut"] > 0
    assert non_payer["dividend_yield"] == 0 and non_payer["dividend_years_no_cut"] == 0
    quoted = prices["SYN000"].set_index("date").loc[:"2019-07-01", "close_raw"].iloc[-1]
    assert payer["price"] == quoted


# ---------------------------------------------------------------------------
# fair-value model
# ---------------------------------------------------------------------------
def test_fair_value_recovers_planted_mispricing():
    panel, diag = add_fair_value(make_fair_value_panel())
    summary = summarize_diagnostics(diag)
    for key in ("pe", "pb", "ps", "ev_ebitda", "pfcf"):
        r2_model = summary.loc[key, "r2_model"].mean()
        r2_base = summary.loc[key, "r2_sector_median"].mean()
        assert r2_model > r2_base + 0.1, f"{key}: model R2 {r2_model:.2f} should clearly beat sector median {r2_base:.2f}"

    labelled = _labelled(panel)
    cheap = labelled[labelled["ticker"].isin([f"T{i:03d}" for i in range(15)])]
    rich = labelled[labelled["ticker"].isin([f"T{i:03d}" for i in range(15, 30)])]
    assert (cheap["valuation_label"] == "저평가").mean() > 0.8
    assert (rich["valuation_label"] == "고평가").mean() > 0.8
    assert np.isclose(np.expm1(cheap["valuation_gap"]).median(), np.expm1(-0.6), atol=0.1)
    # every multiple was mispriced the same way, so every view should agree;
    # Financials are only judged on PER/PBR (config exclude_sectors)
    financial = cheap["sector"] == "Financials"
    assert (cheap.loc[~financial, "n_gaps"] == 5).all() and (cheap.loc[financial, "n_gaps"] == 2).all()
    assert cheap["gap_agreement"].mean() > 0.95


def test_financials_skip_sales_ebitda_fcf_multiples():
    out, _ = add_fair_value(make_fair_value_panel(n_dates=1))
    fin = out[out["sector"] == "Financials"]
    assert fin[["ps_gap", "ev_ebitda_gap", "pfcf_gap"]].isna().all().all()
    assert fin["pe_gap"].notna().all() and (fin["valuation_basis"] == "PER+PBR").all()
    report = report_at(screen_ready(out))
    text = report.loc[report["sector"] == "Financials", "explanation"].iloc[0]
    assert "금융업" in text and "PSR 값 없음" not in text


def test_fair_value_is_out_of_fold():
    """A stock's own multiple must not move its own fair multiple."""
    panel = make_fair_value_panel(n_dates=1)
    before, _ = add_fair_value(panel)
    changed = panel.copy()
    changed.loc[changed["ticker"] == "T100", "trailing_pe"] *= 1.3  # stays inside the PER bounds
    after, _ = add_fair_value(changed)
    b = before.set_index("ticker").loc["T100"]
    a = after.set_index("ticker").loc["T100"]
    assert pd.notna(b["fair_pe"]) and np.isclose(a["fair_pe"], b["fair_pe"], rtol=1e-12)
    assert np.isclose(a["pe_gap"], b["pe_gap"] + np.log(1.3))


def test_contributions_add_up():
    panel, _ = add_fair_value(make_fair_value_panel(n_dates=1))
    contrib_cols = [f"pe_contrib_{d}" for d in [*FAIR_VALUE_FEATURES, "sector"]]
    implied_intercept = np.log(panel["fair_pe"]) - panel[contrib_cols].sum(axis=1)
    # the per-fold intercept (training-fold mean of log PE) is nearly constant
    assert implied_intercept.std() < 0.05


def test_out_of_range_multiples_get_no_gap():
    # all non-Financials tickers (i % 4 != 2), so every multiple applies
    panel = make_fair_value_panel(n_dates=1)
    panel.loc[panel["ticker"] == "T049", "trailing_pe"] = 400.0   # above max
    panel.loc[panel["ticker"] == "T051", "price_to_book"] = 0.01  # below min (data error)
    panel.loc[panel["ticker"] == "T053", ["trailing_pe", "eps"]] = [np.nan, -1.0]  # loss, other multiples usable
    panel.loc[panel["ticker"] == "T055", ["trailing_pe", "eps", "price_to_sales"]] = [np.nan, -1.0, np.nan]
    panel.loc[panel["ticker"] == "T057", "price_to_sales"] = 90.0  # above PSR max
    out, _ = add_fair_value(panel)
    out = _labelled(out).set_index("ticker")
    assert pd.isna(out.loc["T049", "pe_gap"]) and pd.notna(out.loc["T049", "pb_gap"])
    assert out.loc["T049", "valuation_basis"] == "PBR+PSR+EV/EBITDA+P/FCF" and out.loc["T049", "n_gaps"] == 4
    assert pd.isna(out.loc["T051", "pb_gap"])
    assert pd.isna(out.loc["T057", "ps_gap"]) and "PSR" not in out.loc["T057", "valuation_basis"]
    # loss-makers get no verdict, even when other multiples' gaps exist (kept for reference)
    for t in ("T053", "T055"):
        row = out.loc[t]
        assert row["loss_flag"] and row["valuation_label"] == "판단 보류(적자)"
        assert pd.isna(row["valuation_gap"]) and pd.isna(row["cheapness_rank"])
    assert pd.notna(out.loc["T053", "ps_gap"]) and pd.notna(out.loc["T053", "pb_gap"])


def test_per_multiple_features_and_reference_view():
    """extra_features reach only their own multiple; pe_norm (in_verdict
    False) gets a gap but stays out of the combined verdict."""
    panel = make_fair_value_panel(n_dates=1)
    rng = np.random.default_rng(3)
    panel["fcf_margin"] = rng.normal(0.1, 0.03, len(panel))
    panel["normalized_pe"] = panel["trailing_pe"] * rng.uniform(0.8, 1.2, len(panel))
    out, _ = add_fair_value(panel)
    non_fin = out["sector"] != "Financials"  # P/FCF skips Financials
    assert out.loc[non_fin, "pfcf_contrib_fcf_margin"].notna().all() and "pe_contrib_fcf_margin" not in out.columns
    assert out["pe_norm_gap"].notna().mean() > 0.9
    assert not out["valuation_basis"].str.contains("정규화").any() and out["n_gaps"].max() == 5
    text = report_at(screen_ready(out))["explanation"].iloc[0]
    assert "정규화 PER" in text and "참고용, 종합 판단 제외" in text


def test_rank_transform_caps_an_extreme_value():
    """With rank features one absurd ROE moves a fair multiple no further
    than the highest ordinary ROE would, and the model still fits."""
    panel = make_fair_value_panel(n_dates=1)
    panel.loc[panel["ticker"] == "T100", "return_on_equity"] = 50.0  # buyback-shrunk equity
    ranked, diag = add_fair_value(panel, transform="rank")
    contrib = ranked.set_index("ticker")["pb_contrib_return_on_equity"]
    # coefficients differ slightly per out-of-fold fold, hence the tolerance
    assert contrib["T100"] <= contrib.drop("T100").max() + 0.05
    assert summarize_diagnostics(diag).loc["pb", "r2_model"].mean() > 0.5
    table = compare_feature_sets(panel, {"current": FAIR_VALUE_FEATURES, "rank": (FAIR_VALUE_FEATURES, "rank")})
    assert {"current", "rank"} <= set(table.columns)


def test_compare_feature_sets():
    """A feature that drives the planted multiples must beat a set without
    it, on the same folds; a {"drop": ...} set is the same as leaving the
    feature out of the list."""
    panel = make_fair_value_panel(n_dates=2)
    table = compare_feature_sets(panel, {
        "without_growth": [f for f in FAIR_VALUE_FEATURES if f != "revenue_growth_yoy"],
        "-revenue_growth_yoy": {"drop": ("revenue_growth_yoy",)},
        "current": FAIR_VALUE_FEATURES,
    })
    assert {"sector_median", "without_growth", "-revenue_growth_yoy", "current"} <= set(table.columns)
    assert (table.loc["pe", "current"] > table.loc["pe", "without_growth"] + 0.05).all()
    assert np.allclose(table["without_growth"], table["-revenue_growth_yoy"])


def test_parallel_fits_match_sequential():
    """Dates and feature sets fitted in worker processes give exactly the
    single-process result."""
    panel = make_fair_value_panel(n_tickers=100, n_dates=8)  # >= fair_value._MIN_DATES_FOR_PARALLEL
    seq, seq_diag = add_fair_value(panel, n_jobs=1)
    par, par_diag = add_fair_value(panel, n_jobs=2)
    pd.testing.assert_frame_equal(seq, par)
    pd.testing.assert_frame_equal(seq_diag, par_diag)
    sets = {"current": FAIR_VALUE_FEATURES, "-payout": {"drop": ("payout_ratio_ttm",)}}
    pd.testing.assert_frame_equal(compare_feature_sets(panel, sets, n_jobs=1), compare_feature_sets(panel, sets, n_jobs=2))


def test_parallel_panel_matches_sequential():
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=50)  # >= features._MIN_TICKERS_FOR_PARALLEL
    as_of = list(pd.date_range("2019-01-01", "2020-01-01", freq="QS"))
    seq = build_raw_panel(tickers, prices, fundamentals, universe, as_of, n_jobs=1)
    par = build_raw_panel(tickers, prices, fundamentals, universe, as_of, n_jobs=2)
    pd.testing.assert_frame_equal(seq, par)


# ---------------------------------------------------------------------------
# data collection cache
# ---------------------------------------------------------------------------
def test_cache_refetches_when_a_field_was_added():
    """Files cached before a field existed (a Finnhub series, or the dividend
    / close_raw price columns) are fetched again once; current ones are read
    from cache."""
    import tempfile
    import data

    cache = Path(tempfile.mkdtemp())
    for kind in ("fundamentals", "prices"):
        (cache / kind).mkdir()
    period = [pd.Timestamp("2020-03-31")]
    pd.DataFrame({"period": period, "trailing_pe": [20.0]}).to_parquet(cache / "fundamentals" / "OLD.parquet")
    current_fund = pd.DataFrame({"period": period, **{k: [1.0] for k in data.FINNHUB_FIELD_MAP}})
    current_fund.to_parquet(cache / "fundamentals" / "NEW.parquet")
    day = [pd.Timestamp.today().normalize()]  # recent, so only the missing columns force a re-fetch
    pd.DataFrame({"date": day, "close": [1.0], "volume": [1.0]}).to_parquet(cache / "prices" / "OLD.parquet")
    current_price = pd.DataFrame({"date": day, "close": [1.0], "close_raw": [1.0], "volume": [1.0], "dividend": [0.0]})
    current_price.to_parquet(cache / "prices" / "NEW.parquet")

    fund_calls, price_calls = [], []
    orig_fund, orig_price = data.fetch_fundamentals, data.fetch_price_history
    data.fetch_fundamentals = lambda t, api_key=None: fund_calls.append(t) or current_fund
    data.fetch_price_history = lambda t: price_calls.append(t) or current_price
    try:
        prices, fundamentals = data.collect(["OLD", "NEW"], api_key="x", cache_dir=cache)
    finally:
        data.fetch_fundamentals, data.fetch_price_history = orig_fund, orig_price
    assert fund_calls == ["OLD"] and price_calls == ["OLD"]
    assert "ev_to_ebitda" in fundamentals["OLD"].columns and "dividend" in prices["OLD"].columns
    assert "ev_to_ebitda" in pd.read_parquet(cache / "fundamentals" / "OLD.parquet").columns


def test_stale_price_cache_is_refreshed_and_delisted_history_kept():
    import tempfile
    import data

    cache = Path(tempfile.mkdtemp())
    (cache / "prices").mkdir()
    (cache / "fundamentals").mkdir()
    stale_day = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)  # stale, but not a truncation
    stale = pd.DataFrame({"date": [stale_day], "close": [1.0], "close_raw": [1.0], "volume": [1.0], "dividend": [0.0]})
    for t in ("LIVE", "GONE"):
        stale.to_parquet(cache / "prices" / f"{t}.parquet")
        pd.DataFrame({"period": [pd.Timestamp("2025-12-31")], **{k: [1.0] for k in data.FINNHUB_FIELD_MAP}}) \
            .to_parquet(cache / "fundamentals" / f"{t}.parquet")
    fresh = stale.assign(date=pd.Timestamp.today().normalize(), close=2.0)
    calls = []
    orig = data.fetch_price_history
    data.fetch_price_history = lambda t: calls.append(t) or (fresh if t == "LIVE" else pd.DataFrame(columns=data.PRICE_COLUMNS))
    try:
        prices, _ = data.collect(["LIVE", "GONE"], api_key="x", cache_dir=cache)
    finally:
        data.fetch_price_history = orig
    assert calls == ["LIVE", "GONE"]
    assert prices["LIVE"]["close"].iloc[-1] == 2.0 and pd.read_parquet(cache / "prices" / "LIVE.parquet")["close"].iloc[-1] == 2.0
    assert len(prices["GONE"]) == 1, "a delisted ticker keeps its cached history"


def test_truncated_refetch_keeps_cached_history():
    """A re-fetch that returns a delisting stub (EA: 1 row, 2026-09-29) must
    not replace years of cached prices; the quality report names short
    histories."""
    import tempfile
    import data

    cache = Path(tempfile.mkdtemp())
    (cache / "prices").mkdir()
    (cache / "fundamentals").mkdir()
    days = pd.date_range("2015-01-02", "2026-08-03", freq="B")
    full = pd.DataFrame({"date": days, "close": 1.0, "close_raw": 1.0, "volume": 1.0, "dividend": 0.0})
    full.to_parquet(cache / "prices" / "EA.parquet")
    pd.DataFrame({"period": [pd.Timestamp("2025-12-31")], **{k: [1.0] for k in data.FINNHUB_FIELD_MAP}}) \
        .to_parquet(cache / "fundamentals" / "EA.parquet")
    stub = full.tail(1).assign(close=2.0)
    orig = data.fetch_price_history
    data.fetch_price_history = lambda t: stub
    try:
        prices, fundamentals = data.collect(["EA"], api_key="x", cache_dir=cache)
    finally:
        data.fetch_price_history = orig
    assert len(prices["EA"]) == len(full) and len(pd.read_parquet(cache / "prices" / "EA.parquet")) == len(full)
    issues = data.data_quality_report({"EA": stub, "OK": full}, {"EA": fundamentals["EA"], "OK": fundamentals["EA"]})
    assert issues == ["EA"]


def test_parse_constituents_and_load_universe():
    import tempfile
    import universe

    html = """<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>GICS Sub-Industry</th></tr>
    <tr><td>MOG.A</td><td>Moog</td><td>Industrials</td><td>Aerospace</td></tr>
    <tr><td>ABC</td><td>Abc Corp</td><td>Information Technology</td><td>Software</td></tr>
    <tr><td>AMC</td><td>AMC Entertainment</td><td>Communication Services</td><td>Movies</td></tr>
    <tr><td>PMT</td><td>PennyMac Mortgage</td><td>Real Estate</td><td>Mortgage REITs</td></tr>
    <tr><td>ABC</td><td>dup</td><td>Health Care</td><td>x</td></tr></table>"""
    members = universe.parse_constituents(html)
    assert members["ticker"].tolist() == ["MOG.A", "ABC", "AMC", "PMT"]
    assert members.set_index("ticker").loc["ABC", "sector"] == "Technology"

    cache = Path(tempfile.mkdtemp())
    (cache / "universe").mkdir()
    members.to_csv(cache / "universe" / "sp600.csv", index=False)
    loaded = universe.load_universe(cache, indexes=("sp600",))
    assert loaded["MOG.A"] == {"name": "Moog", "sector": "Industrials", "size": "small"}
    assert loaded["AMC"]["sector"] == universe.UNIVERSE["AMC"]["sector"] and loaded["AMC"]["size"] == "small"
    assert loaded["AAPL"]["size"] == "large" and len(loaded) == len(universe.UNIVERSE) + 3
    assert loaded["PMT"]["sector"] == "Financials", "mortgage REITs are judged like other lenders"


def test_size_features_are_price_free():
    """log_revenue = TTM sales/share x share count; moving every price after
    the period end must not change it (it's a size, not a valuation)."""
    tickers, universe, prices, fundamentals = make_raw_data(n_tickers=2)
    universe = {t: {**m, "size": "small"} for t, m in universe.items()}
    as_of = [pd.Timestamp("2022-07-01")]
    base = build_raw_panel(tickers, prices, fundamentals, universe, as_of).set_index("ticker")
    moved = {t: df.assign(close_raw=np.where(df["date"] > pd.Timestamp("2022-03-31"), df["close_raw"] * 3, df["close_raw"]))
             for t, df in prices.items()}
    after = build_raw_panel(tickers, moved, fundamentals, universe, as_of).set_index("ticker")
    t = tickers[0]
    assert base.loc[t, "size_group"] == "small"
    assert np.isclose(base.loc[t, "log_revenue"], after.loc[t, "log_revenue"])
    assert np.isclose(base.loc[t, "log_book_value"], np.log(300.0))
    assert not np.isclose(base.loc[t, "trailing_pe"], after.loc[t, "trailing_pe"])


def test_r2_and_return_test_by_size_group():
    panel = make_fair_value_panel(n_tickers=200, n_dates=5)  # a split needs >= 4 dates to be tested
    panel["size_group"] = np.where(panel["ticker"].str[1:].astype(int) % 2 == 0, "large", "small")
    out, _ = add_fair_value(panel)
    r2 = r2_by_group(out)
    assert set(r2.index.get_level_values("group")) == {"large", "small"} and (r2["r2"] > 0.2).all()
    rng = np.random.default_rng(0)
    out["fwd_return_1m"] = rng.normal(0, 0.05, len(out))
    tests = gap_return_test(out, horizons=[1], by="size_group")
    assert set(tests["group"]) == {"all", "large", "small"} and tests["p_value_fdr"].notna().all()


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------
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


def test_single_view_flag():
    """A verdict that rests on one multiple is flagged and says so."""
    panel = make_fair_value_panel(n_dates=1)
    only_pb = panel["ticker"] == "T001"  # non-Financials, planted cheap
    panel.loc[only_pb, ["trailing_pe", "price_to_sales", "ev_to_ebitda", "price_to_fcf"]] = np.nan
    report = report_at(flag_single_view(screen_ready(add_fair_value(panel)[0]))).set_index("ticker")
    row = report.loc["T001"]
    assert row["n_gaps"] == 1 and row["valuation_label"] == "저평가"
    assert row["single_view_flag"] and "PBR 한 가지 배수" in row["single_view_reason"]
    assert "한 가지 관점으로만 판단" in row["explanation"]
    assert not report.drop("T001")["single_view_flag"].any()


def test_report_lag_flag():
    """Price move since the fundamentals' quarter, read back from rescaled /
    reported multiples: a spin-off-sized drop is flagged, a normal move not."""
    panel = pd.DataFrame({
        "ticker": ["SPIN", "CALM", "OLD"],
        "fundamentals_period": pd.to_datetime(["2026-06-30"] * 3),
        "price_to_book": [1.0, 2.2, 3.0],
        "price_to_book_reported": [2.5, 2.0, np.nan],   # OLD: no reported value to compare
        "trailing_pe": [8.0, 22.0, 15.0],
        "trailing_pe_reported": [20.0, 20.0, 15.0],
    })
    out = flag_report_lag(panel).set_index("ticker")
    assert out.loc["SPIN", "report_lag_flag"] and np.isclose(out.loc["SPIN", "price_move_since_report"], -0.6)
    assert "2026-06-30" in out.loc["SPIN", "report_lag_reason"] and "-60%" in out.loc["SPIN", "report_lag_reason"]
    assert not out.loc["CALM", "report_lag_flag"] and out.loc["CALM", "report_lag_reason"] == ""
    assert not out.loc["OLD", "report_lag_flag"] and np.isclose(out.loc["OLD", "price_move_since_report"], 0.0)


def test_end_to_end():
    panel = build_synthetic_panel(n_tickers=80)  # >= FAIR_VALUE_MIN_ROWS per date
    screened = screen(panel)
    report = report_at(screened)
    assert len(report) == panel["ticker"].nunique()
    assert report["valuation_label"].isin(["저평가", "중립", "고평가", "데이터 부족", "판단 보류(적자)"]).all()
    assert report["explanation"].str.len().gt(0).all()
    assert report["explanation"].str.contains("종합:").any()
    labelled = report["valuation_label"].isin(["저평가", "고평가"])
    assert 0.25 < labelled.mean() < 0.55, "top/bottom 20% should be labelled"
    assert report["n_gaps"].max() == 5
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
