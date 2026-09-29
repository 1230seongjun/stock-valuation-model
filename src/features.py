"""
Point-in-time panel: one row per (ticker, as_of) with only what was knowable
on that as_of date.

This is deliberately the single place where "what counts as known at time T"
is decided, so look-ahead bugs have one place to hide:
  - fundamentals: the latest quarter whose period + REPORTING_LAG_DAYS <= as_of
    (trend features are computed from that quarter and earlier ones only)
  - valuation multiples: Finnhub reports them at the period-end price;
    rescaled to the as_of price (config.RESCALE_MULTIPLES_TO_AS_OF_PRICE)
    using two prices that are both <= as_of
  - prices / technicals / dividends: daily rows with date <= as_of only
  - sector percentiles: ranked among rows of the SAME as_of (and sector)
  - fwd_return_*: the only columns that look past as_of — they are labels
    for fair_value.gap_return_test and must never be used as features

Two price series (data.fetch_price_history): `close` is dividend-adjusted
(for returns and volatility), `close_raw` is the price actually quoted (for
the `price` column, dividend yield and rescaling Finnhub's multiples).

Derived indicators (see data.py for the raw fields):
  - revenue_growth_yoy: YoY % change in the quarter's sales per share
  - revenue_cagr_3y: 3-year CAGR of trailing-12-month sales per share
  - op_margin_volatility: std of quarterly operating margin, last 12 quarters
    (a one-off gain/charge or a cyclical business shows up here)
  - op_margin_avg_3y / roe_avg_3y: 12-quarter means (candidate features)
  - normalized_pe: as_of price / (4 x mean quarterly EPS of the last 12
    quarters) — PER on 3-year average earnings, so one quarter's one-off
    gain or charge weighs 1/12 instead of 1/4 (a reference view, see
    config.FAIR_VALUE_TARGETS "in_verdict")
  - eps_one_off_* / per_share_break_*: quarters inside the TTM window that
    make trailing figures unreliable (_add_fundamental_breaks)
  - log_revenue / log_book_value: company size without the snapshot price
    (_size_features); size_group: today's index membership (universe.py),
    for splitting reports only
  - dividend_yield / dividend_growth_3y / dividend_years_no_cut: from actual
    dividend payments (see _dividend_features)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ALL_INDICATORS,
    FUNDAMENTAL_INDICATORS,
    HORIZONS_MONTHS,
    REBALANCE_FREQ,
    RESCALE_MULTIPLES_TO_AS_OF_PRICE,
    TECHNICAL_INDICATORS,
)

REPORTING_LAG_DAYS = 45  # typical 10-Q lag; Finnhub has no filing dates to check against
# A snapshot needs a trade within this many days before as_of. Covers
# weekends/holidays (e.g. as_of = Jan 1); anything longer means the stock
# wasn't trading yet (pre-IPO) or anymore (delisted/acquired), and carrying
# a months-old price forward would fake a valuation.
MAX_PRICE_STALENESS_DAYS = 10

# Columns taken from the latest known fundamentals row. dividend_yield is not
# here: it comes from actual dividend payments (_dividend_features).
_AS_OF_FUNDAMENTAL_KEYS = [k for k in FUNDAMENTAL_INDICATORS if k != "dividend_yield"] + [
    # statement ratios (candidate features, see config.FAIR_VALUE_FEATURE_CANDIDATES)
    "roic", "roa", "gross_margin", "net_margin", "fcf_margin", "net_debt_to_capital",
    "current_ratio", "asset_turnover", "sga_to_sales",
    # trend features (add_fundamental_trends)
    "revenue_cagr_3y", "op_margin_volatility", "op_margin_avg_3y", "roe_avg_3y",
    # TTM-reliability checks (_add_fundamental_breaks)
    "eps_one_off_period", "eps_one_off_ratio", "per_share_break_period", "per_share_break_ratio",
    # raw components
    "eps", "eps_ttm", "eps_avg_3y", "payout_ratio_ttm", "sales_per_share", "sps_ttm", "enterprise_value", "book_value",
]
_ANOMALY_SIGNAL_KEYS = ["price_spike_5d", "volume_zscore_63d"]
_DIVIDEND_KEYS = ["dividend_yield", "dividend_growth_3y", "dividend_years_no_cut"]

# Multiples that scale 1:1 with the share price. ev_to_ebitda is handled
# separately: only the market-cap part of EV moves with the price.
PRICE_MULTIPLES = ("trailing_pe", "price_to_book", "price_to_sales", "price_to_fcf")
RESCALED_MULTIPLES = (*PRICE_MULTIPLES, "ev_to_ebitda")


def build_as_of_dates(start: pd.Timestamp, today: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """Quarterly snapshots from `start`, plus `today` itself (normalized) so
    the screening report has a current snapshot instead of one that can be up
    to three months stale."""
    today = (today or pd.Timestamp.today()).normalize()
    dates = list(pd.date_range(start, today, freq=REBALANCE_FREQ))
    if not dates or dates[-1] != today:
        dates.append(today)
    return dates


def _match_prior(df: pd.DataFrame, value_col: str, days: int, out_col: str) -> pd.DataFrame:
    """Adds `out_col` = value_col of the row whose period is ~`days` earlier
    (nearest within +-45 days; NaN if none). Only earlier rows can match."""
    df = df.copy()
    df["_target"] = df["period"] - pd.Timedelta(days=days)
    lookup = df[["period", value_col]].rename(columns={"period": "_matched", value_col: out_col})
    merged = pd.merge_asof(
        df.sort_values("_target"),
        lookup.dropna().sort_values("_matched"),
        left_on="_target",
        right_on="_matched",
        direction="nearest",
        tolerance=pd.Timedelta(days=45),
    )
    return merged.drop(columns=["_target", "_matched"]).sort_values("period").reset_index(drop=True)


def add_fundamental_trends(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker features that need several quarters of history. Each row
    only uses its own and EARLIER periods, and the reporting lag is applied
    afterwards to the row as a whole, so this adds no look-ahead:
      revenue_growth_yoy   — quarter's sales/share vs. same quarter a year ago
      revenue_cagr_3y      — trailing-12-month sales/share vs. 3 years ago,
                             annualized (smooths a single odd quarter)
      op_margin_volatility — std of the last 12 quarterly operating margins
                             (min 8)"""
    df = fundamentals.sort_values("period").reset_index(drop=True).copy()
    for col in ("sales_per_share", "operating_margin"):
        if col not in df.columns:
            df[col] = np.nan

    df = _match_prior(df, "sales_per_share", 365, "_sps_1y")
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (df["sales_per_share"] - df["_sps_1y"]) / df["_sps_1y"].abs()
    df["revenue_growth_yoy"] = growth.replace([np.inf, -np.inf], np.nan)

    # TTM = last 4 quarters, only if they really are 4 consecutive quarters
    ttm = df["sales_per_share"].rolling(4, min_periods=4).sum()
    consecutive = (df["period"] - df["period"].shift(3)) <= pd.Timedelta(days=300)
    df["_sps_ttm"] = ttm.where(consecutive)
    df["sps_ttm"] = df["_sps_ttm"]
    df = _match_prior(df, "_sps_ttm", 3 * 365, "_sps_ttm_3y")
    ratio = df["_sps_ttm"] / df["_sps_ttm_3y"]
    df["revenue_cagr_3y"] = np.where((df["_sps_ttm"] > 0) & (df["_sps_ttm_3y"] > 0), ratio ** (1 / 3) - 1, np.nan)

    df["op_margin_volatility"] = df["operating_margin"].rolling(12, min_periods=8).std()

    # 3-year averages: a one-off quarter weighs 1/12 instead of 1/4
    if "return_on_equity" not in df.columns:
        df["return_on_equity"] = np.nan
    df["op_margin_avg_3y"] = df["operating_margin"].rolling(12, min_periods=8).mean()
    df["roe_avg_3y"] = df["return_on_equity"].rolling(12, min_periods=8).mean()
    eps = df["eps"] if "eps" in df.columns else pd.Series(np.nan, index=df.index)
    df["eps_ttm"] = eps.rolling(4, min_periods=4).sum().where(consecutive)
    consecutive_3y = (df["period"] - df["period"].shift(11)) <= pd.Timedelta(days=3 * 365)
    df["eps_avg_3y"] = eps.rolling(12, min_periods=12).mean().where(consecutive_3y)

    df = _add_fundamental_breaks(df.assign(eps=eps))
    return df.drop(columns=["_sps_1y", "_sps_ttm", "_sps_ttm_3y"])


# A quarter counts as a break when it is off by this factor against BOTH of
# its references (see _add_fundamental_breaks). Heuristics, not tuned on any
# split; evaluate prints how many rows they catch.
EPS_ONE_OFF_FACTOR = 3.0
PER_SHARE_BREAK_FACTOR = 1.5


def _add_fundamental_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """Quarters that make trailing-12-month figures unreliable, found on
    2026-09-29 for HON: Q2 2026 EPS 17.83 vs ~2.2-2.9 normally (a one-off
    gain, operating margin 77%) and sales/share doubling from Q1 2026 while
    the price didn't move (per-share basis change or data error). Together
    they made every multiple look cheap at once (-77%, 5/5 views agreeing).
      one-off EPS      — quarter EPS vs. the median of the 8 quarters before
                         it AND vs. the same quarter a year earlier, both
                         beyond x3 (or below x1/3, including a loss).
                         Needing both keeps seasonal businesses (a retailer's
                         Q4) from being flagged every year.
      per-share break  — quarter sales/share vs. the previous quarter AND
                         vs. a year earlier, both beyond x1.5 or x1/1.5
                         (spin-off, merger, share-count change, data error;
                         also a genuinely explosive quarter like NVDA 2023,
                         where trailing multiples mislead as well).
    For each row: the latest such quarter within its TTM window (itself and
    the 3 quarters before, if they span <= 300 days) as *_period / *_ratio
    (ratio vs. a year earlier), NaT/NaN if none. Uses only this row's and
    earlier periods, like the other trends."""
    df = _match_prior(df, "eps", 365, "_eps_1y")
    df = _match_prior(df, "sales_per_share", 365, "_sps_yoy")
    eps, sps = df["eps"], df["sales_per_share"]

    median_8q = eps.shift(1).rolling(8, min_periods=6).median()
    vs_median = eps / median_8q.where(median_8q > 0)
    vs_year = eps / df["_eps_1y"].where(df["_eps_1y"] > 0)
    off = lambda r: (r > EPS_ONE_OFF_FACTOR) | (r < 1 / EPS_ONE_OFF_FACTOR)
    eps_hit = (off(vs_median) & off(vs_year)).to_numpy()

    prev_q = sps.shift(1)
    vs_prev = sps / prev_q.where(prev_q > 0)
    sps_vs_year = sps / df["_sps_yoy"].where(df["_sps_yoy"] > 0)
    jump = lambda r: (r > PER_SHARE_BREAK_FACTOR) | (r < 1 / PER_SHARE_BREAK_FACTOR)
    sps_hit = (jump(vs_prev) & jump(sps_vs_year)).to_numpy()

    periods = df["period"].to_numpy()
    for name, hits, ratios in (("eps_one_off", eps_hit, vs_year.to_numpy()),
                               ("per_share_break", sps_hit, sps_vs_year.to_numpy())):
        when, ratio = [pd.NaT] * len(df), [np.nan] * len(df)
        for i in range(len(df)):
            for j in range(i, max(i - 4, -1), -1):  # this quarter back to 3 before
                if periods[i] - periods[j] > np.timedelta64(300, "D"):
                    break
                if hits[j]:
                    when[i], ratio[i] = periods[j], ratios[j]
                    break
        df[f"{name}_period"] = pd.to_datetime(when)
        df[f"{name}_ratio"] = ratio
    return df.drop(columns=["_eps_1y", "_sps_yoy"])


def apply_reporting_lag(fundamentals: pd.DataFrame, lag_days: int = REPORTING_LAG_DAYS) -> pd.DataFrame:
    df = fundamentals.copy()
    df["available_date"] = df["period"] + pd.Timedelta(days=lag_days)
    return df


def _as_of_fundamentals(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """Latest fundamentals row with available_date <= as_of, plus its fiscal
    `fundamentals_period` (needed to rescale the multiples)."""
    empty = {k: np.nan for k in _AS_OF_FUNDAMENTAL_KEYS} | {"fundamentals_period": pd.NaT}
    if fundamentals.empty:
        return empty
    valid = fundamentals[fundamentals["available_date"] <= as_of]
    if valid.empty:
        return empty
    row = valid.sort_values("available_date").iloc[-1]
    return {k: row.get(k, np.nan) for k in _AS_OF_FUNDAMENTAL_KEYS} | {"fundamentals_period": row["period"]}


def _rescale_multiples(row: dict, quoted: pd.DataFrame) -> float:
    """Move Finnhub's period-end multiples to the as_of price, in place
    (originals kept as *_reported). `quoted`: date + quoted price (close_raw)
    up to as_of. Both prices used are <= as_of, so no look-ahead.
      price multiples: x ratio, ratio = price(as_of) / price(period end)
      ev_to_ebitda:    EV = market cap + net debt, and only market cap moves
                       with the price: market cap = pb * book_value (both at
                       period end), net debt = EV - market cap, so
                       EV(as_of) / EV(period end) = 1 + mcap/EV * (ratio - 1).
                       Falls back to x ratio if EV/book value are missing.
    Returns the quoted price at the period end (NaN if unknown)."""
    for col in RESCALED_MULTIPLES:
        row[f"{col}_reported"] = row[col]
    if pd.isna(row["fundamentals_period"]) or pd.isna(row["price"]):
        return np.nan
    at_period = quoted[quoted["date"] <= row["fundamentals_period"]]
    if at_period.empty or at_period["price"].iloc[-1] <= 0:
        return np.nan
    period_price = at_period["price"].iloc[-1]
    if not RESCALE_MULTIPLES_TO_AS_OF_PRICE:
        return period_price
    ratio = row["price"] / period_price
    for col in PRICE_MULTIPLES:
        row[col] = row[col] * ratio

    ev, mcap = row["enterprise_value"], row["price_to_book_reported"] * row["book_value"]
    if pd.notna(ev) and pd.notna(mcap) and ev > 0 and mcap > 0:
        ev_factor = 1 + (mcap / ev) * (ratio - 1)
    else:
        ev_factor = ratio
    row["ev_to_ebitda"] = row["ev_to_ebitda"] * ev_factor if ev_factor > 0 else np.nan
    return period_price


def _size_features(row: dict, period_price: float) -> dict[str, float]:
    """Company size without the snapshot price (a market cap would carry
    the price, like momentum — config FAIR_VALUE_FEATURE_CANDIDATES):
      log_book_value — log total equity (Finnhub bookValue, millions); NaN
                       for negative equity
      log_revenue    — log TTM revenue = TTM sales/share x shares, shares =
                       market cap / price, both at the fiscal period end
                       (market cap = PBR x book value, as in the EV/EBITDA
                       rescaling), so the price cancels out: it is a share
                       count, not a valuation."""
    book, pb, sps_ttm = row["book_value"], row["price_to_book_reported"], row["sps_ttm"]
    out = {"log_book_value": np.log(book) if book > 0 else np.nan, "log_revenue": np.nan}
    if pb > 0 and book > 0 and sps_ttm > 0 and period_price > 0:
        out["log_revenue"] = np.log(sps_ttm * pb * book / period_price)
    return out


def _dividend_features(payments: pd.DataFrame, as_of: pd.Timestamp, price: float) -> dict[str, float]:
    """From actual cash dividends (ex-date <= as_of; `payments`: date +
    dividend, only rows with a payment). Built on per-payment amounts rather
    than calendar-year sums: quarterly ex-dates drift, so a 365-day window can
    hold 3 or 5 payments and fake a cut or a jump.
      dividend_yield        — median of the last `freq` payments x freq /
                              quoted price; freq = 365 / median gap between
                              payments over the last 2 years (a special
                              one-off or a drifted ex-date moves neither
                              median). 0 for a non-payer.
      dividend_growth_3y    — median of the last 4 payments vs. the 4 before
                              as_of - 3y, annualized; NaN for non-payers or a
                              dividend younger than 3 years
      dividend_years_no_cut — years since the first payment or the last cut
                              (a payment < 95% of the median of the 4 before
                              it; the next 3 payments still see pre-cut
                              amounts in that median, so they count as the
                              same cut), capped at 25; 0 for a non-payer
    A company that stopped paying more than 400 days ago is a non-payer."""
    none = {"dividend_yield": 0.0, "dividend_growth_3y": np.nan, "dividend_years_no_cut": 0.0}
    past = payments[payments["date"] <= as_of]
    if past.empty or (as_of - past["date"].iloc[-1]).days > 400 or pd.isna(price) or price <= 0:
        return none

    recent = past[past["date"] > as_of - pd.Timedelta(days=730)]
    gaps = recent["date"].diff().dt.days.dropna()
    freq = int(np.clip(round(365 / gaps.median()), 1, 12)) if len(gaps) else 1
    annual = past["dividend"].tail(freq).median() * freq

    then = past[past["date"] <= as_of - pd.Timedelta(days=3 * 365)]
    growth = np.nan
    if len(then) >= 4:
        now_amt, then_amt = past["dividend"].tail(4).median(), then["dividend"].tail(4).median()
        if now_amt > 0 and then_amt > 0:
            growth = (now_amt / then_amt) ** (1 / 3) - 1

    amounts = past["dividend"].to_numpy()
    dates = past["date"].to_numpy()
    since, last_cut = dates[0], -4
    for i in range(4, len(amounts)):
        if amounts[i] < 0.95 * np.median(amounts[i - 4:i]) and i - last_cut >= 4:
            since, last_cut = dates[i], i
    years = min(25.0, (as_of - pd.Timestamp(since)).days / 365.25)
    return {"dividend_yield": annual / price, "dividend_growth_3y": growth, "dividend_years_no_cut": years}


def _technicals_asof(close: pd.Series, volume: pd.Series) -> dict[str, float]:
    """close (dividend-adjusted) / volume: trailing <=260 daily rows up to
    and including as_of."""
    if len(close) < 60:
        return {k: np.nan for k in TECHNICAL_INDICATORS}

    ma50 = close.tail(50).mean()
    ma200 = close.tail(200).mean() if len(close) >= 200 else np.nan
    high_52w = close.tail(252).max()
    daily_ret = close.pct_change(fill_method=None).dropna().tail(63)
    vol_63d = volume.tail(63).mean()
    return {
        "ma50_vs_ma200": ma50 / ma200 - 1 if pd.notna(ma200) and ma200 else np.nan,
        "pct_from_52w_high": (high_52w - close.iloc[-1]) / high_52w if high_52w else np.nan,
        "volatility_63d": daily_ret.std() * np.sqrt(252) if len(daily_ret) > 5 else np.nan,
        "relative_volume": volume.tail(21).mean() / vol_63d if vol_63d else np.nan,
    }


def _anomaly_signals_asof(close: pd.Series, volume: pd.Series) -> dict[str, float]:
    """Inputs for screening.flag_meme_stock, not indicators:
    price_spike_5d = 5-trading-day log return; volume_zscore_63d = last day's
    volume vs. its trailing 63-day mean/std (a one-day spike, unlike
    relative_volume's sustained 21d-vs-63d ratio)."""
    if len(close) < 60:
        return {k: np.nan for k in _ANOMALY_SIGNAL_KEYS}
    spike = float(np.log(close.iloc[-1] / close.iloc[-6])) if close.iloc[-6] > 0 else np.nan
    vol_63d = volume.tail(63)
    std = vol_63d.std()
    zscore = float((volume.iloc[-1] - vol_63d.mean()) / std) if std else np.nan
    return {"price_spike_5d": spike, "volume_zscore_63d": zscore}


def _forward_log_return(prices: pd.DataFrame, as_of: pd.Timestamp, base_close: float, months: int) -> float:
    """Label only (see module docstring), on dividend-adjusted closes. NaN if
    the horizon runs past the last available price, so recent snapshots don't
    get a truncated return."""
    if pd.isna(base_close) or base_close <= 0:
        return np.nan
    target_date = as_of + pd.Timedelta(days=round(months * 30.4))
    if prices["date"].iloc[-1] < target_date:
        return np.nan
    future = prices[(prices["date"] > as_of) & (prices["date"] <= target_date)]
    if future.empty:
        return np.nan
    return float(np.log(future["close"].iloc[-1] / base_close))


def build_raw_panel(
    tickers: list[str],
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    universe: dict[str, dict[str, str]],
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """One row per (ticker, as_of) with raw indicators, price, dividend and
    anomaly features and forward-return labels. A (ticker, as_of) with no
    trade in the MAX_PRICE_STALENESS_DAYS before as_of gets no row (not listed
    yet / anymore); missing fundamentals become NaN rather than an error.
    Price data cached before 2026-09-28 has no close_raw / dividend columns:
    close then stands in for the quoted price and dividends count as none."""
    lagged = {t: apply_reporting_lag(add_fundamental_trends(df)) for t, df in fundamentals.items() if not df.empty}

    rows = []
    for ticker in tickers:
        price_df = prices.get(ticker)
        if price_df is None or price_df.empty:
            continue
        # yfinance returns empty closes on some days — including TODAY's row
        # before the market closes, which made every "today" snapshot price
        # NaN (and silently skipped the multiple rescaling) until 2026-09-23.
        # Drop them so the latest price is the last real trade.
        price_df = price_df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        if price_df.empty:
            continue
        raw_col = "close_raw" if "close_raw" in price_df.columns else "close"
        quoted = price_df[["date", raw_col]].rename(columns={raw_col: "price"}).dropna()
        payments = (
            price_df.loc[price_df["dividend"] > 0, ["date", "dividend"]]
            if "dividend" in price_df.columns else pd.DataFrame(columns=["date", "dividend"])
        )
        fund_df = lagged.get(ticker, pd.DataFrame())
        sector = universe.get(ticker, {}).get("sector", "Unknown")
        size_group = universe.get(ticker, {}).get("size", "large")

        for as_of in as_of_dates:
            window = price_df[price_df["date"] <= as_of].tail(260)
            if window.empty or (as_of - window["date"].iloc[-1]).days > MAX_PRICE_STALENESS_DAYS:
                continue
            close = window["close"].reset_index(drop=True)
            volume = window["volume"].reset_index(drop=True)
            quoted_now = quoted[quoted["date"] <= as_of]
            price = quoted_now["price"].iloc[-1] if len(quoted_now) else np.nan

            row = {"as_of": as_of, "ticker": ticker, "sector": sector, "size_group": size_group, "price": price}
            row.update(_as_of_fundamentals(fund_df, as_of))
            period_price = _rescale_multiples(row, quoted_now)
            row.update(_size_features(row, period_price))
            # already at the as_of price (no Finnhub period-end price involved)
            row["normalized_pe"] = price / (4 * row["eps_avg_3y"]) if row["eps_avg_3y"] > 0 else np.nan
            row.update(_dividend_features(payments, as_of, price))
            row.update(_technicals_asof(close, volume))
            row.update(_anomaly_signals_asof(close, volume))
            for h in HORIZONS_MONTHS:
                row[f"fwd_return_{h}m"] = _forward_log_return(price_df, as_of, close.iloc[-1], h)
            rows.append(row)

    panel = pd.DataFrame(rows)
    for col in ("eps_one_off_period", "per_share_break_period"):  # all-NaT vs. mixed rows infer different units
        if col in panel.columns:
            panel[col] = pd.to_datetime(panel[col]).astype("datetime64[ns]")
    return panel


def add_percentile_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """<indicator>_pct = percentile (0-100) within the same (as_of, sector),
    flipped for lower_is_better so higher always means "better/cheaper".
    Same definition as before the 2026-09-23 refactor (share of peers with a
    value <= this one; NaN if the value is missing or fewer than 3 peers have
    data), just vectorized."""
    df = panel.copy()
    groups = df.groupby(["as_of", "sector"])
    for indicator, direction in ALL_INDICATORS.items():
        ranks = groups[indicator].rank(method="max")
        counts = groups[indicator].transform("count")
        pct = (100.0 * ranks / counts).where(counts >= 3)
        df[f"{indicator}_pct"] = 100.0 - pct if direction == "lower_is_better" else pct
    return df
