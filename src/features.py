"""
Point-in-time panel: one row per (ticker, as_of) with only what was knowable
on that as_of date.

This is deliberately the single place where "what counts as known at time T"
is decided, so look-ahead bugs have one place to hide:
  - fundamentals: the latest quarter whose period + REPORTING_LAG_DAYS <= as_of
  - trailing_pe / price_to_book: Finnhub reports them at the period-end
    price; rescaled to the as_of price (config.RESCALE_MULTIPLES_TO_AS_OF_
    PRICE) using two prices that are both <= as_of
  - prices/technicals: daily rows with date <= as_of only
  - sector percentiles: ranked among rows of the SAME as_of (and sector)
  - fwd_return_*: the only columns that look past as_of — they are labels
    for fair_value.gap_return_test and must never be used as features

Derived indicators (no direct Finnhub field, see data.py):
  - revenue_growth_yoy: YoY % change in sales_per_share, computed on each
    ticker's full quarterly history BEFORE the reporting lag is applied (it
    only compares two already-reported quarters, so it is known as soon as
    the later one is)
  - dividend_yield: payout_ratio_ttm * eps / that row's price
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
# here: it needs a price and is derived per row in build_raw_panel.
_AS_OF_FUNDAMENTAL_KEYS = [k for k in FUNDAMENTAL_INDICATORS if k != "dividend_yield"] + [
    "eps",
    "payout_ratio_ttm",
    "sales_per_share",
]
_ANOMALY_SIGNAL_KEYS = ["price_spike_5d", "volume_zscore_63d"]


def build_as_of_dates(start: pd.Timestamp, today: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """Quarterly snapshots from `start`, plus `today` itself (normalized) so
    the screening report has a current snapshot instead of one that can be up
    to three months stale."""
    today = (today or pd.Timestamp.today()).normalize()
    dates = list(pd.date_range(start, today, freq=REBALANCE_FREQ))
    if not dates or dates[-1] != today:
        dates.append(today)
    return dates


def add_revenue_growth_yoy(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """revenue_growth_yoy = % change in sales_per_share vs. the quarter ~365
    days earlier (matched within +-45 days to absorb irregular gaps)."""
    df = fundamentals.sort_values("period").reset_index(drop=True).copy()
    if "sales_per_share" not in df.columns:
        df["revenue_growth_yoy"] = np.nan
        return df

    df["target_prior_period"] = df["period"] - pd.Timedelta(days=365)
    lookup = df[["period", "sales_per_share"]].rename(
        columns={"period": "matched_period", "sales_per_share": "prior_sales_per_share"}
    )
    merged = pd.merge_asof(
        df.sort_values("target_prior_period"),
        lookup.sort_values("matched_period"),
        left_on="target_prior_period",
        right_on="matched_period",
        direction="nearest",
        tolerance=pd.Timedelta(days=45),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (merged["sales_per_share"] - merged["prior_sales_per_share"]) / merged["prior_sales_per_share"].abs()
    merged["revenue_growth_yoy"] = growth.replace([np.inf, -np.inf], np.nan)
    return (
        merged.drop(columns=["target_prior_period", "matched_period", "prior_sales_per_share"])
        .sort_values("period")
        .reset_index(drop=True)
    )


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


def _rescale_multiples(row: dict, price_df: pd.DataFrame) -> None:
    """trailing_pe/price_to_book *= price(as_of) / price(period end), in
    place; originals kept as *_reported. Both prices are on or before as_of
    (period end < available_date <= as_of), so this adds no look-ahead.
    Uses yfinance adjusted closes, so splits cancel out; dividends paid in
    between shift the ratio by ~1-2% at most."""
    for col in ("trailing_pe", "price_to_book"):
        row[f"{col}_reported"] = row[col]
    if not RESCALE_MULTIPLES_TO_AS_OF_PRICE or pd.isna(row["fundamentals_period"]) or pd.isna(row["price"]):
        return
    at_period = price_df[price_df["date"] <= row["fundamentals_period"]]
    if at_period.empty or at_period["close"].iloc[-1] <= 0:
        return
    ratio = row["price"] / at_period["close"].iloc[-1]
    for col in ("trailing_pe", "price_to_book"):
        row[col] = row[col] * ratio


def _dividend_yield(payout_ratio_ttm: float, eps: float, price: float) -> float:
    """NaN on any missing input or non-positive price, never a nonsense value.
    Note: Finnhub leaves payoutRatioTTM empty for many non-payers (and some
    payers), so a NaN here often means "unknown", not "no dividend"."""
    if pd.isna(price) or price <= 0 or pd.isna(payout_ratio_ttm) or pd.isna(eps):
        return np.nan
    return float(payout_ratio_ttm * eps / price)


def _technicals_asof(close: pd.Series, volume: pd.Series) -> dict[str, float]:
    """close/volume: the trailing <=260 daily rows up to and including as_of."""
    if len(close) < 60:
        return {k: np.nan for k in TECHNICAL_INDICATORS}

    ma50 = close.tail(50).mean()
    ma200 = close.tail(200).mean() if len(close) >= 200 else np.nan
    high_52w = close.tail(252).max()
    daily_ret = close.pct_change().dropna().tail(63)
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
    """Label only (see module docstring). NaN if the horizon runs past the
    last available price, so recent snapshots don't get a truncated return."""
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
    """One row per (ticker, as_of) with raw indicators, price, anomaly
    signals and forward-return labels. A (ticker, as_of) with no trade in
    the MAX_PRICE_STALENESS_DAYS before as_of gets no row (not listed yet /
    anymore); missing fundamentals become NaN rather than an error."""
    lagged = {t: apply_reporting_lag(add_revenue_growth_yoy(df)) for t, df in fundamentals.items() if not df.empty}

    rows = []
    for ticker in tickers:
        price_df = prices.get(ticker)
        if price_df is None or price_df.empty:
            continue
        price_df = price_df.sort_values("date").reset_index(drop=True)
        fund_df = lagged.get(ticker, pd.DataFrame())
        sector = universe.get(ticker, {}).get("sector", "Unknown")

        for as_of in as_of_dates:
            window = price_df[price_df["date"] <= as_of].tail(260)
            if window.empty or (as_of - window["date"].iloc[-1]).days > MAX_PRICE_STALENESS_DAYS:
                continue
            close = window["close"].reset_index(drop=True)
            volume = window["volume"].reset_index(drop=True)
            price = close.iloc[-1]

            row = {"as_of": as_of, "ticker": ticker, "sector": sector, "price": price}
            row.update(_as_of_fundamentals(fund_df, as_of))
            _rescale_multiples(row, price_df)
            row["dividend_yield"] = _dividend_yield(row["payout_ratio_ttm"], row["eps"], price)
            row.update(_technicals_asof(close, volume))
            row.update(_anomaly_signals_asof(close, volume))
            for h in HORIZONS_MONTHS:
                row[f"fwd_return_{h}m"] = _forward_log_return(price_df, as_of, price, h)
            rows.append(row)

    return pd.DataFrame(rows)


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
