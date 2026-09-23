"""
Point-in-time feature engineering.

Builds one row per (ticker, as_of_date) with:
  - raw indicator values (fundamentals as of `as_of_date`, technicals computed
    from price history up to and including `as_of_date`)
  - sector-relative percentile per indicator, computed ONLY from other
    tickers' values that were themselves already available at `as_of_date`
    (no look-ahead — technical-spec.md section 4.2)
  - a composite score (mean of available percentiles)
  - forward log returns for each horizon in config.HORIZONS_MONTHS

This is deliberately the single place where "what counts as known at time
T" is decided, so look-ahead bugs have one place to hide instead of three.

Two of the FUNDAMENTAL_INDICATORS (dividend_yield, revenue_growth_yoy) have
no direct Finnhub field (data_collection.py docstring) and are derived here
from the raw components data_collection.py fetches instead:
  - revenue_growth_yoy: YoY % change in sales_per_share, computed once per
    ticker across its whole fundamentals history (add_revenue_growth_yoy),
    before the reporting lag is applied.
  - dividend_yield: (payout_ratio_ttm * eps) / price, computed per
    (ticker, as_of) row in build_raw_panel since it needs that day's price,
    which isn't part of the fundamentals frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ALL_INDICATORS,
    FUNDAMENTAL_INDICATORS,
    HORIZONS_MONTHS,
    TECHNICAL_INDICATORS,
    VALUE_COMPOSITE_INDICATORS,
)

REPORTING_LAG_DAYS = 45  # see data_collection.py docstring — verify vs. real filing dates later

# FUNDAMENTAL_INDICATORS keys that exist as direct Finnhub-sourced columns
# (i.e. all of them except dividend_yield, which is derived per-row in
# build_raw_panel once a price is available).
_DIRECT_FUNDAMENTAL_KEYS = [k for k in FUNDAMENTAL_INDICATORS if k != "dividend_yield"]
# raw components (data_collection.FINNHUB_DERIVED_COMPONENTS_MAP) needed
# downstream to compute dividend_yield.
_DIVIDEND_YIELD_INPUT_KEYS = ["eps", "payout_ratio_ttm"]
_AS_OF_FUNDAMENTAL_KEYS = _DIRECT_FUNDAMENTAL_KEYS + _DIVIDEND_YIELD_INPUT_KEYS


def apply_reporting_lag(fundamentals: pd.DataFrame, lag_days: int = REPORTING_LAG_DAYS) -> pd.DataFrame:
    df = fundamentals.copy()
    df["available_date"] = df["period"] + pd.Timedelta(days=lag_days)
    return df


def add_revenue_growth_yoy(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Derive revenue_growth_yoy as the % change in sales_per_share vs. the
    quarter ~1 year (365 days) earlier, matched with a +-45 day tolerance to
    absorb irregular reporting gaps. No direct Finnhub revenue-growth field
    exists (data_collection.py docstring) — this is a proxy, biased if
    buybacks/issuance move the share count a lot; revisit with SEC EDGAR
    revenue figures if that turns out to matter.

    Call this BEFORE apply_reporting_lag: it only needs `period` and
    `sales_per_share`, and operates on one ticker's full fundamentals
    history at a time (so it must run before rows get filtered per as_of).
    """
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

    current = merged["sales_per_share"]
    prior = merged["prior_sales_per_share"]
    with np.errstate(divide="ignore", invalid="ignore"):
        growth = (current - prior) / prior.abs()
    merged["revenue_growth_yoy"] = growth.replace([np.inf, -np.inf], np.nan)

    return (
        merged.drop(columns=["target_prior_period", "matched_period", "prior_sales_per_share"])
        .sort_values("period")
        .reset_index(drop=True)
    )


def _as_of_fundamentals(fundamentals: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """Most recent fundamentals row whose available_date <= as_of.

    Returns every direct FUNDAMENTAL_INDICATORS value (revenue_growth_yoy
    included, since add_revenue_growth_yoy adds it as a real column before
    this runs) PLUS the raw eps/payout_ratio_ttm needed to derive
    dividend_yield afterwards in build_raw_panel. dividend_yield itself is
    NOT included here — it needs a price, which this function doesn't have.
    """
    if fundamentals.empty:
        return {k: np.nan for k in _AS_OF_FUNDAMENTAL_KEYS}
    valid = fundamentals[fundamentals["available_date"] <= as_of]
    if valid.empty:
        return {k: np.nan for k in _AS_OF_FUNDAMENTAL_KEYS}
    row = valid.sort_values("available_date").iloc[-1]
    return {k: row.get(k, np.nan) for k in _AS_OF_FUNDAMENTAL_KEYS}


def _dividend_yield(payout_ratio_ttm: float, eps: float, price: float | None) -> float:
    """(trailing payout ratio * trailing EPS) / price — see module docstring.
    Any missing input, or a non-positive price, yields NaN rather than a
    divide-by-zero or a nonsense negative yield."""
    if price is None or pd.isna(price) or price <= 0:
        return np.nan
    if pd.isna(payout_ratio_ttm) or pd.isna(eps):
        return np.nan
    return float((payout_ratio_ttm * eps) / price)


def _price_asof(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series | None:
    valid = prices[prices["date"] <= as_of]
    return None if valid.empty else valid.iloc[-1]


def _technicals_asof(prices: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    window = prices[prices["date"] <= as_of].tail(260)  # ~1y of trading days
    if len(window) < 60:
        return {k: np.nan for k in TECHNICAL_INDICATORS}

    close = window["close"]
    volume = window["volume"]

    ma50 = close.tail(50).mean()
    ma200 = close.tail(200).mean() if len(close) >= 200 else np.nan
    ma50_vs_ma200 = (ma50 / ma200 - 1) if ma200 and not np.isnan(ma200) else np.nan

    high_52w = close.tail(252).max()
    current = close.iloc[-1]
    pct_from_52w_high = (high_52w - current) / high_52w if high_52w else np.nan

    daily_ret = close.pct_change().dropna().tail(63)
    volatility_63d = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 5 else np.nan

    vol_21d = volume.tail(21).mean()
    vol_63d = volume.tail(63).mean()
    relative_volume = (vol_21d / vol_63d) if vol_63d else np.nan

    return {
        "ma50_vs_ma200": ma50_vs_ma200,
        "pct_from_52w_high": pct_from_52w_high,
        "volatility_63d": volatility_63d,
        "relative_volume": relative_volume,
    }


# Diagnostic-only signals for screening.py's meme-stock flag (2026-09-18
# pivot to a relative-valuation screening tool). Deliberately NOT added to
# config.TECHNICAL_INDICATORS/ALL_INDICATORS — these are not meant to feed
# composite_score or the factor-validity tests already run; they're a
# separate "is this snapshot even trustworthy" check computed alongside the
# real indicators, using the same point-in-time price window so there's no
# extra look-ahead risk.
_ANOMALY_SIGNAL_KEYS = ["price_spike_5d", "volume_zscore_63d"]


def _anomaly_signals_asof(prices: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float]:
    """price_spike_5d: 5-trading-day cumulative log return as of this date
    (a sudden, large move in either direction — the "something just happened
    to this stock" signal a slower 63-day volatility figure would smooth
    away). volume_zscore_63d: how many standard deviations the most recent
    day's volume is above/below its own trailing 63-day mean (a single-day
    volume spike, as opposed to relative_volume's 21d-vs-63d-average ratio,
    which is a sustained-elevated-activity measure, not a spike detector).
    Both need their own >=60-row window check since screening.py's meme
    flag combines them and a NaN in either silently defeats the AND."""
    window = prices[prices["date"] <= as_of].tail(260)
    if len(window) < 60:
        return {k: np.nan for k in _ANOMALY_SIGNAL_KEYS}

    close = window["close"]
    volume = window["volume"]

    if len(close) >= 6 and close.iloc[-6] > 0:
        price_spike_5d = float(np.log(close.iloc[-1] / close.iloc[-6]))
    else:
        price_spike_5d = np.nan

    vol_63d = volume.tail(63)
    vol_mean, vol_std = vol_63d.mean(), vol_63d.std()
    volume_zscore_63d = float((volume.iloc[-1] - vol_mean) / vol_std) if vol_std else np.nan

    return {"price_spike_5d": price_spike_5d, "volume_zscore_63d": volume_zscore_63d}


def _forward_log_return(prices: pd.DataFrame, as_of: pd.Timestamp, months: int) -> float:
    base = _price_asof(prices, as_of)
    if base is None:
        return np.nan
    target_date = as_of + pd.Timedelta(days=round(months * 30.4))
    future = prices[(prices["date"] > as_of) & (prices["date"] <= target_date)]
    if future.empty:
        return np.nan
    future_price = future.iloc[-1]["close"]
    return float(np.log(future_price / base["close"]))


def build_raw_panel(
    tickers: list[str],
    prices: dict[str, pd.DataFrame],
    fundamentals: dict[str, pd.DataFrame],
    universe: dict[str, dict[str, str]],
    as_of_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """One row per (ticker, as_of_date) with raw indicator values + forward
    returns. No cross-sectional percentiles yet — see add_percentile_scores.

    dividend_yield and revenue_growth_yoy don't exist as direct Finnhub
    fields (see module docstring) — revenue_growth_yoy is derived once per
    ticker below (add_revenue_growth_yoy, before the lag is applied, since
    it needs the ticker's whole history); dividend_yield is derived per row
    right here, since it needs that row's price.
    """
    lagged_fundamentals = {
        t: apply_reporting_lag(add_revenue_growth_yoy(df)) for t, df in fundamentals.items() if not df.empty
    }

    rows = []
    for ticker in tickers:
        price_df = prices.get(ticker)
        if price_df is None or price_df.empty:
            continue
        fund_df = lagged_fundamentals.get(ticker, pd.DataFrame())

        for as_of in as_of_dates:
            row = {
                "as_of": as_of,
                "ticker": ticker,
                "sector": universe.get(ticker, {}).get("sector", "Unknown"),
            }
            fund_values = _as_of_fundamentals(fund_df, as_of)
            price_row = _price_asof(price_df, as_of)
            fund_values["dividend_yield"] = _dividend_yield(
                fund_values.get("payout_ratio_ttm", np.nan),
                fund_values.get("eps", np.nan),
                price_row["close"] if price_row is not None else None,
            )
            row.update(fund_values)
            # Raw close price as of this snapshot — not an indicator itself
            # (not in ALL_INDICATORS, doesn't touch composite_score/FEATURE_COLS),
            # but needed downstream by screening.classify_valuation_transition
            # to tell a price-driven vs. earnings-driven valuation-label flip
            # apart (2026-09-23). Added here rather than recomputed later so
            # every consumer uses the exact same point-in-time price this row's
            # dividend_yield/technicals already used.
            row["price"] = price_row["close"] if price_row is not None else np.nan
            row.update(_technicals_asof(price_df, as_of))
            row.update(_anomaly_signals_asof(price_df, as_of))
            for h in HORIZONS_MONTHS:
                row[f"fwd_return_{h}m"] = _forward_log_return(price_df, as_of, h)
            rows.append(row)

    return pd.DataFrame(rows)


def _percentile_rank(value: float, population: pd.Series) -> float:
    pop = population.dropna()
    if pd.isna(value) or len(pop) < 3:
        return np.nan
    return 100.0 * (pop <= value).sum() / len(pop)


def add_percentile_scores(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional (same as_of + sector) percentile per indicator, plus
    a composite_score column. Operates only on rows already in `panel`, so
    it never reaches outside what build_raw_panel already restricted to
    known-at-the-time values."""
    df = panel.copy()

    for indicator, direction in ALL_INDICATORS.items():
        pct_col = f"{indicator}_pct"
        df[pct_col] = np.nan
        for (as_of, sector), group in df.groupby(["as_of", "sector"]):
            pop = group[indicator]
            ranks = pop.apply(lambda v: _percentile_rank(v, pop))
            if direction == "lower_is_better":
                ranks = 100.0 - ranks
            df.loc[group.index, pct_col] = ranks

    pct_cols = [f"{k}_pct" for k in ALL_INDICATORS]
    df["composite_score"] = df[pct_cols].mean(axis=1, skipna=True)

    # Diagnostic-only second score (config.VALUE_COMPOSITE_INDICATORS'
    # docstring: isolates the indicators that showed the theoretically
    # expected sign in the 2026-09-18 real-data run, to check whether they
    # get canceled out inside the full equal-weighted composite_score).
    value_pct_cols = [f"{k}_pct" for k in VALUE_COMPOSITE_INDICATORS]
    df["value_composite_score"] = df[value_pct_cols].mean(axis=1, skipna=True)
    return df
