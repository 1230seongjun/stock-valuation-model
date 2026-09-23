"""
Data collection for the offline training/backtest pipeline.

Two sources, matching technical-spec.md section 4:
- yfinance: daily price/volume history (works today, no key needed).
- Finnhub: historical point-in-time fundamentals (needs a free API key —
  set FINNHUB_API_KEY as an env var; https://finnhub.io/register).

FIELD NAMES — VERIFIED against a live account on 2026-09-14 (AAPL,
`finnhub_field_check.py`). Two of the originally-assumed indicators don't
exist as direct fields in the free `company_basic_financials` series:

- `dividend_yield`: no direct field. Reconstructed as
  `(payoutRatioTTM * eps) / price_at_that_date` — needs `eps` and
  `payoutRatioTTM` (fetched below) plus the price series, so the actual
  division happens in features.py where price is already joined in.
- `revenue_growth_yoy`: no direct field either. Approximated from YoY
  change in `salesPerShare` (fetched below as `sales_per_share`) — a
  reasonable proxy, but biased if buybacks change share count a lot;
  revisit with SEC EDGAR revenue figures if that turns out to matter.

Both raw components are fetched here under their own internal names;
features.py does the actual derived-indicator math once price is joined.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    import finnhub
except ImportError:  # pragma: no cover
    finnhub = None

# internal indicator name -> Finnhub quarterly series field name
# (verified 2026-09-14 against a live AAPL response — see module docstring)
FINNHUB_FIELD_MAP = {
    "trailing_pe": "peTTM",
    "price_to_book": "pb",
    "return_on_equity": "roeTTM",
    "debt_to_equity": "totalDebtToEquity",
    "operating_margin": "operatingMargin",
}

# raw components needed to DERIVE dividend_yield and revenue_growth_yoy
# in features.py (not direct fields themselves — see module docstring)
FINNHUB_DERIVED_COMPONENTS_MAP = {
    "eps": "eps",
    "payout_ratio_ttm": "payoutRatioTTM",
    "sales_per_share": "salesPerShare",
}

_FINNHUB_CALLS_PER_MIN = 50
_MIN_INTERVAL_SEC = 60.0 / _FINNHUB_CALLS_PER_MIN


def fetch_price_history(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Daily OHLCV via yfinance. Returns columns: date, close, volume."""
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    df = hist.reset_index()[["Date", "Close", "Volume"]]
    df.columns = ["date", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


def fetch_point_in_time_fundamentals(ticker: str, api_key: str | None = None) -> pd.DataFrame:
    """Historical quarterly fundamentals via Finnhub, one row per reported
    quarter with a `period` date (Finnhub's period end date) and one column
    per indicator in FINNHUB_FIELD_MAP.

    NOTE: Finnhub's free `company_basic_financials` series does not include
    a distinct "filed" (public disclosure) date the way SEC EDGAR does —
    only the fiscal `period` end date. For strict look-ahead-bias avoidance
    (technical-spec.md 4.2), treat each quarter's fundamentals as available
    starting `period + ~45 days` (typical reporting lag) rather than on
    `period` itself, until you've confirmed actual filing dates. This
    function returns the raw period dates; the lag is applied later in
    features.py so it stays a single, visible assumption.
    """
    if finnhub is None:
        raise ImportError("pip install finnhub-python")
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Set FINNHUB_API_KEY (see https://finnhub.io/register)")

    client = finnhub.Client(api_key=api_key)
    raw = client.company_basic_financials(ticker, "all")
    quarterly = (raw or {}).get("series", {}).get("quarterly", {})

    all_fields = {**FINNHUB_FIELD_MAP, **FINNHUB_DERIVED_COMPONENTS_MAP}
    frames = []
    for internal_name, finnhub_key in all_fields.items():
        series = quarterly.get(finnhub_key)
        if not series:
            continue
        df = pd.DataFrame(series)  # columns: period, v
        df = df.rename(columns={"v": internal_name})
        df["period"] = pd.to_datetime(df["period"])
        frames.append(df.set_index("period")[[internal_name]])

    if not frames:
        return pd.DataFrame(columns=["period", *all_fields.keys()])

    merged = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "period"})
    return merged.sort_values("period").reset_index(drop=True)


def fetch_universe_fundamentals(tickers: list[str], api_key: str | None = None) -> dict[str, pd.DataFrame]:
    """Sequential fetch respecting Finnhub's free-tier rate limit
    (50 calls/min). For N tickers this takes roughly N * 1.2 seconds."""
    out = {}
    for i, ticker in enumerate(tickers):
        out[ticker] = fetch_point_in_time_fundamentals(ticker, api_key)
        if i < len(tickers) - 1:
            time.sleep(_MIN_INTERVAL_SEC)
    return out


# ---- On-disk caching (added 2026-09-18) ----------------------------------
# yfinance/Finnhub collection for the full universe takes minutes (Finnhub
# alone is ~N*1.2s from the rate limit above — 283 tickers is ~6 minutes),
# and almost none of that data changes between re-runs on the same day: a
# given historical quarter's fundamentals or a given past trading day's
# price never change once fetched. Re-fetching everything on every
# iteration of the pipeline (tweaking features.py, re-running the model,
# testing screening.py) wastes minutes AND burns through Finnhub's free-tier
# rate limit for no reason. These cached wrappers save each ticker's raw
# response to its own Parquet file under `cache_dir` and skip the network
# call entirely on a hit.
#
# This is a SEPARATE cache from run_real_data.py's panel.parquet: that one
# holds the fully processed, feature-engineered panel (post
# add_percentile_scores) — great for re-running the model/screening report,
# useless if you need to change something in features.py or data_collection
# itself, since the raw inputs aren't in it. This cache holds the raw
# per-ticker API responses, so changing feature engineering doesn't require
# re-hitting the API either.
#
# Caveat: only the OLDEST-first parts of history are truly immutable. The
# most recent quarter's fundamentals or the last few days' prices can still
# be revised/extended after they're first cached (a late earnings revision,
# or simply that "today" moves forward each run since
# config.as_of_dates in run_real_data.py reaches to pd.Timestamp.today()).
# Pass force_refresh=True (or delete the relevant file/the whole
# `cache_dir`) to pick up anything new — do this at least before a "final"
# run you're about to draw conclusions from, not just during iteration.
#
# Cache lives on local disk, so in Colab it survives repeated cells within
# the SAME runtime but is lost on a runtime restart/disconnect. To persist
# across sessions, mount Google Drive and pass a cache_dir under
# /content/drive/MyDrive/... instead of the default.
DEFAULT_CACHE_DIR = Path("data_cache")


def _price_cache_path(cache_dir: Path, ticker: str) -> Path:
    return Path(cache_dir) / "prices" / f"{ticker}.parquet"


def _fundamentals_cache_path(cache_dir: Path, ticker: str) -> Path:
    return Path(cache_dir) / "fundamentals" / f"{ticker}.parquet"


def fetch_price_history_cached(
    ticker: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    period: str = "max",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """fetch_price_history, but reads/writes a per-ticker Parquet cache
    file first. See the module-level caching note above for the staleness
    caveat."""
    path = _price_cache_path(cache_dir, ticker)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    df = fetch_price_history(ticker, period=period)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_point_in_time_fundamentals_cached(
    ticker: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """fetch_point_in_time_fundamentals, but reads/writes a per-ticker
    Parquet cache file first. See the module-level caching note above for
    the staleness caveat."""
    path = _fundamentals_cache_path(cache_dir, ticker)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    df = fetch_point_in_time_fundamentals(ticker, api_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_universe_fundamentals_cached(
    tickers: list[str],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    api_key: str | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """fetch_universe_fundamentals, but skips the network entirely (and the
    rate-limit sleep) for any ticker already cached — only tickers that
    actually need a fresh Finnhub call get the 1.2s spacing. On a full
    cache hit for the whole universe this returns in well under a second
    instead of ~N*1.2s."""
    out = {}
    n_api_calls = 0
    for ticker in tickers:
        path = _fundamentals_cache_path(cache_dir, ticker)
        if path.exists() and not force_refresh:
            out[ticker] = pd.read_parquet(path)
            continue
        if n_api_calls > 0:
            time.sleep(_MIN_INTERVAL_SEC)
        out[ticker] = fetch_point_in_time_fundamentals(ticker, api_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        out[ticker].to_parquet(path)
        n_api_calls += 1
    return out
