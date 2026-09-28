"""
Raw data collection: yfinance (daily prices + dividend history) + Finnhub
(point-in-time quarterly fundamentals), with a per-ticker Parquet cache.

FINNHUB FIELD NAMES — every quarterly series key was listed from a live
account on 2026-09-28 (AAPL) before the multiples/quality fields below were
added. Note salesPerShare is the QUARTER's sales (AAPL 7.44 vs. ~27 TTM), so
PSR comes from psTTM, and TTM sales per share is summed in features.py.
Derived in features.py from raw components fetched here:
- revenue_growth_yoy / revenue_cagr_3y ~ from salesPerShare (biased if
  buybacks move the share count a lot)
- dividend_yield / dividend features ~ from yfinance's actual dividend
  history (ex-dividend dates), not from payoutRatioTTM (30% missing)

Finnhub gives only the fiscal `period` end date, not the filing date, so
features.py treats each quarter as known from period + REPORTING_LAG_DAYS.

CACHE: fetching the whole universe takes minutes (Finnhub free tier is 50
calls/min), and a past quarter's fundamentals or a past day's price never
change, so each ticker's raw response is stored under `cache_dir` and reused.
The newest quarter / last few days CAN still change — pass
force_refresh=True before a run whose conclusions matter. In Colab the cache
lives on local disk and is lost on a runtime restart unless cache_dir points
into a mounted Google Drive.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from universe import TICKERS

# internal name -> Finnhub quarterly series field
FINNHUB_FIELD_MAP = {
    "trailing_pe": "peTTM",
    "price_to_book": "pb",
    "price_to_sales": "psTTM",
    "ev_to_ebitda": "evEbitdaTTM",
    "price_to_fcf": "pfcfTTM",
    # financial-statement ratios (candidate model features)
    "return_on_equity": "roeTTM",
    "roic": "roicTTM",
    "roa": "roaTTM",
    "gross_margin": "grossMargin",
    "operating_margin": "operatingMargin",
    "net_margin": "netMargin",
    "fcf_margin": "fcfMargin",
    "debt_to_equity": "totalDebtToEquity",
    "net_debt_to_capital": "netDebtToTotalCapital",
    "current_ratio": "currentRatio",
    "asset_turnover": "assetTurnoverTTM",
    "sga_to_sales": "sgaToSale",
    # raw components (see module docstring and features.py)
    "eps": "eps",
    "payout_ratio_ttm": "payoutRatioTTM",
    "sales_per_share": "salesPerShare",
    # levels used to rescale EV/EBITDA to the as_of price (features.py):
    # market cap = pb * bookValue, net debt = ev - market cap (same units)
    "enterprise_value": "ev",
    "book_value": "bookValue",
}

DEFAULT_CACHE_DIR = Path("data_cache")
_FINNHUB_CALLS_PER_MIN = 50
_MIN_INTERVAL_SEC = 60.0 / _FINNHUB_CALLS_PER_MIN
_RETRY_WAITS_SEC = (5, 20, 60)  # waits between attempts; 4 attempts total
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _is_retryable(exc: Exception) -> bool:
    """Network hiccups (timeouts, dropped connections) and rate-limit /
    server errors are worth retrying; anything else (e.g. 401 invalid API
    key) is not, and is raised immediately."""
    import requests

    if isinstance(exc, requests.exceptions.RequestException):
        return True
    return getattr(exc, "status_code", None) in _RETRYABLE_STATUS


def _with_retry(fetch, ticker: str, source: str, retry_all: bool = False):
    """Call fetch(ticker) up to len(_RETRY_WAITS_SEC)+1 times. Returns None
    (after printing why) if it still fails, so one bad ticker doesn't stop
    the whole collection; nothing is cached for it, so the next run retries.
    retry_all: treat every exception as transient (yfinance raises a zoo of
    exception types for what are usually network problems)."""
    for attempt, wait in enumerate((*_RETRY_WAITS_SEC, None), start=1):
        try:
            return fetch(ticker)
        except Exception as exc:  # noqa: BLE001 — filtered by _is_retryable below
            if not (retry_all or _is_retryable(exc)):
                raise
            if wait is None:
                print(f"  {ticker}: {source} failed after {attempt} attempts ({type(exc).__name__}) — skipped")
                return None
            print(f"  {ticker}: {source} {type(exc).__name__}, retrying in {wait}s ({attempt}/{len(_RETRY_WAITS_SEC)})")
            time.sleep(wait)


PRICE_COLUMNS = ["date", "close", "close_raw", "volume", "dividend"]


def fetch_price_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Daily prices via yfinance (period="max": TRAIN_START reaches 2004):
      close     — split- AND dividend-adjusted: for returns / volatility
      close_raw — split-adjusted only, i.e. the price actually quoted then:
                  for dividend yield and anything compared with Finnhub's
                  multiples. The difference is large: KO's 2010 adjusted close
                  is 60% of the quoted one, so a yield on adjusted prices
                  would be 1.67x too high.
      dividend  — cash dividend per share on its ex-date (0 on other days),
                  split-adjusted like close_raw."""
    hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if hist.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    if "Dividends" not in hist.columns:
        hist["Dividends"] = 0.0
    df = hist.reset_index()[["Date", "Adj Close", "Close", "Volume", "Dividends"]]
    df.columns = PRICE_COLUMNS
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["dividend"] = df["dividend"].fillna(0.0)
    return df


def _finnhub_client(api_key: str | None):
    """Imported here, not at module load: in a notebook, `pip install`
    after this module was first imported would otherwise leave a stale
    "not installed" state until the runtime restarts."""
    try:
        import finnhub
    except ImportError as exc:
        raise ImportError("pip install finnhub-python") from exc
    api_key = api_key or os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY가 필요합니다 (https://finnhub.io/register)")
    return finnhub.Client(api_key=api_key)


def fetch_fundamentals(ticker: str, api_key: str | None = None) -> pd.DataFrame:
    """Quarterly fundamentals via Finnhub: one row per fiscal `period` end
    date, one column per FINNHUB_FIELD_MAP key (a series Finnhub doesn't have
    for this ticker is an all-NaN column, so the cached file always records
    which fields were requested — see _cache_is_current)."""
    raw = _finnhub_client(api_key).company_basic_financials(ticker, "all")
    quarterly = (raw or {}).get("series", {}).get("quarterly", {})

    frames = []
    for name, field in FINNHUB_FIELD_MAP.items():
        series = quarterly.get(field)
        if not series:
            continue
        df = pd.DataFrame(series).rename(columns={"v": name})
        df["period"] = pd.to_datetime(df["period"])
        frames.append(df.set_index("period")[[name]])

    if not frames:
        return pd.DataFrame(columns=["period", *FINNHUB_FIELD_MAP])
    merged = pd.concat(frames, axis=1).reset_index().rename(columns={"index": "period"})
    merged = merged.reindex(columns=["period", *FINNHUB_FIELD_MAP])
    return merged.sort_values("period").reset_index(drop=True)


def _cache_path(cache_dir: str | Path, kind: str, ticker: str) -> Path:
    return Path(cache_dir) / kind / f"{ticker}.parquet"


def _cache_is_current(cached: pd.DataFrame, required: list[str]) -> bool:
    """A file cached before a field was added (a new FINNHUB_FIELD_MAP key,
    or the dividend column on prices, both 2026-09-28) lacks that column, so
    it is re-fetched once instead of silently leaving the new field empty."""
    return set(required).issubset(cached.columns)


def collect(
    tickers: list[str] = TICKERS,
    api_key: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Prices + fundamentals for every ticker, cache first. Only tickers
    that actually need a Finnhub call are rate-limited, so a full cache hit
    returns in seconds.

    Network failures are retried (_with_retry); a ticker that still fails is
    left out and reported at the end. Empty results are not cached either,
    so both kinds of gap are fetched again on the next run instead of being
    frozen into the cache — just re-run to fill them in."""
    prices: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    print(f"Prices for {len(tickers)} tickers (yfinance, cache={cache_dir})...")
    for ticker in tickers:
        path = _cache_path(cache_dir, "prices", ticker)
        if path.exists() and not force_refresh:
            cached = pd.read_parquet(path)
            if _cache_is_current(cached, PRICE_COLUMNS):
                prices[ticker] = cached
                continue
        df = _with_retry(fetch_price_history, ticker, "yfinance", retry_all=True)
        if df is None:
            failed.append(f"{ticker} (prices)")
            if path.exists():  # outdated cache still beats nothing for this run
                prices[ticker] = pd.read_parquet(path)
            continue
        prices[ticker] = df
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)

    fundamentals: dict[str, pd.DataFrame] = {}
    n_calls = 0
    print(f"Fundamentals for {len(tickers)} tickers (Finnhub, cache={cache_dir})...")
    for ticker in tickers:
        path = _cache_path(cache_dir, "fundamentals", ticker)
        if path.exists() and not force_refresh:
            cached = pd.read_parquet(path)
            if _cache_is_current(cached, list(FINNHUB_FIELD_MAP)):
                fundamentals[ticker] = cached
                continue
        if n_calls:
            time.sleep(_MIN_INTERVAL_SEC)
        n_calls += 1
        df = _with_retry(lambda t: fetch_fundamentals(t, api_key), ticker, "Finnhub")
        if df is None:
            failed.append(f"{ticker} (fundamentals)")
            if path.exists():  # outdated cache still beats nothing for this run
                fundamentals[ticker] = pd.read_parquet(path)
            continue
        fundamentals[ticker] = df
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
    print(f"  {n_calls} Finnhub calls made ({len(tickers) - n_calls} from cache)")
    if failed:
        print(f"  {len(failed)} fetches failed after retries: {', '.join(failed)}")
        print("  -> re-run build to fetch just these (everything else is cached)")
    return prices, fundamentals


def data_quality_report(prices: dict[str, pd.DataFrame], fundamentals: dict[str, pd.DataFrame]) -> list[str]:
    """Surface (not fix) obvious collection problems. A "no price history"
    ticker has so far always meant a delisting/merger or a ticker rename —
    check that before assuming a bug (universe.py docstring)."""
    print("\n--- Data quality report ---")
    flagged = []
    for ticker in sorted(prices.keys() | fundamentals.keys()):
        price_df, fund_df = prices.get(ticker), fundamentals.get(ticker)
        issues = []
        if price_df is None or price_df.empty:
            issues.append("no price history")
        elif (price_df["close"] <= 0).any():
            issues.append("non-positive close price present")
        if fund_df is None or fund_df.empty:
            issues.append("no fundamentals")
        else:
            recent = fund_df.sort_values("period").tail(8)
            if "trailing_pe" in fund_df.columns and fund_df["trailing_pe"].isna().all():
                issues.append("trailing_pe entirely missing")
            # BNY's series reads PER ~0.04 / PBR ~0.006 for its whole history —
            # the Finnhub symbol probably still maps to the pre-rename data (BK).
            if recent.get("trailing_pe", pd.Series(dtype=float)).median() < 1 or \
                    recent.get("price_to_book", pd.Series(dtype=float)).median() < 0.1:
                issues.append("implausible PER/PBR (<1 / <0.1) — check the Finnhub symbol")
        if issues:
            flagged.append(ticker)
            print(f"  {ticker}: {', '.join(issues)}")
    if not flagged:
        print("  no issues found")
    return flagged
