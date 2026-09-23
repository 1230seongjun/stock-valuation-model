"""
Raw data collection: yfinance (daily prices) + Finnhub (point-in-time
quarterly fundamentals), with a per-ticker Parquet cache.

FINNHUB FIELD NAMES — verified against a live account on 2026-09-14 (AAPL).
Two indicators have no direct field in the free `company_basic_financials`
series and are derived in features.py from raw components fetched here:
- dividend_yield     ~ payoutRatioTTM * eps / price at that date
- revenue_growth_yoy ~ YoY change in salesPerShare (biased if buybacks move
  the share count a lot — revisit with SEC EDGAR revenue if that matters)

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
    "return_on_equity": "roeTTM",
    "debt_to_equity": "totalDebtToEquity",
    "operating_margin": "operatingMargin",
    # raw components for derived indicators (see module docstring)
    "eps": "eps",
    "payout_ratio_ttm": "payoutRatioTTM",
    "sales_per_share": "salesPerShare",
}

DEFAULT_CACHE_DIR = Path("data_cache")
_FINNHUB_CALLS_PER_MIN = 50
_MIN_INTERVAL_SEC = 60.0 / _FINNHUB_CALLS_PER_MIN


def fetch_price_history(ticker: str, period: str = "max") -> pd.DataFrame:
    """Daily close/volume via yfinance. Returns columns: date, close, volume.
    period="max" because TRAIN_START reaches back to 2004."""
    hist = yf.Ticker(ticker).history(period=period, interval="1d")
    if hist.empty:
        return pd.DataFrame(columns=["date", "close", "volume"])
    df = hist.reset_index()[["Date", "Close", "Volume"]]
    df.columns = ["date", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
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
    date, one column per FINNHUB_FIELD_MAP key (missing series -> absent)."""
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
    return merged.sort_values("period").reset_index(drop=True)


def fetch_current_metrics(ticker: str, api_key: str | None = None) -> dict:
    """Finnhub's CURRENT-price metrics (not cached): used only by
    `main.py verify-multiples` to check the period-end-price assumption
    behind config.RESCALE_MULTIPLES_TO_AS_OF_PRICE."""
    raw = _finnhub_client(api_key).company_basic_financials(ticker, "all")
    return (raw or {}).get("metric", {})


def _cache_path(cache_dir: str | Path, kind: str, ticker: str) -> Path:
    return Path(cache_dir) / kind / f"{ticker}.parquet"


def collect(
    tickers: list[str] = TICKERS,
    api_key: str | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Prices + fundamentals for every ticker, cache first. Only tickers
    that actually need a Finnhub call are rate-limited, so a full cache hit
    returns in seconds."""
    prices: dict[str, pd.DataFrame] = {}
    print(f"Prices for {len(tickers)} tickers (yfinance, cache={cache_dir})...")
    for ticker in tickers:
        path = _cache_path(cache_dir, "prices", ticker)
        if path.exists() and not force_refresh:
            prices[ticker] = pd.read_parquet(path)
            continue
        prices[ticker] = fetch_price_history(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        prices[ticker].to_parquet(path)

    fundamentals: dict[str, pd.DataFrame] = {}
    n_calls = 0
    print(f"Fundamentals for {len(tickers)} tickers (Finnhub, cache={cache_dir})...")
    for ticker in tickers:
        path = _cache_path(cache_dir, "fundamentals", ticker)
        if path.exists() and not force_refresh:
            fundamentals[ticker] = pd.read_parquet(path)
            continue
        if n_calls:
            time.sleep(_MIN_INTERVAL_SEC)
        fundamentals[ticker] = fetch_fundamentals(ticker, api_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fundamentals[ticker].to_parquet(path)
        n_calls += 1
    print(f"  {n_calls} Finnhub calls made ({len(tickers) - n_calls} from cache)")
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
