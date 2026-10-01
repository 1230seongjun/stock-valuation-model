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

SEC XBRL company facts (load_sec_facts, 2026-10-01): net income, operating
cash flow and total assets WITH their filing dates, for the accruals feature
(features._accruals_asof). sec.gov refuses scripted downloads from some
networks ("Undeclared Automated Tool", 403), so the two bulk files are
downloaded by hand into <cache_dir>/sec/ (see SEC_* below). Without them the
feature is simply empty and the model fits as before. XBRL starts 2009-2011,
so earlier snapshots have no accruals either.

CACHE: fetching the whole universe takes minutes (Finnhub free tier is 50
calls/min), and a past quarter's fundamentals or a past day's price never
change, so each ticker's raw response is stored under `cache_dir` and reused.
The newest quarter / last few days CAN still change — pass
force_refresh=True before a run whose conclusions matter. Cached prices
older than PRICE_REFRESH_DAYS are re-fetched automatically (until
2026-09-29 they never were, so a later build's "today" snapshot silently
used the prices of the day the cache was made); a re-fetch that comes back
as a short delisting stub does not replace a longer cached history. In Colab the cache
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
PRICE_REFRESH_DAYS = 3  # covers a weekend; a cache older than this is re-fetched
# A re-fetched history that starts this much later than the cached one is a
# truncated stub, not an update (see _is_truncated) — the cache is kept.
TRUNCATED_HISTORY_DAYS = 365
# Fewer daily rows than this is reported by data_quality_report: too short
# for any snapshot's technicals, and usually a delisting stub (AVB, EA, EQR,
# HLX, LEG on 2026-09-29: 1-8 rows each, so the panel silently had no row
# for them) or a very recent IPO.
MIN_PRICE_ROWS = 60
_PROGRESS_EVERY = 200
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
    # class shares: Wikipedia/Finnhub write MOG.A, Yahoo writes MOG-A
    hist = yf.Ticker(ticker.replace(".", "-")).history(period=period, interval="1d", auto_adjust=False)
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


def fetch_fiscal_year_ends(ticker: str, api_key: str | None = None) -> pd.DataFrame:
    """Fiscal year-end dates (one row per `period`) from Finnhub's annual
    series — which quarterly period is a fiscal Q4, whose numbers come with
    the 10-K and so later (features.ANNUAL_REPORTING_LAG_DAYS). AAPL's are
    late September on a 52/53-week calendar, WMT's January 31."""
    raw = _finnhub_client(api_key).company_basic_financials(ticker, "all")
    annual = (raw or {}).get("series", {}).get("annual", {})
    periods = {d["period"] for series in annual.values() for d in series if d.get("period")}
    return pd.DataFrame({"period": sorted(pd.to_datetime(list(periods)))})


def collect_fiscal_year_ends(
    tickers: list[str], api_key: str | None = None, cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> dict[str, list[pd.Timestamp]]:
    """Fiscal year-ends per ticker, cache first (a company rarely changes its
    fiscal year; force_refresh re-fetches). Rate-limited like the
    fundamentals; a ticker that fails or has no annual series is left out
    and features.py falls back to December."""
    out: dict[str, list[pd.Timestamp]] = {}
    n_calls, failed = 0, []
    print(f"Fiscal year-ends for {len(tickers)} tickers (Finnhub, cache={cache_dir})...")
    for i, ticker in enumerate(tickers, start=1):
        if i % _PROGRESS_EVERY == 0:
            print(f"  fiscal year-ends {i}/{len(tickers)} ({n_calls} calls so far)")
        path = _cache_path(cache_dir, "fiscal_year", ticker)
        if path.exists() and not force_refresh:
            out[ticker] = list(pd.read_parquet(path)["period"])
            continue
        if n_calls:
            time.sleep(_MIN_INTERVAL_SEC)
        n_calls += 1
        df = _with_retry(lambda t: fetch_fiscal_year_ends(t, api_key), ticker, "Finnhub")
        if df is None or df.empty:
            failed.append(ticker)
            continue
        out[ticker] = list(df["period"])
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    print(f"  {n_calls} Finnhub calls made ({len(tickers) - n_calls} from cache)")
    if failed:
        shown = ", ".join(failed[:_MAX_LISTED]) + (f" ... (+{len(failed) - _MAX_LISTED})" if len(failed) > _MAX_LISTED else "")
        print(f"  no fiscal year-ends for {len(failed)} (December assumed): {shown}")
    return out


def _cache_path(cache_dir: str | Path, kind: str, ticker: str) -> Path:
    return Path(cache_dir) / kind / f"{ticker}.parquet"


def _prices_are_recent(cached: pd.DataFrame, today: pd.Timestamp | None = None) -> bool:
    today = (today or pd.Timestamp.today()).normalize()
    last = pd.to_datetime(cached["date"]).max() if len(cached) else pd.NaT
    return pd.notna(last) and (today - last).days <= PRICE_REFRESH_DAYS


def _cache_is_current(cached: pd.DataFrame, required: list[str]) -> bool:
    """A file cached before a field was added (a new FINNHUB_FIELD_MAP key,
    or the dividend column on prices, both 2026-09-28) lacks that column, so
    it is re-fetched once instead of silently leaving the new field empty."""
    return set(required).issubset(cached.columns)


def _is_truncated(fetched: pd.DataFrame, cached: pd.DataFrame) -> bool:
    """yfinance returns only the last few days for some acquired/delisted
    tickers (2026-09-29: EA 1 row, AVB 2, HLX 8, where 20 years were
    expected) — and, once, for two listed ones: AEP and XEL came back as a
    single day on 2026-09-29 and overwrote 60+ years of cache (a re-fetch an
    hour later returned the full history). Overwriting a cached history with
    such a stub would throw the history away, so a fetch starting
    TRUNCATED_HISTORY_DAYS after the cached start is treated as truncated."""
    if fetched.empty or cached.empty:
        return False
    first_fetched = pd.to_datetime(fetched["date"]).min()
    first_cached = pd.to_datetime(cached["date"]).min()
    return (first_fetched - first_cached).days > TRUNCATED_HISTORY_DAYS


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
    truncated: list[str] = []
    print(f"Prices for {len(tickers)} tickers (yfinance, cache={cache_dir})...")
    for i, ticker in enumerate(tickers, start=1):
        if i % _PROGRESS_EVERY == 0:
            print(f"  prices {i}/{len(tickers)}")
        path = _cache_path(cache_dir, "prices", ticker)
        if path.exists() and not force_refresh:
            cached = pd.read_parquet(path)
            # a delisted ticker's cache never becomes recent: re-fetching it
            # returns the same history (or nothing, then the cache is kept)
            if _cache_is_current(cached, PRICE_COLUMNS) and _prices_are_recent(cached):
                prices[ticker] = cached
                continue
        df = _with_retry(fetch_price_history, ticker, "yfinance", retry_all=True)
        if df is not None and df.empty and path.exists():
            prices[ticker] = pd.read_parquet(path)  # delisted since it was cached: keep its history
            continue
        if df is not None and path.exists():
            cached = pd.read_parquet(path)
            if _is_truncated(df, cached):
                prices[ticker] = cached  # a delisting stub: keep the full history
                truncated.append(ticker)
                continue
        if df is None:
            failed.append(f"{ticker} (prices)")
            if path.exists():  # outdated cache still beats nothing for this run
                prices[ticker] = pd.read_parquet(path)
            continue
        prices[ticker] = df
        if not df.empty:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
    if truncated:
        print(f"  {len(truncated)} re-fetched histories were truncated stubs, cached history kept: {', '.join(truncated)}")

    fundamentals: dict[str, pd.DataFrame] = {}
    n_calls = 0
    print(f"Fundamentals for {len(tickers)} tickers (Finnhub, cache={cache_dir})...")
    for i, ticker in enumerate(tickers, start=1):
        if i % _PROGRESS_EVERY == 0:
            print(f"  fundamentals {i}/{len(tickers)} ({n_calls} calls so far)")
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


# Hand-downloaded SEC bulk files (see the module docstring), in <cache_dir>/sec/:
#   https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip  (~1.4 GB, never unzipped)
#   https://www.sec.gov/files/company_tickers.json
SEC_DIR = "sec"
SEC_COMPANYFACTS = "companyfacts.zip"
SEC_TICKERS = "company_tickers.json"
SEC_TAGS = ("NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities", "Assets")
SEC_FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")
SEC_FACT_COLUMNS = ["tag", "start", "end", "val", "filed"]


def _sec_extract(pairs: list[tuple[str, int]], zip_path: Path, out_dir: Path) -> None:
    """SEC_TAGS of each (ticker, CIK) from the zip into out_dir/<ticker>.parquet.
    Balance-sheet values are instants (no start): start = end."""
    import json
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        for ticker, cik in pairs:
            name = f"CIK{cik:010d}.json"
            rows = []
            if name in names:
                facts = json.loads(z.read(name)).get("facts", {}).get("us-gaap", {})
                rows = [(tag, x.get("start", x["end"]), x["end"], x["val"], x["filed"])
                        for tag in SEC_TAGS for x in facts.get(tag, {}).get("units", {}).get("USD", [])
                        if x.get("form") in SEC_FORMS]
            pd.DataFrame(rows, columns=SEC_FACT_COLUMNS).to_parquet(out_dir / f"{ticker}.parquet")


def load_sec_facts(
    tickers: list[str], cache_dir: str | Path = DEFAULT_CACHE_DIR, force_refresh: bool = False,
    n_jobs: int | None = None,
) -> dict[str, pd.DataFrame]:
    """SEC_TAGS facts per ticker (columns SEC_FACT_COLUMNS, dates as
    Timestamps), extracted once from the hand-downloaded companyfacts.zip into
    <cache_dir>/sec_facts/ and re-extracted when the zip changes (a newer
    download) or force_refresh. Tickers are matched by CIK through
    company_tickers.json (class shares: our MOG.A = SEC's MOG-A). No files ->
    {} with a note; a ticker without an SEC filer (OZK, PFBC file with bank
    regulators) gets no entry."""
    from config import N_JOBS

    sec_dir, out_dir = Path(cache_dir) / SEC_DIR, Path(cache_dir) / "sec_facts"
    zip_path, map_path = sec_dir / SEC_COMPANYFACTS, sec_dir / SEC_TICKERS
    if not (zip_path.exists() and map_path.exists()):
        print(f"SEC facts: {zip_path} / {map_path.name} not found -> accruals left empty "
              f"(download them by hand, see data.SEC_DIR)")
        return {}
    import json

    stat = zip_path.stat()
    source = f"{stat.st_size} {int(stat.st_mtime)}"
    marker = out_dir / "_source.txt"
    fresh = force_refresh or not marker.exists() or marker.read_text() != source
    ciks = {v["ticker"].upper(): int(v["cik_str"]) for v in json.loads(map_path.read_text()).values()}
    pairs = [(t, ciks.get(t.upper()) or ciks.get(t.upper().replace(".", "-"))) for t in tickers]
    missing = [t for t, c in pairs if c is None]
    pairs = [(t, c) for t, c in pairs if c is not None]
    todo = pairs if fresh else [(t, c) for t, c in pairs if not (out_dir / f"{t}.parquet").exists()]
    print(f"SEC facts for {len(pairs)} tickers ({len(todo)} to extract from {zip_path})...")
    if todo:
        out_dir.mkdir(parents=True, exist_ok=True)
        n_jobs = N_JOBS if n_jobs is None else n_jobs
        if n_jobs == 1 or len(todo) < 50:
            _sec_extract(todo, zip_path, out_dir)
        else:
            from joblib import Parallel, delayed

            k = 8  # chunks, each opening the zip once
            Parallel(n_jobs=min(k, os.cpu_count() or 1) if n_jobs == -1 else n_jobs)(
                delayed(_sec_extract)(todo[i::k], zip_path, out_dir) for i in range(k))
        marker.write_text(source)
    out = {}
    for t, _ in pairs:
        df = pd.read_parquet(out_dir / f"{t}.parquet")
        if not df.empty:
            for col in ("start", "end", "filed"):
                df[col] = pd.to_datetime(df[col])
            out[t] = df
    if missing:
        print(f"  no SEC CIK for {len(missing)}: {', '.join(missing[:_MAX_LISTED])}")
    print(f"  {len(out)} tickers with facts")
    return out


_MAX_LISTED = 25  # tickers printed per issue; the rest are counted


def data_quality_report(prices: dict[str, pd.DataFrame], fundamentals: dict[str, pd.DataFrame]) -> list[str]:
    """Surface (not fix) obvious collection problems. A "no price history"
    ticker has so far always meant a delisting/merger or a ticker rename —
    check that before assuming a bug (universe.py docstring)."""
    print("\n--- Data quality report ---")
    flagged: list[str] = []
    by_issue: dict[str, list[str]] = {}
    for ticker in sorted(prices.keys() | fundamentals.keys()):
        price_df, fund_df = prices.get(ticker), fundamentals.get(ticker)
        issues = []
        if price_df is None or price_df.empty:
            issues.append("no price history")
        else:
            if len(price_df) < MIN_PRICE_ROWS:
                issues.append(f"price history under {MIN_PRICE_ROWS} days (delisting stub with the history lost, "
                              "or a new listing)")
            if (price_df["close"] <= 0).any():
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
            for issue in issues:
                by_issue.setdefault(issue, []).append(ticker)
    for issue, names in by_issue.items():
        shown = ", ".join(names[:_MAX_LISTED])
        more = f" ... (+{len(names) - _MAX_LISTED})" if len(names) > _MAX_LISTED else ""
        print(f"  {issue}: {len(names)} — {shown}{more}")
    if not flagged:
        print("  no issues found")
    return flagged
