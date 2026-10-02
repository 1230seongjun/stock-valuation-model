"""
Market-wide valuation context (2026-10-01): where the whole S&P 500 sits
against the macro environment. A DESCRIPTIVE number for the report and for a
later LLM explanation — never a model input (it is the same for every stock on
a date, so it cannot change any fair multiple or label) and never a forecast.

Model (validated before it was added, see config / README experiment log):
    log CAPE_m = a + b1 * 10y yield_m + b2 * CPI inflation_(m-1) + b3 * unemployment_(m-1)
fitted on 1990-01..2015-12 and kept fixed. Out of sample (2016-01..2026-08)
R^2 +0.30 against the training mean, both expected signs (yield -, inflation
-); monthly-change R^2 only 0.01 — it describes the valuation REGIME the macro
backdrop is consistent with, not month-to-month moves. VIX and the credit
spread were left out on purpose: they move with stock prices in the same month
(full model out of sample +0.54, monthly-change R^2 0.61 — co-movement, not
explanation).

Point in time: a snapshot dated as_of uses the last full month before it;
yields are that month's mean, CPI and unemployment the month before (they are
published with a lag). Data: Shiller's monthly CAPE (shillerdata.com) and FRED,
cached in <cache_dir>/macro/ and re-fetched after MACRO_REFRESH_DAYS.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

SHILLER_URL = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
               "70fec4f5-727f-4e53-b5f1-179af109c5fa/ie_data.xls")  # linked from shillerdata.com
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
# FRED let requests' default User-Agent (and curl's) through at once but timed
# out on browser-like and custom ones (2026-10-01) — so no custom header.
FRED_SERIES = {"yield10y": "DGS10", "cpi": "CPIAUCNS", "unemployment": "UNRATE"}
MACRO_REFRESH_DAYS = 7
FIT_START, FIT_END = pd.Period("1990-01", "M"), pd.Period("2015-12", "M")
MIN_FIT_MONTHS = 120
INPUTS = ["yield10y", "inflation", "unemployment"]
# measured once on 2026-10-01 (data to 2026-08), for the reader of the output
VALIDATION = {"out_of_sample_period": "2016-01..2026-08", "out_of_sample_r2": 0.30,
              "in_sample_r2": 0.55, "monthly_change_r2": 0.01}
INTERPRETATION_RULES = [
    "Descriptive: where the S&P 500's CAPE sits against the level that 1990-2015 relationships with "
    "rates, inflation and unemployment imply. Not a forecast of market returns or of their timing.",
    "A high gap can persist for years (2021 averaged +0.34 log).",
    "CAPE has drifted up structurally (index composition, buybacks, accounting), so part of a positive "
    "gap may be structural rather than temporary.",
    "Stock labels (저평가/중립/고평가) compare stocks on the same date and do not depend on this value; "
    "a stock can be cheap relative to its peers while the whole market is expensive.",
    "Effective sample is about 2-3 business cycles; treat the numbers as approximate.",
]


def _fetch(url: str) -> bytes:
    import requests

    for wait in (0, 5, 20):
        time.sleep(wait)
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
        except requests.RequestException:
            pass
    raise RuntimeError(f"download failed: {url}")


def _cached(path: Path, url: str, force_refresh: bool) -> Path:
    """path, downloaded if missing / older than MACRO_REFRESH_DAYS. A failed
    refresh keeps the old file (stale context beats none)."""
    fresh = path.exists() and (time.time() - path.stat().st_mtime) < MACRO_REFRESH_DAYS * 86400
    if fresh and not force_refresh:
        return path
    try:
        content = _fetch(url)
    except RuntimeError:
        if path.exists():
            print(f"  {path.name}: refresh failed, using the cached copy")
            return path
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def read_shiller_cape(path: str | Path) -> pd.Series:
    """Monthly CAPE from Shiller's ie_data.xls, indexed by Period('M').
    Dates are written as 2026.09 / 2026.1 (= October)."""
    return parse_shiller(pd.read_excel(path, sheet_name="Data", header=7))


def parse_shiller(d: pd.DataFrame) -> pd.Series:
    """read_shiller_cape's parsing of the sheet (Date / CAPE columns)."""
    date = pd.to_numeric(d["Date"], errors="coerce")
    cape = pd.to_numeric(d["CAPE"], errors="coerce")
    ok = date.notna() & cape.notna()
    year = date[ok].astype(int)
    month = ((date[ok] - year) * 100).round().astype(int)
    return pd.Series(cape[ok].to_numpy(), index=pd.PeriodIndex.from_fields(year=year, month=month, freq="M"), name="cape")


def read_fred(path: str | Path) -> pd.Series:
    d = pd.read_csv(path, na_values=".")
    d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"])
    return d.dropna().set_index("date")["value"].sort_index()


def load_market_data(cache_dir: str | Path, force_refresh: bool = False) -> pd.DataFrame:
    """Shiller CAPE + FRED series (cached) -> monthly_inputs."""
    macro = Path(cache_dir) / "macro"
    cape = read_shiller_cape(_cached(macro / "ie_data.xls", SHILLER_URL, force_refresh))
    fred = {k: read_fred(_cached(macro / f"{code}.csv", FRED_URL.format(code), force_refresh))
            for k, code in FRED_SERIES.items()}
    return monthly_inputs(cape, fred)


def monthly_inputs(cape: pd.Series, fred: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per month m: log_cape_m, the 10-year yield's mean over m, CPI
    inflation (log YoY, %) and unemployment of m-1 — what was published by the
    end of m."""
    to_month = lambda s: s.groupby(pd.PeriodIndex(s.index, freq="M")).mean()
    cpi, une = to_month(fred["cpi"]), to_month(fred["unemployment"])
    return pd.DataFrame({
        "log_cape": np.log(cape),
        "yield10y": to_month(fred["yield10y"]),
        "inflation": (np.log(cpi / cpi.shift(12)) * 100).shift(1),
        "unemployment": une.shift(1),
    }).sort_index()


# Latest usable month: the last full month before as_of or the one before it.
MAX_LAG_MONTHS = 2


def market_context(data: pd.DataFrame, as_of: str | pd.Timestamp) -> dict | None:
    """The context for a snapshot dated as_of (JSON-ready), from monthly_inputs
    rows up to the last full month before as_of only. Early in a month that
    month may not be published yet, so the latest month within
    MAX_LAG_MONTHS is used ("month" says which). None if there is none or
    MIN_FIT_MONTHS of fitting data are missing."""
    import statsmodels.api as sm

    last_full = pd.Period(pd.Timestamp(as_of), "M") - 1
    known = data[data.index <= last_full].dropna()
    if known.empty or (last_full - known.index[-1]).n >= MAX_LAG_MONTHS:
        return None
    month = known.index[-1]
    fit_rows = known[(known.index >= FIT_START) & (known.index <= FIT_END)]
    if len(fit_rows) < MIN_FIT_MONTHS:
        return None
    fit = sm.OLS(fit_rows["log_cape"], sm.add_constant(fit_rows[INPUTS])).fit()
    implied = fit.predict(sm.add_constant(known[INPUTS], has_constant="add"))
    gap = known["log_cape"] - implied
    since = gap[gap.index >= FIT_START]
    recent = gap[gap.index >= FIT_END + 1]
    now = known.iloc[-1]
    return {
        "as_of": str(pd.Timestamp(as_of).date()),
        "month": str(month),
        "index": "S&P 500",
        "cape": round(float(np.exp(now["log_cape"])), 2),
        "macro_implied_cape": round(float(np.exp(implied.iloc[-1])), 2),
        "gap_log": round(float(gap.iloc[-1]), 3),
        "gap_pct": round(float(np.expm1(gap.iloc[-1]) * 100), 1),
        "gap_percentile_since_1990": round(float((since <= gap.iloc[-1]).mean() * 100)),
        # 5-95% of the monthly gaps, not min/max: one month (2020-04, unemployment 14.7%) reached +1.0
        "gap_p5_p95_since_2016": [round(float(recent.quantile(.05)), 3), round(float(recent.quantile(.95)), 3)] if len(recent) else None,
        "inputs": {
            "yield10y_pct": {"value": round(float(now["yield10y"]), 2), "month": str(month)},
            "inflation_yoy_pct": {"value": round(float(now["inflation"]), 2), "month": str(month - 1)},
            "unemployment_pct": {"value": round(float(now["unemployment"]), 2), "month": str(month - 1)},
        },
        "model": {
            "target": "log Shiller CAPE (S&P 500, monthly)",
            "inputs": INPUTS,
            "fit_period": f"{fit_rows.index[0]}..{fit_rows.index[-1]}",
            "coefficients": {k: round(float(v), 4) for k, v in fit.params.items()},
            **VALIDATION,
        },
        "interpretation_rules": INTERPRETATION_RULES,
    }


def context_lines(ctx: dict) -> list[str]:
    """The numbers of a context as a short neutral block for the console."""
    i = ctx["inputs"]
    lo, hi = ctx["gap_p5_p95_since_2016"] or (np.nan, np.nan)
    return [
        f"S&P 500 CAPE {ctx['cape']:.1f} | 거시 기준 CAPE {ctx['macro_implied_cape']:.1f} | 차이 {ctx['gap_pct']:+.0f}% "
        f"(log {ctx['gap_log']:+.2f}) | 1990년 이후 백분위 {ctx['gap_percentile_since_1990']} | 2016년 이후 5~95% 범위 {lo:+.2f}~{hi:+.2f}",
        f"입력 ({ctx['month']} 기준): 10년 금리 {i['yield10y_pct']['value']:.2f}%, "
        f"물가상승률 {i['inflation_yoy_pct']['value']:.2f}% ({i['inflation_yoy_pct']['month']}), "
        f"실업률 {i['unemployment_pct']['value']:.1f}% ({i['unemployment_pct']['month']})",
    ]
