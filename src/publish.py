"""
Scheduled update for a web/app front end. Everything is computed here in one
batch; the site only reads the files written here, nothing is fitted per
request (main.publish runs the whole job).

    <out_dir>/latest.json            what the site opens first: date, counts, paths
    <out_dir>/<date>/index.json      every stock in one list (llm_context layout)
    <out_dir>/<date>/<TICKER>.json   one stock
    <out_dir>/<date>/market_context.json

A date folder is written under a temporary name and renamed when complete,
and latest.json is replaced last, so a reader never sees a half-written day.
"""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from features import add_percentile_scores, build_raw_panel
from llm_context import SCHEMA_VERSION, export_date


def update_panel(panel: pd.DataFrame, dates: list[pd.Timestamp], tickers: list[str], **build_kwargs) -> tuple[pd.DataFrame, list]:
    """The saved panel restricted to `dates`, plus rows for the dates it lacks
    (today, a quarter start since the last run). Existing rows are reused as
    they are, so their forward-return labels keep the values of the run that
    made them (`build` refreshes those). Rows keep build_raw_panel's order
    (ticker, then date). Returns (panel, dates added)."""
    kept = panel[panel["as_of"].isin(dates)]
    have = set(kept["as_of"])
    missing = [d for d in dates if d not in have]
    if not missing:
        return kept.reset_index(drop=True), []
    new = add_percentile_scores(build_raw_panel(tickers, as_of_dates=missing, **build_kwargs))
    order = {t: i for i, t in enumerate(dict.fromkeys([*panel["ticker"], *tickers]))}
    out = pd.concat([kept, new], ignore_index=True)
    out = out.sort_values(["as_of"], kind="stable")
    out = out.iloc[out["ticker"].map(order).argsort(kind="stable")].reset_index(drop=True)
    return out, missing


def _replace_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    src.rename(dst)


def write_site(screened: pd.DataFrame, out_dir: str | Path, market: dict | None = None, keep: int = 14) -> Path:
    """The latest as_of of a screening.screen() result as site files (module
    docstring). Keeps the newest `keep` date folders. Returns latest.json."""
    out_dir = Path(out_dir)
    as_of = screened["as_of"].max()
    date = str(as_of.date())
    staging = out_dir / ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    export_date(screened, as_of, staging, market)
    out_dir.mkdir(parents=True, exist_ok=True)
    _replace_dir(staging / date, out_dir / date)
    shutil.rmtree(staging, ignore_errors=True)

    index = json.loads((out_dir / date / "index.json").read_text(encoding="utf-8"))
    counts = Counter(s["label"] for s in index["stocks"])
    latest = {
        "schema_version": SCHEMA_VERSION,
        "as_of": date,
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_stocks": index["n_stocks"],
        "label_counts": dict(counts.most_common()),
        "index": f"{date}/index.json",
        "market_context": f"{date}/market_context.json" if market is not None else None,
        "dates": [],
    }
    dates = sorted(p.name for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    for old in dates[:-keep] if keep > 0 else []:
        if old != date:
            shutil.rmtree(out_dir / old)
    latest["dates"] = sorted(p.name for p in out_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    tmp = out_dir / "latest.json.tmp"
    tmp.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "latest.json")
    return out_dir / "latest.json"
