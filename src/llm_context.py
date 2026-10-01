"""
Per-stock JSON for an LLM (or a web page) to explain a screening result
(2026-10-01). The JSON is the contract: it carries every number an explanation
may use, the reasons a multiple was skipped, the drivers with their caveats
(denominator effect, imputed value), the flags, how well the model fit on that
date, and the rules for reading it. An LLM should write only from this file —
it adds no new judgement and changes no label.

Layout (export_date): <out_dir>/<as_of>/
    index.json            one line per stock: label, rank, combined gap
    <TICKER>.json         full context per stock (schema below, SCHEMA_VERSION)
    market_context.json   market_context.market_context() when available

All percentages are in percent (+74.0 = actual 74% above fair); a multiple's
gap is actual / fair - 1. Missing values are null, never NaN.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CHEAP_THRESHOLD,
    EXPENSIVE_THRESHOLD,
    FAIR_VALUE_FEATURE_EXCLUDE_SECTORS,
    FAIR_VALUE_FEATURES,
    FAIR_VALUE_TARGETS,
    IMPLIED_GROWTH_YEARS,
)
from fair_value import FEATURE_LABELS_KO, target_features
from screening import DENOMINATOR_DRIVERS, MIN_DRIVER_EFFECT, multiple_status

SCHEMA_VERSION = "1.1"  # 1.1: cash_backing, without_accruals, near_label_boundary
MAX_DRIVERS = 3  # per direction and multiple
BOUNDARY_POINTS = 3  # cheapness_rank this close to a label threshold -> near_label_boundary
LABEL_EN = {"저평가": "cheap", "중립": "neutral", "고평가": "expensive",
            "판단 보류(적자)": "withheld_loss", "데이터 부족": "insufficient_data"}
FLAGS = [  # (column, reason column, key, Korean title)
    ("fundamental_break_flag", "fundamental_break_reason", "fundamental_break", "최근 12개월 재무 단절"),
    ("report_lag_flag", "report_lag_reason", "price_moved_since_statements", "재무 기준일 이후 주가 급변"),
    ("single_view_flag", "single_view_reason", "single_multiple", "한 가지 배수로만 판단"),
    ("value_trap_flag", "value_trap_reason", "persistent_discount_low_growth", "지속 할인·저성장 경고"),
    ("meme_flag", "meme_reason", "price_volume_anomaly", "단기 가격·거래량 이상"),
    ("transition_flag", "transition_reason", "cheap_to_expensive", "저평가→고평가 전환"),
]
FUNDAMENTALS = [*FAIR_VALUE_FEATURES, "roe_avg_3y", "net_debt_to_capital", "asset_turnover", "fcf_margin",
                "cash_conversion_3y", "eps_volatility_3y", "accruals"]
# fraction: 0.18 = 18%; ratio: a plain multiple (debt / equity = 0.78)
UNITS = {"debt_to_equity": "ratio", "asset_turnover": "ratio", "cash_conversion_3y": "ratio", "eps_volatility_3y": "ratio",
         "volatility_63d": "fraction (annualized)"}
INTERPRETATION_RULES = [
    "Fair multiple = the multiple the market typically gives stocks with these fundamentals on the same "
    "date (sector and size included). It is not intrinsic value; 저평가/고평가 mean below/above that "
    "baseline, not economically cheap/expensive.",
    "Labels compare stocks on the same date: cheapest 20% = 저평가, most expensive 20% = 고평가.",
    "Not a return forecast. Gaps showed no link to later returns among large caps (0/24 tests after FDR); "
    "never say a stock will rise or fall.",
    "Drivers are the model's contributions to the fair multiple, not causes. denominator_effect=true means "
    "the effect comes from how the multiple is calculated (e.g. PER = PBR / ROE), not from the market's view.",
    "value_missing=true: the figure was missing and filled with the date's median; the effect describes "
    "stocks without that figure, not the company's actual value.",
    "A single multiple's gap within about +-40% is inside the model's typical error (see model_fit); weight "
    "the combined verdict and how many multiples agree.",
    "growth.label_detail compares past 3-year growth with the growth the price requires; it does not say "
    "which one will turn out right.",
    "Flags ask for a second look (one-off items, stale statements, a single multiple); they do not change "
    "the label.",
    "Small caps lean 저평가 and large caps 고평가 because the size discount is left in the gap.",
    "fundamentals: unit 'fraction' means 0.18 = 18%; percentile_same_date ranks the value among all stocks "
    "on this date (100 = highest); used_by_model=false means the model ignores it for this sector.",
    "cash_backing.cash_to_earnings < 1 means part of the last 12 months' net income has not come in as "
    "operating cash yet (accruals > 0). Possible causes include receivables or inventory building up and "
    "non-cash gains (e.g. revaluing investments); this file cannot tell which. Compare with the value a "
    "year earlier for the trend. The model lowers the fair PER for high accruals (Financials excepted).",
    "without_accruals shows the verdict the model gives without the accruals feature: when the label "
    "differs, say that the cash backing of earnings is what moved it.",
    "near_label_boundary=true: the stock sits within a few rank points of the 저평가/고평가 cut-off, so a "
    "small change in price or fundamentals can change the label; describe it as borderline.",
    "Use only numbers present in this file; say 'not available' instead of estimating missing ones.",
]


def _num(v, digits: int = 2):
    """JSON-safe number: None for NaN / None / inf, rounded float otherwise."""
    if v is None or isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else round(f, digits)


def _x100(v) -> float | None:
    """A fraction (0.07) as percent (7.0); None when missing. Not `v or nan`: 0 is a value."""
    return _num(float(v) * 100, 1) if v is not None and pd.notna(v) else None


def _pct(log_gap) -> float | None:
    return _num(np.expm1(log_gap) * 100, 1) if pd.notna(log_gap) else None


def _drivers(row: pd.Series, key: str) -> tuple[list[dict], list[dict]]:
    drivers = [*target_features(FAIR_VALUE_TARGETS[key], FAIR_VALUE_FEATURES), "sector"]
    effects = {d: row.get(f"{key}_contrib_{d}", np.nan) for d in drivers}
    effects = {d: v for d, v in effects.items() if pd.notna(v) and abs(v) >= MIN_DRIVER_EFFECT}

    def item(d: str) -> dict:
        return {"feature": d, "label": FEATURE_LABELS_KO.get(d, d), "effect_pct": _pct(effects[d]),
                "denominator_effect": d in DENOMINATOR_DRIVERS.get(key, ()),
                "value_missing": d != "sector" and pd.isna(row.get(d))}

    ups = sorted((d for d in effects if effects[d] > 0), key=lambda d: -effects[d])[:MAX_DRIVERS]
    downs = sorted((d for d in effects if effects[d] < 0), key=lambda d: effects[d])[:MAX_DRIVERS]
    return [item(d) for d in ups], [item(d) for d in downs]


def model_fit(rows: pd.DataFrame) -> dict:
    """How well each multiple's fair value fit on this date (out of fold):
    R^2 and the median / 90th-percentile |actual / fair - 1| in percent."""
    out = {}
    for key, spec in FAIR_VALUE_TARGETS.items():
        gap = rows.get(f"{key}_gap")
        if gap is None or gap.notna().sum() < 10:
            continue
        ok = gap.notna()
        y = np.log(rows.loc[ok, spec["column"]].astype(float))
        err = np.expm1(gap[ok].abs()) * 100
        out[spec["label"]] = {"r2": _num(1 - (gap[ok] ** 2).sum() / ((y - y.mean()) ** 2).sum(), 3),
                              "median_abs_error_pct": _num(err.median(), 1), "p90_abs_error_pct": _num(err.quantile(.9), 1),
                              "n_stocks": int(ok.sum()), "in_verdict": spec.get("in_verdict", True)}
    return out


def _cash_backing(row: pd.Series, year_ago: pd.Series | None, ranks: pd.DataFrame) -> dict | None:
    ni, cfo = row.get("sec_net_income_ttm"), row.get("sec_operating_cash_flow_ttm")
    if pd.isna(ni) or pd.isna(cfo):
        return None
    ratio = lambda n, c: _num(c / n, 3) if pd.notna(n) and pd.notna(c) and n > 0 else None
    contrib = row.get("pe_contrib_accruals")
    return {
        "net_income_ttm_usd_bn": _num(ni / 1e9, 2), "operating_cash_flow_ttm_usd_bn": _num(cfo / 1e9, 2),
        "cash_to_earnings": ratio(ni, cfo),
        "cash_to_earnings_1y_ago": ratio(year_ago.get("sec_net_income_ttm"), year_ago.get("sec_operating_cash_flow_ttm"))
        if year_ago is not None else None,
        "accruals": _num(row.get("accruals"), 4),
        "accruals_percentile_same_date": _num(ranks.at[row.name, "accruals"] * 100, 0) if "accruals" in ranks.columns else None,
        "effect_on_fair_pe_pct": _pct(contrib),
        "used_by_model": row.get("sector") not in FAIR_VALUE_FEATURE_EXCLUDE_SECTORS.get("accruals", ()),
        "source": "SEC XBRL filings, same fiscal period as the other fundamentals",
    }


def _without_accruals(alt: pd.Series | None) -> dict | None:
    if alt is None:
        return None
    return {"label": alt.get("valuation_label"), "cheapness_rank": _num(alt.get("cheapness_rank"), 0),
            "combined_gap_pct": _pct(alt.get("valuation_gap")), "pe_fair": _num(alt.get("fair_pe")),
            "pe_gap_pct": _pct(alt.get("pe_gap"))}


def stock_context(row: pd.Series, ranks: pd.DataFrame, fit: dict, market_file: str | None,
                  year_ago: pd.Series | None = None, alt: pd.Series | None = None) -> dict:
    """One stock's JSON. ranks: percentile (0-1) of FUNDAMENTALS among the
    stocks of the same date, indexed like the row (fundamental_ranks);
    year_ago: the same stock about a year earlier; alt: its row in the fit
    without accruals (labels added)."""
    multiples = []
    for key, spec in FAIR_VALUE_TARGETS.items():
        if spec["column"] not in row.index:
            continue
        status, reason = multiple_status(row, key)
        m = {"name": spec["label"], "key": key, "in_verdict": spec.get("in_verdict", True), "status": status,
             "actual": _num(row.get(spec["column"])), "fair": None, "gap_pct": None, "drivers_up": [], "drivers_down": []}
        if status == "evaluated":
            m["fair"] = _num(row.get(f"fair_{key}"))
            m["gap_pct"] = _pct(row.get(f"{key}_gap"))
            m["drivers_up"], m["drivers_down"] = _drivers(row, key)
        else:
            m["status_reason"] = reason
        multiples.append(m)
    n = int(row.get("n_gaps") or 0)
    agree = row.get("gap_agreement")
    detail = row.get("label_detail") if isinstance(row.get("label_detail"), str) else ""
    fundamentals = {}
    for f in ranks.columns:
        fundamentals[f] = {"label": FEATURE_LABELS_KO.get(f, f), "value": _num(row.get(f), 4),
                           "unit": UNITS.get(f, "fraction"),
                           "percentile_same_date": _num(ranks.at[row.name, f] * 100, 0),
                           "used_by_model": row.get("sector") not in FAIR_VALUE_FEATURE_EXCLUDE_SECTORS.get(f, ())}
    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": row["ticker"],
        "as_of": str(pd.Timestamp(row["as_of"]).date()),
        "sector": row.get("sector"), "industry": row.get("industry"), "size_group": row.get("size_group"),
        "verdict": {
            "label": row.get("valuation_label"), "label_en": LABEL_EN.get(row.get("valuation_label")),
            "label_detail": detail or None, "view": row.get("valuation_view"),
            "cheapness_rank": _num(row.get("cheapness_rank"), 0),
            "cheapness_rank_note": "percentile among stocks on this date, 100 = cheapest",
            "combined_gap_pct": _pct(row.get("valuation_gap")),
            "basis": [b for b in str(row.get("valuation_basis") or "").split("+") if b],
            "multiples_used": n, "multiples_agreeing": int(round(agree * n)) if pd.notna(agree) and n else None,
            "loss_making": bool(row.get("loss_flag")),
            "near_label_boundary": bool(pd.notna(row.get("cheapness_rank")) and min(
                abs(row["cheapness_rank"] - CHEAP_THRESHOLD), abs(row["cheapness_rank"] - EXPENSIVE_THRESHOLD)) <= BOUNDARY_POINTS),
            "label_thresholds": {"저평가": f"rank >= {CHEAP_THRESHOLD:.0f}", "고평가": f"rank <= {EXPENSIVE_THRESHOLD:.0f}"},
        },
        "without_accruals": _without_accruals(alt),
        "multiples": multiples,
        "growth": {
            "label_detail": detail or None,
            "priced_in_excess_growth_pct": _x100(row.get("implied_excess_growth")),
            "realized_excess_growth_pct": _x100(row.get("realized_excess_growth")),
            "earnings_cagr_3y_pct": _x100(row.get("earnings_cagr_3y")),
            "median_stock_earnings_cagr_3y_pct": _x100(row.get("earnings_cagr_3y_median")),
            "horizon_years": IMPLIED_GROWTH_YEARS,
            "note": "excess = relative to the median stock on this date; priced-in = yearly EPS growth the PER "
                    "needs to return to the median PER after the horizon at the same discount rate",
        },
        "cash_backing": _cash_backing(row, year_ago, ranks),
        "fundamentals": fundamentals,
        "flags": [{"type": key, "title": title, "reason": row.get(reason) or None}
                  for col, reason, key, title in FLAGS if bool(row.get(col))],
        "model_fit": fit,
        "market_context_file": market_file,
        "interpretation_rules": INTERPRETATION_RULES,
    }


def fundamental_ranks(rows: pd.DataFrame) -> pd.DataFrame:
    """Percentile (0-1) of each FUNDAMENTALS column among the given rows (NaN stays NaN)."""
    return rows[[f for f in FUNDAMENTALS if f in rows.columns]].astype(float).rank(pct=True)


def without_feature(rows: pd.DataFrame, feature: str = "accruals") -> pd.DataFrame:
    """The same date refitted without one feature, labels added — the fit is
    per date, so the date's rows alone give exactly the full-panel result."""
    from fair_value import add_fair_value
    from screening import add_labels

    fitted, _ = add_fair_value(rows.drop(columns=[feature]), n_jobs=1)
    return add_labels(fitted).set_index("ticker")


def export_date(screened: pd.DataFrame, as_of: pd.Timestamp | str | None, out_dir: str | Path,
                market: dict | None = None) -> Path:
    """Write the layout in the module docstring for one as_of (default: the
    latest) from a screening.screen() result. Returns the date's folder."""
    target = pd.Timestamp(as_of) if as_of is not None else screened["as_of"].max()
    rows = screened[screened["as_of"] == target]
    alt = without_feature(rows) if "accruals" in rows.columns and rows["accruals"].notna().any() else None
    # about a year earlier: the snapshot 330-400 days before (quarterly dates + a mid-quarter "today")
    prior_dates = [d for d in screened["as_of"].unique() if 330 <= (target - d).days <= 400]
    prior = screened[screened["as_of"] == max(prior_dates)].set_index("ticker") if prior_dates else None
    if rows.empty:
        raise ValueError(f"no rows for as_of={target.date()}")
    folder = Path(out_dir) / str(target.date())
    folder.mkdir(parents=True, exist_ok=True)
    fit = model_fit(rows)
    ranks = fundamental_ranks(rows)
    market_file = None
    if market is not None:
        (folder / "market_context.json").write_text(json.dumps(market, ensure_ascii=False, indent=2), encoding="utf-8")
        market_file = "market_context.json"
    index = []
    for _, row in rows.iterrows():
        t = row["ticker"]
        ctx = stock_context(row, ranks, fit, market_file,
                            year_ago=prior.loc[t] if prior is not None and t in prior.index else None,
                            alt=alt.loc[t] if alt is not None and t in alt.index else None)
        (folder / f"{row['ticker']}.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append({"ticker": ctx["ticker"], "sector": ctx["sector"], "size_group": ctx["size_group"],
                      "label": ctx["verdict"]["label"], "label_detail": ctx["verdict"]["label_detail"],
                      "cheapness_rank": ctx["verdict"]["cheapness_rank"], "combined_gap_pct": ctx["verdict"]["combined_gap_pct"],
                      "flags": [f["type"] for f in ctx["flags"]]})
    index.sort(key=lambda r: -(r["cheapness_rank"] if r["cheapness_rank"] is not None else -1))
    (folder / "index.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "as_of": str(target.date()),
                                                   "n_stocks": len(index), "model_fit": fit, "market_context_file": market_file,
                                                   "stocks": index}, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder
