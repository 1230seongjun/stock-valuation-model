"""
Sanity check for screening.classify_valuation_transition (added 2026-09-23,
alongside the 저평가->고평가 전환 원인 분류 feature).

Hand-built panel with a known cause behind each "저평가 -> 고평가" flip (price
up / eps down / both / price crash / neither), so we can check the function
actually tells them apart instead of just running without errors. This is a
unit test for the classification RULE, not a claim about real market
behavior — same spirit as tests/synthetic_smoke_test.py.

Run: python tests/transition_classification_test.py
(or, in Colab with every .py file uploaded flat into one folder:
!python transition_classification_test.py)
"""
import sys
from pathlib import Path

try:
    _this_dir = Path(__file__).resolve().parent
except NameError:
    _this_dir = Path.cwd()

for _candidate in (_this_dir, _this_dir.parent / "src"):
    if _candidate.exists():
        sys.path.insert(0, str(_candidate))

import pandas as pd

from screening import classify_valuation_transition

# Every ticker below goes composite_score 80 ("저평가") at t1 -> 20 ("고평가")
# at t2, EXCEPT TICKF which stays cheap (80 -> 75) as a "don't flag when
# nothing flipped" control. sector is irrelevant to this function but kept
# for realism / in case a future version wants it.
ROWS = [
    # ticker, as_of,        composite_score, price,  eps
    ("TICKA", "2024-01-01", 80, 100.0, 5.00),
    ("TICKA", "2024-04-01", 20, 135.0, 5.05),   # price +30%, eps ~flat -> 주가 상승형

    ("TICKB", "2024-01-01", 80, 100.0, 5.00),
    ("TICKB", "2024-04-01", 20, 101.0, 3.60),   # price ~flat, eps -28% -> 실적 악화형

    ("TICKC", "2024-01-01", 80, 100.0, 5.00),
    ("TICKC", "2024-04-01", 20, 135.0, 3.60),   # both -> 복합형

    ("TICKD", "2024-01-01", 80, 100.0, 5.00),
    ("TICKD", "2024-04-01", 20, 75.0, 5.05),    # price -25%, eps ~flat -> 가격 급락형

    ("TICKE", "2024-01-01", 80, 100.0, 5.00),
    ("TICKE", "2024-04-01", 20, 102.0, 5.10),   # neither moves enough -> 불분명

    ("TICKF", "2024-01-01", 80, 100.0, 5.00),
    ("TICKF", "2024-04-01", 75, 103.0, 5.10),   # still 저평가 (>=70) -> no flag at all
]

EXPECTED = {
    "TICKA": "주가 상승형",
    "TICKB": "실적 악화형",
    "TICKC": "복합형",
    "TICKD": "가격 급락형",
    "TICKE": "불분명",
}


def main():
    panel = pd.DataFrame(ROWS, columns=["ticker", "as_of", "composite_score", "price", "eps"])
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    panel["sector"] = "Test"

    print("Running classify_valuation_transition on a hand-built panel with known causes...")
    result = classify_valuation_transition(panel)

    flagged = result[result["transition_flag"]].set_index("ticker")
    print(flagged[["as_of", "transition_type", "transition_reason"]].to_string())

    print("\nChecking each ticker got the expected transition_type...")
    for ticker, expected_type in EXPECTED.items():
        actual_type = flagged.loc[ticker, "transition_type"]
        status = "OK" if actual_type == expected_type else "MISMATCH"
        print(f"  {ticker}: expected={expected_type!r} actual={actual_type!r} -> {status}")
        assert actual_type == expected_type, f"{ticker}: expected {expected_type!r}, got {actual_type!r}"

    print("\nChecking TICKF (never crosses into 고평가) was NOT flagged...")
    tickf_flag = result.loc[result["ticker"] == "TICKF", "transition_flag"].any()
    assert not tickf_flag, "TICKF should never be flagged — it never left 저평가 (composite_score stayed >= 70)"
    print("  OK")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
