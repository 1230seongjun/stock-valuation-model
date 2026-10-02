"""
Korean explanations of the per-stock JSON (llm_context) with Claude, and a
code check of every explanation before it is shown (2026-10-01).

    explain_one(ctx, market)          one stock, synchronous (pilot / single page)
    submit_batch(folder, tickers)     every stock through the Message Batches API (50% price)
    collect_batch(folder, batch_id)   fetch, check and save the batch results
    check(explanation, ctx)           the code check (numbers, banned phrases, required mentions)

Saved as <folder>/explanations/<TICKER>.json with the explanation, the check
result and token usage. An explanation that fails the check is saved with
passed=false and must not be shown as is.

The model sees only the stock's JSON (+ the market context); the system
prompt carries the interpretation rules once (cached). Credentials come from
the environment (ANTHROPIC_API_KEY or an `ant auth login` profile) — never
from code or files.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 8000
PROMPT_VERSION = "5"  # 2: loss-maker verdicts; 3: five bands; 4: financial risk

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "한 문장 요약 (라벨과 핵심 이유)"},
        "summary": {"type": "string", "description": "2~3문장: 무엇과 비교해 얼마나 싸거나 비싼지, 왜 그런지"},
        "reasons": {"type": "array", "items": {"type": "string"}, "description": "판정 근거 2~4개, 각 한두 문장"},
        "cautions": {"type": "array", "items": {"type": "string"}, "description": "읽을 때 주의할 점 1~4개"},
        "market_note": {"type": "string", "description": "시장 전체 맥락 한 문장, 없으면 빈 문자열"},
    },
    "required": ["headline", "summary", "reasons", "cautions", "market_note"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """당신은 미국 주식의 밸류에이션 스크리닝 결과를 개인 투자자에게 한국어로 설명합니다.
입력은 한 종목의 JSON(모델 결과)과 시장 전체 맥락 JSON입니다. 설명은 이 데이터만으로 씁니다.

이 도구가 무엇인지:
- 기준일마다 그날의 미국 상장 종목(약 1,300개)을 비교해, 재무 특성(수익성, 성장, 부채, 업종, 규모 등)이 비슷한 회사들이 보통 받는 배수(적정 배수)를 추정합니다.
- 실제 배수가 적정 배수보다 얼마나 높은지·낮은지(괴리)로 같은 날 순위를 매겨 20%씩 다섯 구간으로 나눕니다: 매우 저평가 / 저평가 / 중립 / 고평가 / 매우 고평가.
- 수익률을 예측하는 도구가 아닙니다.

지켜야 할 규칙 (입력 JSON의 interpretation_rules와 같음):
{rules}

쓰는 방법:
- 독자는 비전문가입니다. 존댓말(~예요/~이에요)로, 짧고 구체적으로 씁니다. 전문 용어는 처음 나올 때 짧게 풀어 줍니다(예: "발생액(이익 중 아직 현금으로 들어오지 않은 부분)").
- 숫자는 JSON에 있는 값만, 같은 단위로 씁니다(괴리는 %, 배수는 "배", 금액은 $B). 반올림은 해도 되지만 새 숫자를 계산하거나 추정하지 않습니다.
- 가장 중요한 근거부터 씁니다. 여러 배수가 같은 방향이면 그 점을, 엇갈리면 그 점을 말합니다.
- 설명 요인은 "모델이 적정 배수를 높게/낮게 잡은 이유"로 표현하고 인과로 쓰지 않습니다. denominator_effect=true인 요인은 "배수 계산식 때문에 생기는 효과"라고 밝힙니다.
- verdict.near_label_boundary가 true면 "경계선"에 있다고 반드시 씁니다.
- without_accruals의 라벨이 현재 라벨과 다르면, 이익의 현금 뒷받침(cash_backing)이 판정을 바꿨다고 반드시 설명합니다.
- 적자 기업(verdict.comparison_group='loss_makers')은 PER이 없어 주로 PSR·PBR로 판단했고 전체 종목과 같은 잣대로 순위를 매겼다는 점, 적자가 일회성인지 구조적인지는 이 데이터로 알 수 없다는 점을 씁니다. 라벨이 '판단 보류(적자)'면 비교할 배수가 없었다고 씁니다. 라벨이 '판단 보류(재무 위험)'이거나 loss_and_heavy_debt 플래그가 있으면, 빚이 많아 낮은 가격에 부도·재무 위험이 반영됐을 수 있고 모델은 그 위험을 측정하지 못한다고 반드시 씁니다. 라벨이 '판단 보류(신규 상장)'면 상장 1년 미만이고 배수 하나로만 비교돼 데이터가 부족하다고 씁니다. heavy_debt 플래그가 있으면 빚이 많아 자본 구조가 괴리의 일부를 설명할 수 있다고 cautions에 씁니다. 플래그가 있으면 cautions에 그 내용을 씁니다.
- 매수·매도 권유, 앞으로의 주가 방향, 목표주가, "내재가치" 같은 표현은 쓰지 않습니다.
- market_note에는 시장 전체 맥락이 있을 때 한 문장만 쓰고, 이 종목의 라벨과는 별개라는 점이 드러나게 씁니다."""

# Phrases an explanation must never contain (investment advice / forecasts / intrinsic value)
BANNED = ["매수", "매도", "추천", "목표주가", "목표가", "오를 것", "오를것", "상승할 것", "하락할 것", "떨어질 것",
          "내재가치", "내재 가치", "투자 의견", "투자의견", "사야", "팔아야", "저가 매수"]


def _rules(ctx: dict) -> str:
    return "\n".join(f"- {r}" for r in ctx.get("interpretation_rules", []))


def _payload(ctx: dict, market: dict | None) -> str:
    """The user message: compact JSON without the rules (they are in the system prompt)."""
    stock = {k: v for k, v in ctx.items() if k != "interpretation_rules"}
    market_slim = {k: v for k, v in (market or {}).items() if k != "interpretation_rules"} or None
    return ("종목 JSON:\n" + json.dumps(stock, ensure_ascii=False, separators=(",", ":"))
            + "\n\n시장 전체 맥락 JSON:\n" + json.dumps(market_slim, ensure_ascii=False, separators=(",", ":")))


def request_params(ctx: dict, market: dict | None, model: str = MODEL, effort: str = EFFORT) -> dict:
    """Messages API parameters for one stock (shared by the synchronous call and the batch)."""
    market_rules = "\n".join(f"- (시장 맥락) {r}" for r in (market or {}).get("interpretation_rules", []))
    system = SYSTEM_PROMPT.format(rules=_rules(ctx) + ("\n" + market_rules if market_rules else ""))
    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": _payload(ctx, market)}],
        "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    }


# ---------------------------------------------------------------- the check
_NUM = re.compile(r"(?<![\w.])[-+−]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|(?<![\w.])[-+−]?\d+(?:\.\d+)?")


def _numbers_in(obj) -> list[float]:
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _numbers_in(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(m.replace(",", "").replace("−", "-")) for m in _NUM.findall(obj)]
    return out


def _allowed(ctx: dict, market: dict | None) -> list[float]:
    vals = _numbers_in({k: v for k, v in ctx.items() if k != "interpretation_rules"}) + _numbers_in(market or {})
    extra = []
    for v in vals:
        extra += [abs(v), v * 100, abs(v) * 100]  # fractions written as percent, signs dropped
    return vals + extra


def _matches(x: float, allowed: list[float], text_num: str) -> bool:
    decimals = len(text_num.split(".")[1]) if "." in text_num else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(x - v) <= tol or abs(abs(x) - abs(v)) <= tol for v in allowed)


def check(explanation: dict, ctx: dict, market: dict | None = None) -> dict:
    """{passed, issues}: every number must come from the JSON (rounding allowed;
    small counts, years and ranks like '20%' of the label rule are exempt),
    no banned phrase, and the required mentions."""
    text = " ".join([explanation.get("headline", ""), explanation.get("summary", ""), *explanation.get("reasons", []),
                     *explanation.get("cautions", []), explanation.get("market_note", "")])
    issues = []
    for phrase in BANNED:
        if phrase in text:
            issues.append(f"banned phrase: {phrase}")
    allowed = _allowed(ctx, market) + [20.0, 80.0, 60.0, 100.0]  # label thresholds / rank scale
    for raw in _NUM.findall(text):
        x = float(raw.replace(",", "").replace("−", "-"))
        if abs(x) <= 12 and float(x).is_integer():  # counts: "4개 배수", "3년", "10년"
            continue
        if 1990 <= x <= 2035 and float(x).is_integer():  # years
            continue
        if not _matches(x, allowed, raw.lstrip("+-−")):
            issues.append(f"number not in the data: {raw}")
    v = ctx.get("verdict", {})
    if v.get("near_label_boundary") and "경계" not in text:
        issues.append("missing: near the label boundary")
    alt = ctx.get("without_accruals")
    if alt and alt.get("label") and alt.get("label") != v.get("label") and "현금" not in text:
        issues.append("missing: accruals changed the label (cash backing)")
    if (v.get("label") == "판단 보류(재무 위험)" or any(f["type"] == "loss_and_heavy_debt" for f in ctx.get("flags", [])))             and not ("부도" in text or "재무 위험" in text):
        issues.append("missing: financial risk")
    if v.get("loss_making") and "적자" not in text:
        issues.append("missing: loss-making")
    if any(f["type"] == "fundamental_break" for f in ctx.get("flags", [])) and not ("일회성" in text or "단절" in text):
        issues.append("missing: fundamental-break flag")
    return {"passed": not issues, "issues": issues}


# ---------------------------------------------------------------- calls
def _client():
    import anthropic

    return anthropic.Anthropic()


def _parse(message) -> tuple[dict | None, str | None]:
    if message.stop_reason == "refusal":
        return None, "refusal"
    if message.stop_reason == "max_tokens":
        return None, "max_tokens"
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        return None, "no text"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        return None, "invalid json"


def _record(ctx: dict, market: dict | None, message, model: str) -> dict:
    explanation, error = _parse(message)
    usage = message.usage
    return {
        "ticker": ctx["ticker"], "as_of": ctx["as_of"], "model": getattr(message, "model", model),
        "prompt_version": PROMPT_VERSION, "schema_version": ctx.get("schema_version"),
        "explanation": explanation, "error": error,
        "check": check(explanation, ctx, market) if explanation else {"passed": False, "issues": [error]},
        "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                  "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0},
    }


def _load(folder: Path, ticker: str) -> tuple[dict, dict | None]:
    ctx = json.loads((folder / f"{ticker}.json").read_text(encoding="utf-8"))
    mfile = folder / "market_context.json"
    return ctx, (json.loads(mfile.read_text(encoding="utf-8")) if mfile.exists() else None)


def _save(folder: Path, record: dict) -> None:
    out = folder / "explanations"
    out.mkdir(exist_ok=True)
    (out / f"{record['ticker']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def explain_one(folder: str | Path, ticker: str, model: str = MODEL, effort: str = EFFORT, client=None) -> dict:
    """One stock, synchronously, with the server-side refusal fallback; saved and returned."""
    folder = Path(folder)
    ctx, market = _load(folder, ticker)
    client = client or _client()
    message = client.beta.messages.create(**request_params(ctx, market, model, effort),
                                          betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    record = _record(ctx, market, message, model)
    _save(folder, record)
    return record


def submit_batch(folder: str | Path, tickers: list[str] | None = None, model: str = MODEL, effort: str = EFFORT,
                 client=None) -> str:
    """Every stock (or `tickers`) as one Message Batch (50% price; no fallbacks
    on batches — a refusal is saved as an error). Returns the batch id, also
    written to <folder>/explanations/batch_id.txt."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    folder = Path(folder)
    if tickers is None:
        tickers = [s["ticker"] for s in json.loads((folder / "index.json").read_text(encoding="utf-8"))["stocks"]]
    client = client or _client()
    requests = []
    for t in tickers:
        ctx, market = _load(folder, t)
        requests.append(Request(custom_id=t.replace(".", "_"),
                                params=MessageCreateParamsNonStreaming(**request_params(ctx, market, model, effort))))
    batch = client.messages.batches.create(requests=requests)
    (folder / "explanations").mkdir(exist_ok=True)
    (folder / "explanations" / "batch_id.txt").write_text(batch.id, encoding="utf-8")
    return batch.id


def collect_batch(folder: str | Path, batch_id: str | None = None, model: str = MODEL, client=None) -> dict:
    """Fetch a finished batch, check and save every result. Returns counts
    (or {"status": ...} while it is still running)."""
    folder = Path(folder)
    client = client or _client()
    batch_id = batch_id or (folder / "explanations" / "batch_id.txt").read_text(encoding="utf-8").strip()
    status = client.messages.batches.retrieve(batch_id)
    if status.processing_status != "ended":
        return {"status": status.processing_status}
    by_id = {s["ticker"].replace(".", "_"): s["ticker"] for s in json.loads((folder / "index.json").read_text(encoding="utf-8"))["stocks"]}
    counts = {"passed": 0, "failed_check": 0, "errored": 0}
    for result in client.messages.batches.results(batch_id):  # any order: keyed by custom_id
        ticker = by_id.get(result.custom_id, result.custom_id)
        ctx, market = _load(folder, ticker)
        if result.result.type != "succeeded":
            _save(folder, {"ticker": ticker, "as_of": ctx["as_of"], "explanation": None, "error": result.result.type,
                           "check": {"passed": False, "issues": [result.result.type]}})
            counts["errored"] += 1
            continue
        record = _record(ctx, market, result.result.message, model)
        _save(folder, record)
        counts["passed" if record["check"]["passed"] else "failed_check"] += 1
    return counts
