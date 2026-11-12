"""Week 2 mini-project: a hardened invoice-triage pipeline (topics 1.10-3.3)."""
import asyncio, json, os, random, time
from typing import Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIStatusError, APIConnectionError
from pydantic import BaseModel, ValidationError
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.getenv("LLM_MODEL", "")
CONCURRENCY = int(os.getenv("CONCURRENCY", "3"))   # free tier: keep it small

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=0, timeout=60)  # we own retries (2.5)

# ---- Step 1: paste from your Day 9 script ----
# LABELS, APPROVED, TAX, subtotal, computed_total, expected, inv, TEST, EXAMPLES, render, worked, CONTEXT, RULES

class Decision(BaseModel):
    reasoning: str          # first: the model reasons before it commits (3.3)
    computed_total: float   # the model's own arithmetic, checked in code (1.11)
    label: Literal["auto_approve", "manager_approval", "director_approval", "reject"]

def build_messages(invoice):
    """Stable prefix first, the changing invoice last (1.10, 2.6)."""
    shots = "\n\n".join(
        f"<example>\n{render(e)}\n<answer>"
        + json.dumps({"reasoning": worked(e), "computed_total": computed_total(e), "label": expected(e)})
        + "</answer>\n</example>"
        for e in EXAMPLES)
    system = (f"{CONTEXT}\n\n{RULES}\n\n"
              "Reply with ONLY a JSON object with the keys reasoning, computed_total and label, "
              "in that order.\n\n"
              f"<examples>\n{shots}\n</examples>")
    return [{"role": "system", "content": system},
            {"role": "user", "content": render(invoice)}]

def strip_fences(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    return t

RETRY_STATUS = {429, 500, 502, 503, 504, 529}

async def call_with_retry(messages, max_attempts=5, base=1.0, cap=30.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.chat.completions.create(
                model=MODEL, messages=messages, response_format={"type": "json_object"})
        except APIStatusError as e:
            if e.status_code not in RETRY_STATUS or attempt == max_attempts:
                raise          # 400 / 401 / spend limit: retrying cannot help
            # TODO: if e.response.headers has "retry-after", wait at least that long
        except APIConnectionError:   # includes timeouts
            if attempt == max_attempts:
                raise
        await asyncio.sleep(random.uniform(0, min(cap, base * 2 ** attempt)))  # full jitter

async def triage(invoice, sem, log):
    async with sem:
        msgs = build_messages(invoice)
        t0 = time.perf_counter()
        resp = await call_with_retry(msgs)
        text = resp.choices[0].message.content or ""
        try:
            d = Decision.model_validate_json(strip_fences(text))
        except ValidationError as e:
            # one repair call (2.4): show the model its output and the error
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": f"Invalid: {e.errors()[:2]}. Reply with corrected JSON only."}]
            resp = await call_with_retry(msgs)
            d = Decision.model_validate_json(strip_fences(resp.choices[0].message.content or ""))
            # TODO: on a second failure, use label "manager_approval" and count it
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        pred = d.label
        # TODO (1.11): if abs(d.computed_total - computed_total(invoice)) > 1.00,
        #   set pred = "manager_approval" and mark the row as a caught arithmetic error
        log.append({"id": invoice["id"], "pred": pred, "true": expected(invoice),
                    "in": u.prompt_tokens, "out": u.completion_tokens, "cached": cached,
                    "secs": round(time.perf_counter() - t0, 2)})

async def main():
    sem, log = asyncio.Semaphore(CONCURRENCY), []
    t0 = time.perf_counter()
    await asyncio.gather(*(triage(i, sem, log) for i in TEST))
    print(f"wall time {time.perf_counter() - t0:.1f}s at concurrency {CONCURRENCY}")
    for r in sorted(log, key=lambda r: r["id"]):
        print(r)
    y_true, y_pred = [r["true"] for r in log], [r["pred"] for r in log]
    print(confusion_matrix(y_true, y_pred, labels=LABELS))
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))
    # TODO (2.2, 2.6, 2.7): cost per 1,000 invoices on Sonnet 5.5: standard, batch, batch + cached prefix
    # TODO (2.8): Little's law: in flight = (4000 / 60) * mean secs per call

if __name__ == "__main__":
    asyncio.run(main())
