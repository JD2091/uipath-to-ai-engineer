# Week 2 Review — From one forward pass to a production call: cost, reliability and the prompt (2026-11-08)

Sunday review, no new topics. This week covered 1.10–3.3 (Days 4–9): the end of Module 1, all of Module 2, and the start of Module 3. Do the quiz cold (about 15 min), read the synthesis, then pick a weekend slot for the mini-project (1–2 hours).

## Quiz (10 questions)

1. **(MCQ, 1.10 + 2.3)** A request has a 2,000-token prompt and returns a 1,500-token answer. Users say it feels slow. Which change cuts *total* latency the most?
    a) Halve the prompt  b) Cap and shorten the output  c) Turn on streaming  d) Lower the temperature
2. **(Short, 1.10)** A model has 40 layers, 8 KV heads, head_dim 128, and stores the cache in fp16. (a) How much KV cache does one 32k-token request need? (b) How much would it need with 32 KV heads (plain multi-head attention), and what is the name of the trick that gives 8? (c) Why must the prompt prefix be byte-identical for prefix caching to work?
3. **(Short, 1.11 + 3.1)** Your accounts-payable bot filled in "PO-7781" for an invoice that had no PO number. Give the root cause in one sentence, two prompt changes from 3.1 that make this less likely, and one check in code that catches it when the prompt fails.
4. **(MCQ, 1.12 + 3.3)** Which statement is true?
    a) Chain-of-thought prompting updates the model's weights for the rest of the session  b) On a reasoning model, writing "think step by step" is the main way to get more reasoning  c) Thinking tokens are billed as output tokens and add latency, so use thinking for multi-rule decisions and leave it low or off for simple high-volume routing  d) Manual CoT with answer tags never helps small non-reasoning models
5. **(MCQ, 1.13 + 1.14)** You must extract totals from 50,000 scanned invoices a month, and data must stay in your company's EU cloud tenancy. What is the best first move?
    a) Pick the model at the top of a public leaderboard  b) Send full-resolution page images to the largest model so nothing is missed  c) Write down the constraints (residency, volume, latency, budget), shortlist models across tiers that your tenancy offers, crop or downscale images, and compare them on your own labelled sample  d) Fine-tune an open model before trying any API model
6. **(Short, 1.15)** On a test set with 50 invoices that truly are `auto_approve`, your classifier predicted `auto_approve` 40 times and 36 of those were correct. Work out precision, recall and F1 for `auto_approve`. Which of the two matters more for this class, and why?
7. **(MCQ, 2.1 + 2.2)** In a chat app, turn 15 costs about 8× as much as turn 1 and takes longer before the first token appears. The most likely cause is:
    a) The provider throttles long sessions  b) The app resends the whole history every turn, so input tokens and prefill time grow with each turn  c) Streaming is turned off  d) The temperature is too high
8. **(MCQ, pick all that apply, 2.4 + 2.5 + 2.9)** Which should your code retry automatically with backoff?
    a) 400 invalid request (your schema uses an unsupported keyword)  b) 401 authentication error  c) 429 with a `retry-after` header  d) 529 / 503 overloaded  e) A read timeout on an idempotent request  f) A spend-limit error
    Then in one line: your calls now go through an AI gateway that also retries. What do you change in the SDK, and why?
9. **(Short, 2.6 + 2.7)** On Sonnet 5.5 (input $2.00/M, 5-minute cache write $2.50/M, cache read $0.10/M), you send a 6,000-token system prompt with 1,000 requests in one hour, and traffic is steady so the cache stays warm. (a) What does the system-prompt part cost per hour without caching and with caching? (b) What changes if the same job runs as a batch? (c) A teammate moves the current timestamp to the top of the system prompt. What happens to your bill, and why?
10. **(Short, 2.8 + 3.2)** (a) Your rate limit is 1,200 requests per minute and a call takes about 2.5 s. How many calls in flight saturate the limit, and which other limit might stop you first with long prompts? (b) After you added three few-shot examples, all labelled `manager_approval`, your classifier over-predicts that label. Why, and what are two fixes?

## How the week connects

Last week ended with a single forward pass picking one token. This week followed that token into production. Generation is a loop of decode steps, and the **KV cache** (1.10) keeps it affordable at the price of GPU memory that grows with context; the same "reuse a computed prefix" idea comes back as **prompt caching** (2.6), which is why stable content goes first. Because the loop always produces some fluent token, models **hallucinate** (1.11), and **reasoning models** (1.12) spend extra decode steps to think, which costs output tokens and latency. Models now take images, PDFs and audio as tokens too (1.13), and which one you call is a trade-off you settle with **constraints and measurement** (1.14), using the precision, recall and F1 vocabulary of classical ML (1.15). Module 2 turned that into engineering. Every call is **stateless** (2.1), priced per token and rate-limited (2.2), streamed for perceived speed (2.3), parsed against a **schema** (2.4), protected by **retries with backoff** (2.5), made cheaper by caching (2.6) or **batching** (2.7), run many-at-once with **async** (2.8), and routed through a cloud platform or **gateway** (2.9). Module 3 started on what goes inside the request: **clear rules with reasons** (3.1), **examples** that show the format (3.2) and **room to reason** (3.3). Each fix maps to one stage below, and the weekend project builds all of them into one pipeline.

```
prompt: clear rules (3.1) · examples (3.2) · room to reason (3.3)
   │  stable prefix first ─► prompt-cache hit (2.6), the API form of prefix KV caching (1.10)
   ▼
stateless call, whole history resent (2.1) ─ priced per token, rate-limited (2.2)
   │  via SDK / cloud platform / gateway (2.9) · many in flight with a semaphore (2.8) · or as a batch (2.7)
   ▼
model: text / image / PDF tokens (1.13) · chosen by constraints + evals (1.14) · optional thinking (1.12)
   │  prefill ─► KV cache ─► decode one token at a time (1.10) ─► streamed back (2.3)
   ▼
reply ─► parsed against a schema (2.4) ─► retried on 429 / 5xx / timeouts (2.5)
   │  invented facts checked in code (1.11)
   ▼
scored on a labelled set: precision / recall / F1 (1.15)
```

## Weekend mini-project (1–2 hours): a hardened invoice-triage pipeline

**Goal.** Turn Day 9's prompt ladder into a pipeline you could defend in a design review. It should classify the 12 test invoices concurrently, return validated JSON, retry properly, check the model's arithmetic in code, report precision, recall and F1, and estimate what 1,000 invoices would cost on Claude in three ways. One script, `triage_pipeline.py`, in a new folder of your choice (for example `Week-2 project/`).

**Setup.** Your existing `.env` (`GEMINI_API_KEY`, `LLM_MODEL`, optional `LLM_MODEL_SMALL`) and `requirements.txt`. `pydantic` is already installed as a dependency of `openai`. Free-tier limits are low, so keep `CONCURRENCY` at 2–3; the retry code handles the 429s you will still hit. If your model rejects `response_format`, delete that argument: the Pydantic check still catches bad output.

**Steps.** Core steps 1–6 take about an hour; 7–8 are stretch goals.

1. **Reuse Day 9 (3.1, 3.2).** Copy the data and rules block from your Day 9 script: `LABELS`, `APPROVED`, `TAX`, `subtotal`, `computed_total`, `expected`, `inv`, `TEST`, `EXAMPLES`, `render`, `worked`, `CONTEXT`, `RULES`.
2. **Cache-friendly layout (1.10, 2.6).** Build every request as one stable system prompt (context, rules, worked examples) followed by the invoice as the user message. Log `cached_tokens` from the usage block when your provider reports it.
3. **Schema with reasoning first (2.4, 3.3).** Validate each reply into the `Decision` model. `reasoning` comes before `label`, so the model works through the rules before it commits. On a validation error, make one repair call; if that fails too, send the invoice to `manager_approval` and count it.
4. **Retries and concurrency (2.5, 2.8).** Retry only 429, 5xx/529 and connection or timeout errors, with full-jitter exponential backoff, and keep the client's own `max_retries=0`. Run all 12 invoices under an `asyncio.Semaphore`, then record wall time and per-call seconds.
5. **Hallucination guard (1.11).** Compare the model's `computed_total` with `computed_total(inv)` from code. If they differ by more than 1.00, override the label to `manager_approval` and count it as a caught arithmetic error.
6. **Score it (1.15).** Print the confusion matrix and a classification report. Look hardest at `auto_approve` precision, because a false `auto_approve` pays an invoice nobody checked.
7. **Cost on paper (2.2, 2.6, 2.7).** From the mean input, output and cached tokens, print the cost per 1,000 invoices on Sonnet 5.5 at standard price ($2/$10 per M), as a batch (half price), and as a batch with the system prompt cached ($0.10/M read, halved again by the batch discount). Use Little's law to add the concurrency that would saturate a 4,000 RPM limit at your measured latency (2.8).
8. **Stretch (2.3, 1.12, 1.14, 1.13).** (a) Stream one request and print TTFT next to the total time. (b) Rerun with `LLM_MODEL_SMALL` or a lower reasoning effort and compare accuracy, output tokens and seconds per call. (c) Render one invoice to a PNG with Pillow, send it as an image, and check whether the label still holds.

**Starter code.**

```python
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
```

**Done looks like.** One run prints 12 rows, a confusion matrix and a classification report without crashing, including when you hit a 429. You have seen at least one retry wait in action. Every row has a validated label, and any arithmetic override is flagged. `auto_approve` precision is reported, with a sentence on whether you would let this run without human review. The three cost figures and the Little's-law number are printed or written as comments. Add a short `NOTES.md` with five observations in your own words, for example which Day 9 trap the code check caught that the prompt alone missed. This pipeline is a good base for later modules: tools (Module 6), evals (Module 8) and the RPA + LLM design question (12.5).

## Answers

1. **b.** Output tokens are produced one decode step at a time, so 1,500 output tokens dominate total time; prompt tokens are processed in parallel during prefill. Streaming (c) only improves *perceived* latency (the first words arrive sooner); total time stays about the same. Temperature does not change speed.
2. (a) Per token: 2 × 40 × 8 × 128 × 2 bytes = 163,840 bytes = **160 KiB**. × 32,768 tokens ≈ **5 GiB** for one request. (b) 4× more KV heads → about **20 GiB**. Sharing each K/V head across several query heads is **grouped-query attention (GQA)**. (c) Cached keys and values depend on every earlier token. Change one token near the start and every position after it has different K/V, so nothing after that point can be reused.
3. Cause: the model is trained to produce a plausible continuation, and a filled-in PO looks more like a "normal" invoice than a blank, so it guessed instead of saying "missing". Prompt (3.1): state the rule and the reason ("a missing PO routes to a human; never invent one"), and give an explicit output for absence (`"po": null` or `"(none)"`); asking for extracted fields to be quoted from the source also helps. Code: check that the returned PO appears in the source text or the ERP, and route to a human if it does not.
4. **c.** Thinking costs output tokens and time, so it pays off on multi-step decisions and wastes money on simple routing. (a) Inference never changes weights. (b) Reasoning models are steered by an effort or budget setting, not by that phrase. (d) Manual CoT with answer tags still helps small and non-reasoning models.
5. **c.** Constraints first (residency rules out models your tenancy cannot host), then a shortlist across tiers, then your own labelled sample. Cropping and downscaling matter because image tokens scale with resolution. Leaderboards (a) do not measure your documents, (b) multiplies cost for little gain, and (d) is premature before you have measured a prompt-based baseline.
6. Precision = 36 / 40 = **0.90**. Recall = 36 / 50 = **0.72**. F1 = 2 × 0.90 × 0.72 / (0.90 + 0.72) = **0.80**. **Precision** matters more: the 4 false positives are invoices paid with no human check. Low recall only means some good invoices wait in a human queue, which is slower but safe.
7. **b.** The API is stateless, so every turn resends the system prompt and the whole history. Input tokens grow with each turn, and so does prefill time (TTFT). Fixes: trim or summarise old turns, and cache the stable prefix.
8. **c, d, e.** 429 and overloaded errors are temporary, and a timeout on an idempotent request is safe to repeat. Use exponential backoff with jitter, and honour `retry-after`. 400, 401 and spend-limit errors fail the same way every time, so retrying wastes calls; fix the request, the key or the budget. Gateway: set the SDK's `max_retries=0` (or retry in one layer only). Otherwise the retries multiply (3 × 3 = 9 attempts) and make an overload worse.
9. (a) Without caching: 1,000 × 6,000 = 6M tokens × $2/M = **$12.00 per hour**. With a warm cache: 1 write (6,000 × $2.50/M = $0.015) + 999 reads (5.994M × $0.10/M ≈ $0.60) ≈ **$0.61 per hour**, about 95% less. (b) Batch halves every price: about $6.00 without caching and about $0.31 with it. Results come back within 24 hours instead of seconds, and cache hits in batches are best-effort, so the real figure falls between the two. (c) Every request now has a different first line, so the prefix never matches. Every request becomes a cache miss and the bill goes back to about $12 per hour, with no error to warn you. Put changing content last.
10. (a) Little's law: in flight = throughput × latency = (1,200 / 60) × 2.5 = **50**. With long prompts, the **input-tokens-per-minute** limit usually binds before requests per minute. (b) The examples taught a label prior: the model copies the majority label in the examples, especially the most recent one. Fixes: give each label at least one example, include boundary cases, shuffle the order, and put the decision logic in explicit rules rather than leaving it to the examples. Then re-measure on the held-out set.
