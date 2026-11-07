# Day 8 — Scaling out: batch jobs, async concurrency, and where your model runs (2026-11-06)

Day 7 made one call dependable. Today is about running thousands of them. Batch processing trades latency for a 50% discount when nobody is waiting (2.7). Async concurrency gets many live calls in flight without breaking rate limits (2.8). Cloud-hosted models and AI gateways decide where those calls go and who controls them (2.9). Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. Structured output guarantees the JSON matches your schema. Name two things it does *not* guarantee.
2. Why is "exactly one retry layer" a rule, and what happens during an outage if you break it?
3. A request puts `Request time: 13:05` at the top of a 6,000-token system prompt. What happens to prompt caching, and what is the fix?

## 2.7 Batch processing

**The idea.** Many LLM jobs have no user waiting: classify last month's 50,000 tickets, extract fields from an invoice archive, run an eval set every night, generate embeddings for a document store. Running those through the normal API means managing rate limits, retries and concurrency yourself, and paying full price for speed you don't need. A batch API lets you hand the provider the whole job at once. The provider runs it when it has spare capacity, usually within an hour and at most within 24 hours, and charges you half price. You give up latency; you get cost savings and far less plumbing.

**How it works.** The flow is the same on every provider: submit, poll, collect, join.

```
your code                          provider
---------                          --------
build N requests, each with a custom_id
   │  create batch (list or JSONL file) ─►  queued
   │                                        in_progress (runs when capacity is free)
   │  poll status every ~60 s  ◄────────    request counts update
   │                                        ended (all done, or 24 h passed)
   │  download results (.jsonl)  ◄──────    one line per request, ANY order
   ▼
join results to inputs by custom_id → retry only the failed ones
```

On Claude (checked 2026-11-06, [batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)):

- `client.messages.batches.create(requests=[{"custom_id": ..., "params": {...normal Messages body...}}])`, then `retrieve(id)` to poll `processing_status` (`in_progress` → `ended`), then `results(id)` to stream the JSONL.
- 50% of standard price on input and output. Sonnet 5.5 drops from $2/$10 to $1/$5 per million tokens. The discount **stacks with prompt caching**.
- Up to 100,000 requests or 256 MB per batch. Most finish in under an hour; anything not done in 24 hours expires. Results stay downloadable for 29 days.
- Each result is `succeeded`, `errored`, `canceled` or `expired`. Only `succeeded` is billed.
- Almost every Messages feature works inside a batch (tools, vision, thinking, structured outputs, mixed request types). `stream: true` does not.
- `custom_id` must be unique and match `^[a-zA-Z0-9_-]{1,64}$`.

OpenAI's Batch API uses the same idea with a file: upload a JSONL file with `purpose="batch"`, create the batch with `completion_window="24h"`, and download the output file when the status is `completed`. Each line looks like `{"custom_id": "...", "method": "POST", "url": "/v1/chat/completions", "body": {...}}`. Gemini offers a batch mode at 50% off with a 24-hour target, and it now accepts this OpenAI-style flow through its compatibility endpoint.

**Gotchas.**

- **Results come back in any order.** Never zip results to inputs by position. Join on `custom_id`, so make it meaningful (`ticket-48213`), not `0, 1, 2`.
- **Partial failure is normal.** A batch "ended" does not mean every request succeeded. Count each result type, retry `errored` (server-side) and `expired` ones in a new batch, and fix and resubmit invalid requests.
- **Validation errors surface late.** A typo in one request's params shows up as `errored` hours later. Test 5–10 requests synchronously before you submit 50,000.
- **Caching in batches is best-effort.** Anthropic quotes 30–98% hit rates. Put identical `cache_control` blocks in every request and consider the 1-hour TTL so the cache outlives gaps in processing.
- **Not for anything a human waits on.** "Usually under an hour" is not an SLA. Chat, agent steps and approvals stay synchronous.
- **Not on every platform.** The Message Batches API is not available on Bedrock, Vertex AI or Foundry (2.9); those clouds have their own batch job services.

**UiPath bridge.** It is the Orchestrator queue pattern from the other side: you load all items with references up front, the platform processes them at its own pace, and you reconcile each item's status (Successful, Failed, Abandoned) by its reference rather than by position.

**Interview angle.**

- *You need to classify 2 million documents with an LLM by next week. How?* Use the provider's batch API: about half the cost and no rate-limit or retry plumbing. Pilot 100 items synchronously to fix the prompt and schema, then submit in chunks within the per-batch limits, with stable prompt prefixes for cache hits. Join results by `custom_id`, resubmit `errored` and `expired` items, and track success rate and cost per chunk.
- *When would you NOT use batch?* When a user or downstream step is waiting, when you need streaming, or when each step depends on the previous answer (an agent loop). Batch is for independent, latency-tolerant work.

## 2.8 Async and concurrency in Python for LLM calls

**The idea.** An LLM call spends almost all its time waiting: a few milliseconds sending the request, then 1–30 seconds while the model generates. Run 100 calls one after another and your program sits idle for 99% of the time. Since the work is waiting on the network (I/O-bound) rather than computing (CPU-bound), you can start many calls and let them wait together. Python's `asyncio` does this on a single thread: when a call hits `await`, the event loop switches to another task that is ready. The hard part is not going faster. It is going fast *within limits*: your rate limit, the provider's capacity and your budget.

**How it works.**

```
sequential (10 calls × 3 s):   [■■■][■■■][■■■][■■■] ...            ≈ 30 s
async, concurrency = 5:        [■■■][■■■]
                               [■■■][■■■]
                               [■■■][■■■]   (5 lanes)               ≈ 6 s
                               [■■■][■■■]
                               [■■■][■■■]
```

The standard pattern is one async client, a semaphore to cap how many calls are in flight, and `gather`:

```python
import asyncio
from anthropic import AsyncAnthropic          # or: from openai import AsyncOpenAI

client = AsyncAnthropic()                     # ONE client, shared: it owns the connection pool
sem = asyncio.Semaphore(8)                    # at most 8 requests in flight

async def classify(text):
    async with sem:                           # wait for a free slot
        r = await client.messages.create(
            model="claude-haiku-5-5", max_tokens=10,
            messages=[{"role": "user", "content": text}])
        return r.content[0].text

async def main(texts):
    return await asyncio.gather(*(classify(t) for t in texts), return_exceptions=True)

results = asyncio.run(main(texts))            # results are in INPUT order
```

Choosing the concurrency number with Little's law: requests in flight = throughput × latency. If your limit is 600 requests per minute (10 per second) and a call takes 3 seconds, about 10 × 3 = 30 calls in flight saturates the limit. More than that only produces 429s. With long prompts, do the same sum in tokens per minute, which usually binds first (Day 6).

**Gotchas.**

- **One blocking call freezes everything.** `time.sleep()`, `requests.get()` or the *sync* SDK client inside an `async def` stops the whole event loop, and your "concurrent" code runs one at a time. Use `await asyncio.sleep()`, the async client, or wrap unavoidable blocking code in `await asyncio.to_thread(fn, ...)`.
- **Unbounded `gather` is a self-inflicted outage.** `gather` over 10,000 calls with no semaphore fires them all at once: a wall of 429s, retries that multiply (Day 7), and possibly exhausted connections. Always cap.
- **One failure can hide the rest.** Plain `gather` raises the first exception and you lose sight of the others. Use `return_exceptions=True` and inspect each result, or `asyncio.TaskGroup` (Python 3.11+) when one failure should cancel the group.
- **Create the client once.** A new client per call means a new connection pool and TLS handshake per call. Share one client for the whole run.
- **Retry sleeps while holding a slot.** If your retry backoff runs inside `async with sem`, a waiting task blocks a lane. That is acceptable for small jobs; for large ones, release the slot before sleeping.
- **Threads are fine too.** For a sync codebase, `ThreadPoolExecutor(max_workers=8)` with the sync client gives similar gains for I/O-bound calls. Async scales to more concurrent calls with less overhead.

**UiPath bridge.** It is the difference between one robot working a queue and several unattended robots on the same queue. Throughput rises with robots, but the target system's limits (licences, sessions, API quotas) cap how many you can usefully add. The semaphore is your robot count.

**Interview angle.**

- *Your script calls an LLM for 5,000 rows and takes 4 hours. Speed it up.* It is I/O-bound, so add concurrency with the async client and a semaphore sized by Little's law against the rate limit (RPM and tokens per minute), with backoff on 429. If nobody needs the results within the hour, use the batch API instead: slower per item but half the price and no limit management.
- *Why use asyncio rather than threads for LLM calls?* Both work because the GIL is released while waiting on the network. Asyncio handles hundreds of concurrent calls on one thread with less memory and makes the concurrency cap explicit. Threads suit existing sync code. Neither helps CPU-bound work, which needs processes.

## 2.9 Cloud-hosted models (Bedrock/Vertex/Azure) and AI gateways

**The idea.** In an enterprise, "call the Anthropic API with a key" is often not allowed. Security wants traffic to stay inside the company's cloud account, legal wants data to stay in a region, procurement wants it billed against the existing cloud contract, and IT wants access governed by the same IAM roles as everything else. So the big clouds host frontier models themselves: Claude is on Amazon Bedrock, Google Cloud Vertex AI and Microsoft Foundry. Separately, once a company uses several models and providers, it puts an **AI gateway** in front of them: one internal endpoint that handles keys, routing, fallbacks, budgets and logging for every LLM call.

**How it works — cloud-hosted.** The request body stays the Messages API; what changes is the client, the auth and the model name. For Claude (checked 2026-11-06 on the Claude platform docs):

| Platform | Python client | Auth | Sonnet 5.5 model name | Notes |
| --- | --- | --- | --- | --- |
| Claude API | `Anthropic()` | API key | `claude-sonnet-5-5` | Newest features first |
| Amazon Bedrock | `AnthropicBedrockMantle(aws_region=...)` | AWS IAM role or short-term bearer token | `anthropic.claude-sonnet-5-5` | Newer models use the `bedrock-mantle` Messages endpoint; some models need access approval |
| Vertex AI | `AnthropicVertex(project_id=..., region="global")` | Google Cloud credentials (`gcloud auth application-default login`) | `claude-sonnet-5-5` | Newest models need the `global` or `us`/`eu` multi-region endpoint |
| Microsoft Foundry | `AnthropicFoundry(resource=...)` | Foundry API key or Entra ID | your deployment name (default `claude-sonnet-5-5`) | Billed through Azure Marketplace |

```
          your app
             │  same Messages body
   ┌─────────┼──────────────┬──────────────────┐
   ▼         ▼              ▼                  ▼
Claude API  Bedrock       Vertex AI        Foundry
(API key)   (IAM, AWS     (GCP IAM,        (Entra ID,
            bill)          GCP bill)        Azure bill)
```

On Bedrock and Vertex, regional or multi-region endpoints (data stays in the US, EU and so on) cost 10% more than the global endpoint for recent models. Anthropic also offers Claude Platform on AWS, an Anthropic-operated option billed through AWS Marketplace with near-same-day features.

**How it works — AI gateways.** A gateway is a reverse proxy that speaks one API (usually OpenAI-compatible) to your apps and many APIs to providers:

```
apps / agents ──► AI gateway ──► Claude API / Bedrock / Vertex / Azure OpenAI / local Ollama
                  ├─ virtual keys per team, real provider keys stay in the gateway
                  ├─ routing: "fast" → small model, "smart" → large model
                  ├─ retries, fallbacks, cooldowns for unhealthy deployments
                  ├─ budgets and rate limits per team, cost tracking
                  └─ logging, PII redaction, guardrails
```

Examples: LiteLLM (open source, self-hosted proxy), Cloudflare AI Gateway, Portkey, Kong AI Gateway, OpenRouter (hosted multi-provider router), and the cloud providers' own API-management gateways. In LiteLLM, for instance, a config file maps a public `model_name` to real deployments and declares `fallbacks`, `num_retries` and cooldowns, plus special fallbacks for context-window overflow and content-policy refusals ([LiteLLM reliability docs](https://docs.litellm.ai/docs/proxy/reliability)).

**Gotchas.**

- **Feature lag and gaps.** Cloud platforms often get new features later, or not at all. On Bedrock, Vertex and Foundry, Claude's Message Batches API, Files API, Models API and several server-side tools are not supported. Check the platform page before you design around a feature.
- **Different names, quotas and headers.** Model IDs differ per platform, quotas come from the cloud (not your Anthropic tier), and Foundry does not return Anthropic's rate-limit headers. Keep model names in config, not code.
- **Model access is not automatic.** Bedrock may require you to request access to a model, and some models are restricted. Ask for access early in a project; approvals take time.
- **A gateway is another hop and another single point of failure.** It adds a little latency and must itself be highly available. It also joins the retry stack: if the gateway retries, set the SDK's `max_retries=0` (Day 7).
- **OpenAI-compatible translation loses features.** Sending Claude traffic through an OpenAI-shaped gateway can drop or mangle provider-specific options (cache breakpoints, thinking settings, citations). Use pass-through routes for those.
- **Hosted routers see your data.** A third-party gateway processes every prompt. That is a data-processing decision for security and legal, not only a developer convenience.

**UiPath bridge.** The AI Trust Layer is an AI gateway: every LLM call from UiPath products passes through one governed layer with policies, PII masking, auditing and bring-your-own-LLM routing to Azure OpenAI or other providers. A standalone gateway is the same idea for your own code.

**Interview angle.**

- *A bank wants Claude but data must stay in the EU and on their AWS account. What do you propose?* Use Claude through Amazon Bedrock with the EU regional endpoint (about 10% premium), IAM roles for access, and private networking. Check the feature list for gaps such as batch and the Files API, and design around them. Put a gateway in front for logging, PII redaction and budgets.
- *Why put an AI gateway in front of LLM providers?* Central control: one place for keys, routing, fallbacks, retries, caching, budgets, observability and guardrails, so apps don't each reinvent them and can switch providers through config. The cost is an extra hop, a component that must be highly available, and possible loss of provider-specific features.

## Hands-on (30–45 min): one job, three ways to run it

**Goal.** Classify 12 product reviews three ways and compare: sequential calls, async calls with a concurrency cap, and a batch file. Measure wall time, peak in-flight calls and output order, then estimate the batch saving on Claude.

**Setup.** Your existing `.env` (`GEMINI_API_KEY`, `LLM_MODEL`) and `requirements.txt`. Nothing new to install: `AsyncOpenAI` ships with `openai`. Free-tier Gemini is enough for Parts A and B (about 30 short calls). Gemini's free-tier requests-per-minute limit is low, so keep concurrency at 3–5. Part C's provider batch may need a paid tier; if submission fails, the script falls back to a local "batch emulator" over the same file. Save as `day08_throughput.py` in your own Day-08 exercise folder. If your model rejects `reasoning_effort`, delete it from `EXTRA`.

```python
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.environ["LLM_MODEL"]
EXTRA = {"reasoning_effort": "low"}

# 2.8: ONE async client for the whole run; our code owns retries (Day 7)
aclient = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=0, timeout=60)

SYSTEM = "Classify the review's sentiment. Reply with exactly one word: positive, negative or mixed."
REVIEWS = [
    "Battery lasts two full days. Best purchase this year.",
    "Stopped charging after a week. Support never replied.",
    "Great screen, but the speakers are tinny.",
    "Arrived early and works perfectly.",
    "Box was crushed and the charger was missing.",
    "Decent value, though setup took an hour.",
    "Absolutely love it, bought a second one for my dad.",
    "Overheats when gaming. Returning it.",
    "Camera is excellent; the app is buggy.",
    "Does what it says. Nothing more, nothing less.",
    "Refund took three weeks and four emails.",
    "Light, fast, quiet. Five stars.",
]


def build_requests():
    """OpenAI batch-file format: one dict per line, joined later by custom_id."""
    return [
        {
            "custom_id": f"review-{i:03d}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "max_tokens": 512,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                **EXTRA,
            },
        }
        for i, text in enumerate(REVIEWS)
    ]


RETRYABLE = (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)


async def call_one(sem, req, stats):
    async with sem:  # 2.8: wait for a free lane
        stats["in_flight"] += 1
        stats["peak"] = max(stats["peak"], stats["in_flight"])
        try:
            for attempt in range(4):
                try:
                    t0 = time.perf_counter()
                    r = await aclient.chat.completions.create(**req["body"])
                    label = (r.choices[0].message.content or "").strip().lower().rstrip(".")
                    return {"custom_id": req["custom_id"], "status": "succeeded", "label": label,
                            "secs": round(time.perf_counter() - t0, 2)}
                except RETRYABLE as e:
                    if attempt == 3:
                        return {"custom_id": req["custom_id"], "status": "errored", "error": type(e).__name__}
                    await asyncio.sleep(random.uniform(0, 2 ** attempt))  # await, never time.sleep
        finally:
            stats["in_flight"] -= 1


async def run_all(requests, concurrency):
    sem = asyncio.Semaphore(concurrency)
    stats = {"in_flight": 0, "peak": 0}
    t0 = time.perf_counter()
    results = await asyncio.gather(*(call_one(sem, r, stats) for r in requests))
    return results, time.perf_counter() - t0, stats["peak"]


def part_a(concurrency=3):
    reqs = build_requests()
    print("== Part A: sequential vs async ==")
    for c in (1, concurrency):
        results, secs, peak = asyncio.run(run_all(reqs, c))
        ok = sum(r["status"] == "succeeded" for r in results)
        print(f"   concurrency={c}: {secs:5.1f}s wall, peak in flight={peak}, {ok}/{len(reqs)} ok")
        time.sleep(20)  # let the free-tier per-minute window recover between runs
    for r in results:
        print("  ", r)


async def toy(blocking):
    async def fake_call(i):
        if blocking:
            time.sleep(1)           # blocks the whole event loop
        else:
            await asyncio.sleep(1)  # yields to other tasks
        return i
    t0 = time.perf_counter()
    await asyncio.gather(*(fake_call(i) for i in range(5)))
    return time.perf_counter() - t0


def part_b():
    print("== Part B: the blocking trap (no API calls) ==")
    print(f"   await asyncio.sleep: {asyncio.run(toy(False)):.1f}s for 5 'calls'")
    print(f"   time.sleep         : {asyncio.run(toy(True)):.1f}s for 5 'calls'")


def part_c():
    reqs = build_requests()
    path = Path("batch_input.jsonl")
    path.write_text("\n".join(json.dumps(r) for r in reqs) + "\n", encoding="utf-8")
    print(f"== Part C: wrote {path} ({len(reqs)} requests) ==")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        f = client.files.create(file=path.open("rb"), purpose="batch")
        batch = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                                      completion_window="24h")
        Path("batch_id.txt").write_text(batch.id)
        print(f"   submitted provider batch {batch.id}, status={batch.status}. Poll later with: d")
    except openai.APIError as e:
        print(f"   provider batch unavailable ({type(e).__name__}): {str(e)[:120]}")
        print("   running the local batch emulator over the same file instead")
        results, secs, _ = asyncio.run(run_all(reqs, 3))
        random.shuffle(results)  # real batch results arrive in ANY order
        Path("batch_output.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")
        print(f"   wrote batch_output.jsonl in {secs:.1f}s (shuffled). Join it with: e")


def part_d():
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    batch = client.batches.retrieve(Path("batch_id.txt").read_text().strip())
    print(f"== Part D: batch {batch.id} status={batch.status} counts={batch.request_counts} ==")
    if batch.status == "completed" and batch.output_file_id:
        text = client.files.content(batch.output_file_id).text
        Path("batch_output_raw.jsonl").write_text(text, encoding="utf-8")
        print("   saved batch_output_raw.jsonl. Inspect one line, then adapt part_e to its shape.")


def part_e():
    by_id = {}
    for line in Path("batch_output.jsonl").read_text().splitlines():
        row = json.loads(line)
        by_id[row["custom_id"]] = row  # join on custom_id, never on position
    print("== Part E: results joined back to inputs ==")
    for i, text in enumerate(REVIEWS):
        row = by_id.get(f"review-{i:03d}", {"status": "missing"})
        print(f"   {row.get('label', row['status']):9} | {text}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "a"
    if part == "a":
        part_a(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    else:
        {"b": part_b, "c": part_c, "d": part_d, "e": part_e}[part]()
```

**Steps.**

1. `python day08_throughput.py b` (2 min). No API calls. Confirm 5 awaited sleeps take about 1 s and 5 blocking sleeps take about 5 s. That is what one sync call inside async code does to you.
2. `python day08_throughput.py a 3` (10 min). Compare wall time at concurrency 1 and 3, check that peak in-flight never exceeds the cap, and note that results print in input order even though they finished in a different order (`gather` keeps order). Then try `a 6`. If you start seeing 429 retries, you have found your free-tier ceiling.
3. Size it properly (5 min). From Part A, take your average `secs` per call. Using Little's law, write in a comment what concurrency would saturate a 4,000 RPM Claude limit at that latency, and why ITPM might cap you lower for long prompts.
4. `python day08_throughput.py c` (10 min). Open `batch_input.jsonl` and read one line: this is exactly what a batch job is. If the provider accepted it, run `d` later (minutes to hours) and inspect `batch_output_raw.jsonl`. If it fell back to the emulator, run `e` and check that every review got its label despite the shuffled file.
5. Cost on paper (5 min). 10,000 reviews at about 300 input and 5 output tokens each on Sonnet 5.5: work out the cost at standard price ($2/$10 per million) and batch price ($1/$5). Then add: what if the 250-token system prompt were cached at 0.1× on top of the batch discount?
6. Cloud on paper (5 min). Rewrite the Part A client line three ways, as `AnthropicBedrockMantle`, `AnthropicVertex` and `AnthropicFoundry` with the Sonnet 5.5 model name for each. Note which of today's techniques (batch, async) would still work on each platform.

**Done when:** (1) you have the Part B timings and can explain the 5× difference in one sentence; (2) Part A shows a clear speedup at concurrency 3 with peak in-flight equal to the cap; (3) you have a batch result (provider or emulator) joined correctly by `custom_id`; (4) your Day-08 notes contain the Little's-law concurrency number, the standard vs batch vs batch-plus-cache costs for 10,000 reviews, and the three cloud client lines.

## Go deeper (optional)

- [Batch processing (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/batch-processing) — limits, result types, polling and result streaming, and how caching behaves inside batches.
- [LiteLLM proxy: fallbacks, retries and cooldowns](https://docs.litellm.ai/docs/proxy/reliability) — a concrete look at how an open-source AI gateway configures routing and failover.

## Recall answers

1. Any two of: that the values are true (a valid `order_id` can be invented); business rules the schema can't express (formats, ranges on Claude); a complete answer when `stop_reason` is `max_tokens` or the model refuses.
2. Retries multiply across layers: SDK × your loop × gateway, for example 3 × 3 × 3 = 27 calls per request during an outage. That burns quota and money and keeps the overloaded provider down. Let one layer own retries and set `max_retries=0` elsewhere.
3. The prefix changes on every request, so the cache never hits and you pay full price every time. Put stable content (tools, policy) first and dynamic values such as timestamps at the end.
