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
