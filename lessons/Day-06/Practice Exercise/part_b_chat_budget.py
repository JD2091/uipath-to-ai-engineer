import os
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    max_retries=0,  # we handle 429s ourselves so you can see them
    timeout=60,
)
MODEL = os.environ["LLM_MODEL"]
EXTRA = {"reasoning_effort": "low"}

# USD per 1M tokens (input, output) from Claude's pricing page, checked 2026-11-04
PRICES = {"claude-sonnet-5.5": (2.00, 10.00), "claude-haiku-5.5": (0.10, 0.50)}
HISTORY_BUDGET = 1500  # max estimated prompt tokens before we drop old turns
SYSTEM = {"role": "system", "content": "You are JD's AI-engineering tutor. Answer in at most 3 sentences."}

history = []  # user/assistant turns only; the system prompt is added on every call
totals = {"in": 0, "out": 0}


def estimate_tokens(msgs):
    # ~4 characters per token for English (Day 1) + a few tokens of per-message overhead
    return sum(len(m["content"]) // 4 + 4 for m in msgs)


def trim(hist):
    # drop the oldest user+assistant pair until we fit; always keep the latest user message
    while len(hist) > 1 and estimate_tokens([SYSTEM] + hist) > HISTORY_BUDGET:
        del hist[:2]
        print("   [trimmed oldest turn]")


def stream_reply(messages, attempt=0):
    t0, ttft, parts, usage = time.perf_counter(), None, [], None
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, max_tokens=1024, stream=True,
            stream_options={"include_usage": True}, **EXTRA,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                piece = chunk.choices[0].delta.content
                parts.append(piece)
                print(piece, end="", flush=True)
            if chunk.usage:  # final chunk: choices is empty, usage is filled
                usage = chunk.usage
    except openai.RateLimitError as e:
        if attempt >= 3:
            raise
        wait = float(e.response.headers.get("retry-after") or 5 * 2 ** attempt)
        print(f"\n   [429] waiting {wait:.0f}s (attempt {attempt + 1}/3)")
        time.sleep(wait)
        return stream_reply(messages, attempt + 1)
    print()
    return "".join(parts), usage, ttft or 0.0, time.perf_counter() - t0


print("Chat with the tutor. Type 'quit' to stop. Watch prompt tokens grow each turn.\n")
turn = 0
while True:
    user = input("you> ").strip()
    if user.lower() in {"quit", "exit"}:
        break
    history.append({"role": "user", "content": user})
    trim(history)
    print("bot> ", end="")
    text, usage, ttft, total = stream_reply([SYSTEM] + history)
    history.append({"role": "assistant", "content": text})  # forget this line and it loses its memory
    turn += 1
    if usage:
        totals["in"] += usage.prompt_tokens
        totals["out"] += usage.completion_tokens
        tps = usage.completion_tokens / max(total - ttft, 1e-6)
        print(f"   [turn {turn}] prompt={usage.prompt_tokens} completion={usage.completion_tokens} "
              f"TTFT={ttft:.2f}s total={total:.2f}s ~{tps:.0f} tok/s msgs={len(history)}")

print(f"\nSession: {totals['in']} input + {totals['out']} output tokens")
for name, (p_in, p_out) in PRICES.items():
    cost = totals["in"] / 1e6 * p_in + totals["out"] / 1e6 * p_out
    print(f"  {name}: ${cost:.5f} per session -> ${cost * 100_000:,.2f} for 100,000 sessions/month")
