# Day 6 — One stateless call: what goes in, what it costs, and how it arrives (2026-11-04)

Module 2 starts here: from how models work to how you call them. Every LLM API call is a stateless function. You send the system prompt, the whole conversation and some parameters, and you get one assistant message back (2.1). You pay for every token in both directions, and the provider limits how fast you can spend them (2.2). Streaming changes when the answer reaches the user, not how long it takes to generate (2.3). Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. Roughly how many tokens does a 1000×1000 px image cost on Claude, and why does a 4K photo cost no more than a full-page scan?
2. Give two reasons why comparing per-token prices across vendors is misleading.
3. There are 50 real invoices. The model flags 40 documents, and 36 of them are invoices. What are precision, recall and F1?

## 2.1 Messages API anatomy: roles, system prompt, multi-turn state

**The idea.** A chat API has no memory. Each request is a pure function: (instructions + full conversation + settings) → one new assistant message. The "chat" you see in an app exists because the app keeps the history and sends all of it again on every turn. The design exists because statelessness makes servers simple to scale and retry, and it gives you full control: you decide what the model sees, and you can edit, trim, summarise or invent earlier turns.

**How it works.** A request has three parts:

```
POST /v1/messages   (Anthropic)        POST /v1/chat/completions   (OpenAI-style; Gemini's compat endpoint)
{                                       {
  "model": "claude-sonnet-5-5",           "model": "gemini-...",
  "max_tokens": 1024,     <- required     "max_tokens": 1024,          <- optional
  "system": "You are ...", <- top-level   "messages": [
  "messages": [                              {"role":"system","content":"You are ..."},  <- a role
    {"role":"user","content":"Hi, I'm JD"},  {"role":"user","content":"Hi, I'm JD"},
    {"role":"assistant","content":"Hi JD"},  {"role":"assistant","content":"Hi JD"},
    {"role":"user","content":"My name?"}     {"role":"user","content":"My name?"}
  ]                                        ]
}                                       }
-> content: [ {type:"text", text:...} ]  -> choices[0].message.content
   stop_reason: end_turn | max_tokens |     finish_reason: stop | length | tool_calls | ...
                tool_use | refusal ...
   usage: input_tokens, output_tokens       usage: prompt_tokens, completion_tokens
```

- **system**: standing instructions (persona, rules, format, context). On Anthropic it is a top-level field. In OpenAI-style APIs it is the first message with role `system` (newer OpenAI models also accept `developer`).
- **user / assistant**: the alternating turns. Content can be a plain string or a list of blocks (text, image, document, tool_use, tool_result).
- **Multi-turn state is your list.** Turn 3 resends turns 1–2. So input tokens grow every turn, and the total cost of a long chat grows roughly with the square of its length:

```
turn 1 sends: S + U1
turn 2 sends: S + U1 + A1 + U2
turn 3 sends: S + U1 + A1 + U2 + A2 + U3     <- every earlier token is billed again
```

Current Claude-specific facts (checked 2026-11-04): newer models (Opus 5.5, Sonnet 5.5, Haiku 5.5, Opus 4.8) accept `role: "system"` messages *after* a user turn to add instructions mid-conversation. A beta "message threads" feature can keep state server-side for short sessions. The default is still stateless.

**Gotchas.**

- Forgetting to append the assistant's reply to the history is the most common multi-turn bug. The model then "forgets" its own previous answer.
- Prefill (ending the list with a partial assistant message to steer the format) returns a 400 error on Claude 4.6 and later. Use structured outputs (2.4) or system instructions instead. Older tutorials still show prefill.
- On Claude 4.7 and later, `temperature`, `top_p` and `top_k` are not supported (non-default values return 400). Day 3's sampling knobs still matter on many other models, but check per model.
- Always check the stop reason. `max_tokens` / `length` means the answer was cut off, which can break JSON halfway.
- The system prompt is not a security boundary. Users can often get it to leak, so never put secrets in it (Module 11).

**UiPath bridge.** It is like calling a REST API from an HTTP Request activity inside a transactional process. The server keeps nothing between calls, so the state lives in your variables, arguments or queue item, and you pass it in every time.

**Interview angle.**

- *How does a chatbot "remember" if the API is stateless?* The application stores the message list and resends it each turn. For long sessions you trim old turns, summarise them, or move facts into a retrieved long-term memory (Module 7), because the context window and the cost both grow with every turn.
- *What is the difference between the system prompt and a user message?* The system prompt sets standing instructions for the whole conversation and models are trained to give it more weight. User messages are the turn-by-turn input. Both are just tokens in the context, so neither is a hard security control.

## 2.2 Pricing, token budgeting, rate limits

**The idea.** You pay per token, with input and output priced separately, and output usually costs about 5× input because each output token needs its own forward pass (Day 4). Providers also cap how fast you can send requests and tokens, to share capacity fairly. Cost and limits are design inputs, not things to check after launch.

**How it works — pricing.** Cost per request = input_tokens × input_price + output_tokens × output_price. Claude prices per million tokens ([pricing page](https://platform.claude.com/docs/en/about-claude/pricing), checked 2026-11-04):

```
Model                     Input    Output   Cache hit   Batch (in / out)
Opus 5.5                  $4.00    $20.00   $0.20       $2.00 / $10.00
Sonnet 5.5                $2.00    $10.00   $0.10       $1.00 / $5.00
Haiku 5.5 (prompt ≤100K)  $0.10    $0.50    $0.01       $0.05 / $0.25
Haiku 5.5 (prompt >100K)  $0.50    $2.50    $0.05       $0.25 / $1.25
```

Hidden extras: a request with tools adds about 286–406 system tokens plus the tool definitions; reasoning/thinking tokens are billed as output (Day 4); US-only data residency on Claude adds a 1.1× multiplier. Caching (2.6) and batch (2.7) are the two big discounts.

Worked example: a support bot with an 800-token system prompt, 100-token questions and 200-token answers, 10 turns per conversation. Prompt tokens per turn = 900 + 300 × (turn − 1), so ten turns total 22,500 input and 2,000 output tokens.

```
Sonnet 5.5: 22,500 × $2/1M + 2,000 × $10/1M = $0.045 + $0.020 = $0.065 per conversation
            × 100,000 conversations/month ≈ $6,500
Haiku 5.5:  22,500 × $0.10/1M + 2,000 × $0.50/1M ≈ $0.0033 per conversation ≈ $325/month
```

Note that input dominates because the history is resent each turn.

**How it works — token budgeting.** The context window is a fixed budget that every part of the request shares:

```
| system | tool defs | retrieved docs | history (grows) | new user msg | reserved for output (max_tokens) |
|<------------------------------- must fit in the context window ------------------------------->|
```

Budgeting tactics: count tokens before sending (Anthropic has a free token-counting endpoint; tiktoken gives a rough estimate elsewhere); trim or summarise old turns; cap retrieved chunks; set `max_tokens` to what the task needs; and log `usage` from every response so you know real cost per task.

**How it works — rate limits.** Anthropic limits each organisation per model class on RPM (requests/min), ITPM (input tokens/min) and OTPM (output tokens/min), using a token bucket: capacity refills continuously rather than resetting each minute, and 60 RPM may be enforced as 1 per second. Start-tier Sonnet 5.5 allows 4,000 RPM, 2M ITPM and 400K OTPM. With ~2,250 input tokens per request, ITPM is what binds: 2,000,000 / 2,250 ≈ 889 requests/min, well below the RPM cap. Cached reads don't count towards ITPM on most current models, which is why caching also raises throughput. Gemini limits per project (not per key) on RPM, TPM and requests per day (RPD); see your project's limits in Google AI Studio, since the free-tier numbers change often.

When you exceed a limit you get HTTP 429. Anthropic sends a `retry-after` header (seconds) and `anthropic-ratelimit-*-remaining` / `-reset` headers so you can slow down before you hit the wall.

**Gotchas.**

- A 429 is not always temporary. Anthropic's monthly spend-cap 429 has no `retry-after` and won't clear until next month, so don't retry it in a loop.
- Sudden traffic spikes can trigger 429s below your limits ("acceleration limits"). Ramp up gradually.
- On Anthropic, `max_tokens` does not count against OTPM; only tokens actually generated do. Other providers may reserve it, so check.
- Haiku 5.5's price tier is per request: one prompt over 100K tokens pays the higher rate for the whole request.
- A cheap per-token model that needs three retries or long reasoning can cost more per task than a pricier one. Measure cost per task.

**UiPath bridge.** Integration Service connectors and HTTP Request activities hit the same vendor throttling you have handled before. Honouring `Retry-After` and backing off is the same discipline here Logging tokens per task is the LLM version of tracking consumption per process.

**Interview angle.**

- *Estimate the monthly LLM cost of this feature.* Work out tokens per task (system + history + retrieved + output) × tasks per month × price. Point out that history makes input dominate, then name the levers: a smaller model for easy cases, prompt caching, trimming history, batch for offline jobs.
- *You keep hitting 429s at peak. What do you do?* Read the rate-limit headers and throttle on the client (a token-bucket limiter or a queue), retry with exponential backoff and jitter that respects `retry-after`, cut input tokens (caching, shorter context), spread load across models or regions, and request a tier increase. Separate spend-cap errors, which must not be retried.

## 2.3 Streaming

**The idea.** Without streaming, the user stares at a spinner until the last token is generated. With streaming, the server sends tokens as they are produced, so text starts appearing after the time to first token (TTFT), usually under a second or two. The total generation time is the same; perceived latency drops a lot. Streaming also keeps long requests alive: the Anthropic SDKs require streaming for very large `max_tokens` values to avoid HTTP timeouts.

**How it works.** The response is a long-lived HTTP connection using server-sent events (SSE): lines of `event:` and `data:` sent as they happen. Anthropic's event order:

```
message_start                      (message shell, input usage)
  content_block_start  index 0     (text)
    content_block_delta  text_delta "Hel" / "lo" / " JD" ...
  content_block_stop   index 0
  content_block_start  index 1     (tool_use)
    content_block_delta  input_json_delta '{"city": "Pu' / 'ne"}'
  content_block_stop   index 1
message_delta                      (stop_reason, cumulative usage)
message_stop
(ping events can appear anywhere; an error event can arrive mid-stream)
```

OpenAI-style streams send `chunk.choices[0].delta.content` pieces and, if you pass `stream_options={"include_usage": True}`, a final chunk with `usage` and an empty `choices` list. The SDKs hide most of this: in Anthropic's Python SDK, `client.messages.stream(...)` gives you `text_stream` to print and `get_final_message()` to get the same object a non-streaming call returns.

```
Non-streaming:  [------------ 6.0 s generating ------------] -> whole answer appears
Streaming:      [0.6 s TTFT] text appears ... keeps arriving ... done at 6.0 s
```

**Gotchas.**

- Errors can arrive after a 200 OK. An `overloaded_error` event mid-stream means your code must handle failure partway through, and you have already shown the user half an answer.
- You can't validate or moderate the full output before the user sees it. If you need a guardrail or JSON validation, buffer first or validate in chunks.
- Tool arguments stream as partial JSON strings. Don't parse them until the block stops (or use the SDK helper).
- Proxies and API gateways may buffer SSE and undo the benefit. Turn off response buffering on that hop.
- Resuming an interrupted stream: on Claude 4.6 and later, send the partial text back in a user message and ask the model to continue (prefill no longer works).
- Streaming does not help a background robot that needs the whole answer anyway, except to avoid timeouts.

**UiPath bridge.** There is no real RPA equivalent, since robots consume complete responses. Streaming matters when a person is watching a chat-style assistant.

**Interview angle.**

- *What do TTFT and tokens per second measure, and what drives each?* TTFT is queueing plus prefill, so it grows with prompt length (Day 4). Tokens per second is the decode speed, driven by model size and server load. Streaming improves perceived latency, not total latency.
- *How do you stream a response and still return valid JSON?* Stream for UX but validate the accumulated text at the end against the schema, or stream structured output with a partial-JSON parser. On a validation failure, retry or repair before any downstream system acts on it.

## Hands-on (30–45 min): a streaming chat CLI with a token budget

**Goal.** Prove the API is stateless, then build a small terminal chat that keeps its own history, streams replies, measures TTFT and decode speed, trims history to a token budget, handles 429s, and prints what the session would cost on Claude Sonnet 5.5 and Haiku 5.5.

**Setup.** Your existing `.env` (`GEMINI_API_KEY`, `LLM_MODEL`) and the `openai` and `python-dotenv` packages from `requirements.txt`. Nothing new to install. Free-tier Gemini is enough; total use is a few thousand tokens. If your model rejects `reasoning_effort`, delete it from `EXTRA`. Put the scripts in your own `Day-06` exercise folder.

**Part A — prove it's stateless (5 min).**

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
)
MODEL = os.environ["LLM_MODEL"]
EXTRA = {"reasoning_effort": "low"}  # keeps thinking tokens small; remove if your model rejects it


def ask(messages):
    r = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024, **EXTRA)
    return r.choices[0].message.content, r.choices[0].finish_reason, r.usage


# 1) Two independent calls: nothing is shared between them
print(ask([{"role": "user", "content": "My name is JD and I build UiPath agents. Reply 'noted'."}])[0])
print(ask([{"role": "user", "content": "What is my name?"}])[0])

# 2) The same question with the history resent by us
history = [
    {"role": "system", "content": "You are a concise tutor. Answer in one sentence."},
    {"role": "user", "content": "My name is JD and I build UiPath agents. Reply 'noted'."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "What is my name, and what do I build?"},
]
text, finish, usage = ask(history)
print(text, "| finish:", finish, "| prompt:", usage.prompt_tokens, "completion:", usage.completion_tokens)
```

**Part B + C — streaming chat with a budget (30 min).** Save as `chat_budget.py` and run it.

```python
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
```

Then experiment:

1. Chat for 6–8 turns. Write down `prompt=` for each turn and check that it grows by roughly the size of the previous turn.
2. Set `HISTORY_BUDGET = 300`, tell it your name in turn 1, chat for a few turns, then ask your name. Watch the trim message and the model forgetting.
3. Comment out the `history.append(... "assistant" ...)` line and ask "what did you just say?". Then restore it.
4. On paper or in a comment, rewrite the turn-3 request body as an Anthropic Messages API call: top-level `system`, required `max_tokens`, and `stop_reason` in place of `finish_reason`.

**Done when:** (1) Part A shows the first call can't name you and the history call can; (2) the CLI streams text, prints TTFT, total time, tokens/s and prompt/completion tokens per turn, and you can explain why TTFT is a small part of total time; (3) you have a turn-by-turn list of prompt tokens showing growth, and you saw trimming make the model forget; (4) the cost summary prints, and you have written two lines in your Day-6 README: the monthly cost at 100,000 sessions for each model, and one change that would cut it (for example caching the system prompt, trimming history, or routing simple turns to Haiku). Bonus: if `completion_tokens` is much larger than the visible reply, that gap is thinking tokens, billed as output.

## Go deeper (optional)

- [Rate limits (Claude Platform docs)](https://platform.claude.com/docs/en/api/rate-limits) — the token bucket, cache-aware ITPM, tier tables and every `anthropic-ratelimit-*` header.
- [Streaming messages (Claude Platform docs)](https://platform.claude.com/docs/build-with-claude/streaming) — the full SSE event flow, delta types, mid-stream errors and how to recover an interrupted stream.

## Recall answers

1. About 1,296 tokens (ceil(1000/28)² = 36 × 36). Large images are downscaled to the model's cap before tokenising, so a 4K photo and an A4 scan both end up at the cap (about 1,568 tokens on standard models).
2. Any two of: tokenizers differ, so the same text is a different number of tokens; reasoning tokens are billed as output and vary by model; retries, longer prompts or extra calls change the cost per task. Compare cost per task, not per token.
3. TP = 36, FP = 4, FN = 14. Precision = 36/40 = 0.90, recall = 36/50 = 0.72, F1 = 2 × 0.90 × 0.72 / 1.62 = 0.80.
