# Day 4 — Paying for every token: the KV cache, hallucinations, and models that think first (2026-11-02)

Today's thread: generation is a loop of one forward pass per output token, and the **KV cache** is what keeps that loop affordable, and also why long prompts cost memory and time (1.10). That loop always produces *some* fluent next token, even when the model does not know the answer, which is the root of **hallucination** (1.11). **Reasoning models** spend extra output tokens thinking before they answer: more passes through the same loop, bought deliberately to get better answers on hard problems (1.12). Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. With RoPE, what does the attention score between a query and a key depend on: their absolute positions or something else?
2. Name the three post-training stages, and say in one line what DPO removes compared with RLHF.
3. Logits are [2.0, 1.0, 0.0] and top-p = 0.9. Which tokens survive?

## 1.10 KV cache, latency, and why long context costs more

**The idea.** A decoder generates one token per forward pass, and each new token attends to every earlier token. Done naively, step 500 would recompute the keys and values for all 499 earlier tokens, which were already computed at step 499. The **KV cache** stores each layer's key and value vectors for every token seen so far, so each step only computes Q, K and V for the one new token and reuses the rest. It exists to turn generation from "redo everything each step" into "add one row each step".

**How it works.** A request has two phases with very different costs:

```
PREFILL  (your whole prompt, in parallel)          DECODE  (one token at a time)
  N prompt tokens -> one big forward pass            new token -> Q,K,V for 1 token
  fills the KV cache for all N tokens                attends to cached K,V of all previous tokens
  compute-bound; sets time-to-first-token (TTFT)     appends its K,V to the cache
                                                     memory-bandwidth-bound; sets time per output token

total latency  ≈  TTFT (grows with prompt length)  +  output_tokens × time_per_token
```

How big the cache gets, per token: `2 (K and V) × layers × kv_heads × head_dim × bytes_per_value`.

```
Llama 3.1 8B: 32 layers, 8 KV heads, head_dim 128, fp16 (2 bytes)
  2 × 32 × 8 × 128 × 2 = 131,072 bytes = 128 KiB per token
  8k-token prompt   ->  1 GiB of GPU memory, for ONE request
  128k-token prompt -> 16 GiB, about as much as the model's own weights (~16 GB in fp16)
```

That is why long context costs more. Prefill compute grows with prompt length (the attention part grows with its square). Every decode step must read the entire cache from GPU memory, so each output token gets slower as context grows. And a big cache per request means fewer requests fit on a GPU at once, so the provider serves fewer users per GPU. Providers pass this on as per-token input pricing, and some charge a higher rate above a context threshold.

Tricks you will hear about: **grouped-query attention (GQA)** shares each K/V head across several query heads (Llama 3.1 8B has 32 query heads but only 8 KV heads, a 4× smaller cache); **PagedAttention** (vLLM) stores the cache in fixed-size blocks like OS memory pages, so little memory is wasted; **quantised KV cache** stores values in 8 bits or fewer; **prefix caching** keeps the cache of a shared prompt prefix and reuses it across requests. Prefix caching is what API "prompt caching" sells you as cheaper, faster cached input tokens (topic 2.6).

**Gotchas.**

- The KV cache speeds up inference only. Training processes whole sequences in parallel and does not use it.
- The cache holds K and V, not Q. A past token's query is never needed again; only the newest token asks questions.
- Output tokens dominate latency for long answers: 1,000 output tokens is 1,000 sequential decode steps, while 1,000 extra input tokens run in parallel during prefill. Cutting output length is usually the biggest latency win.
- Prefix caching needs a byte-identical prefix. Put stable content (system prompt, tool definitions, documents) first and the changing parts (user message, timestamps) last, or every request is a cache miss.

**UiPath bridge.** It is like an Orchestrator queue processor that keeps already-fetched reference data in memory instead of re-querying the system for every queue item: faster per item, but the in-memory data grows with what you have loaded.

**Interview angle.**

- *What is the KV cache and what does it trade?* It stores per-layer keys and values of previous tokens so each decode step computes only the new token. It trades GPU memory (linear in sequence length × batch size) for removing repeated computation. Memory, not compute, then limits context length and concurrency.
- *Why does latency rise with a long prompt even for a short answer?* Prefill must process every prompt token before the first output token (higher TTFT), and every decode step reads a larger cache. Mitigations: shorter prompts, retrieval instead of stuffing, prompt caching, and streaming to hide TTFT.

## 1.11 Hallucinations: causes and mitigations

**The idea.** A **hallucination** is fluent, confident output that is false or not supported by the given sources: an invented citation, API parameter, policy clause or number. It happens because the model is always predicting a plausible next token. It has no built-in "I don't know" state unless training rewarded one. For an AI engineer this is the core reliability problem: the output looks equally confident whether it is right or wrong.

**How it works: the main causes.**

```
Cause                                  Example                                    Mitigation
-------------------------------------  -----------------------------------------  -------------------------------------------
Rare, arbitrary facts in training      a person's birthday seen once on the web   retrieve the fact (RAG / tool), don't recall it
Training and evals reward guessing     "I don't know" scores 0, a guess may score allow and reward abstaining; eval for it
Knowledge cutoff                       asks about a release after the cutoff      search / tools; state the date in the prompt
Missing context                        asks about your client's SOP              give the documents; say "answer only from them"
Long or noisy context                  fact buried mid-prompt is missed or mixed  less, better-ranked context; quote first
Sycophancy / leading questions         "Why is X true?" when X is false           neutral phrasing; permission to disagree
Sampling                               a low-probability wrong token is drawn     lower T helps a little; it does not fix knowledge gaps
```

OpenAI's September 2025 paper argues two root causes. First, in pretraining, facts that appear only once in the data have no pattern to learn, so the model's error rate on such facts is roughly bounded below by the fraction of them that appear just once. Second, most benchmarks grade accuracy only, so a model that guesses beats one that admits uncertainty. Their proposed fix is to score confident errors worse than abstentions.

Practical mitigation stack, from cheapest to strongest:

1. Ground it: put the source text in the prompt and say to answer only from it.
2. Give an explicit out: "If the answer is not in the documents, reply `NOT_FOUND`."
3. Ask for supporting quotes or citations first, then the answer; verify the quotes exist in the source with code.
4. Constrain the format (enums, JSON schema) so free-form invention has nowhere to go.
5. Verify: check against a database or API, run a second model as a checker, or sample several times and flag disagreement (self-consistency).
6. Measure: keep a test set that includes unanswerable questions and track the hallucination rate (Module 8).

**Gotchas.**

- RAG reduces hallucination but does not remove it: the model can still misread, mix up or embellish the retrieved text. Grounding needs checking too.
- Temperature 0 does not prevent it. A confident wrong answer comes out the same way every time.
- Citations can themselves be hallucinated. Check that the cited passage exists and actually supports the claim.
- Bigger and newer models hallucinate less on common facts but still do on rare ones, and fluent errors are harder to spot.

**UiPath bridge.** Treat an LLM field like a Document Understanding extraction with a confidence score: route anything unverified or low-confidence to Action Center for human validation instead of passing it straight into the downstream system.

**Interview angle.**

- *Why do LLMs hallucinate?* They are trained to produce plausible continuations, not to track truth. Rare facts have no learnable pattern, and both training and benchmarks have rewarded guessing over abstaining. Missing or noisy context and leading prompts make it worse.
- *How would you reduce hallucinations in an enterprise Q&A bot?* Retrieve and ground; instruct to answer only from sources with an explicit "not found" path; require citations and verify them in code; constrain outputs; add a checker for high-risk answers; and run evals with unanswerable questions to measure the rate over time.

## 1.12 Reasoning models and extended thinking

**The idea.** A standard model commits to its answer token by token from the start. A **reasoning model** first generates a long internal chain of thought (planning, trying approaches, checking its work, backtracking) and only then writes the final answer. It exists because some problems (multi-step maths, code, planning, tricky logic) need more computation than one pass per answer token gives. Spending more tokens at inference time buys accuracy. This is called **test-time compute** scaling.

**How it works.**

```
Standard:   prompt ──► answer tokens
Reasoning:  prompt ──► [ thinking tokens: plan → try → check → fix ] ──► answer tokens
                        (billed as output; often hidden or summarised; costs latency)
```

- **Training.** Reasoning behaviour is mostly learned with reinforcement learning on tasks a program can grade (maths answers, unit tests). The model is rewarded for correct final answers, and long, self-correcting reasoning emerges because it helps get them right. DeepSeek-R1 (January 2025) showed this openly.
- **Controls in today's APIs (checked 2026-11-02).** Anthropic's Claude API now uses adaptive thinking: you set `thinking: {"type": "adaptive"}` and an optional `effort` level (`low` to `max`) under `output_config`, and the model decides per request whether and how much to think. Fixed `budget_tokens` budgets are documented as the legacy approach. Thinking tokens are billed as output tokens even when only a summary is shown, and they count against `max_tokens`. OpenAI's reasoning models and Gemini's OpenAI-compatible endpoint both take a `reasoning_effort` parameter (Gemini accepts `minimal`, `low`, `medium`, `high`; some Gemini models cannot switch thinking off).
- **With tools.** Current models can think between tool calls (interleaved thinking), which matters for agents: they reason about each tool result before choosing the next step.

When to use it:

```
Use more thinking                         Use little or none
multi-step maths, logic, scheduling       extraction, classification, short rewrites
non-trivial code changes, debugging       simple lookups and chat
agent planning across many tool calls     high-volume, latency-sensitive steps
```

**Gotchas.**

- Thinking costs real money and time: thousands of hidden output tokens can sit behind a one-line answer. Always read the reasoning-token count in `usage`.
- The visible thinking is not a guaranteed faithful record of how the model reached its answer. Anthropic's research found reasoning models often fail to mention hints they actually used. Do not treat the chain of thought as an audit trail.
- More effort is not always better. Simple tasks can get slower and sometimes worse ("overthinking"). Tune effort per step with evals.
- Reasoning reduces some hallucinations (arithmetic, logic slips) but not missing knowledge: a model reasoning carefully about a fact it never learned is still guessing.
- Long thinking also grows the KV cache and output length, so latency effects from 1.10 apply in full.

**UiPath bridge.** In Maestro terms, it is the difference between a single automated task and an agent task that plans before acting. Use it where judgment is needed and keep cheap deterministic steps elsewhere in the process.

**Interview angle.**

- *What makes a reasoning model different from prompting "think step by step"?* Both produce intermediate reasoning, but reasoning models were trained with RL to generate long, self-checking chains and to use them well; plain chain-of-thought prompting relies on behaviour learned incidentally. Reasoning models also expose controls (effort or budget) and bill the thinking as output tokens.
- *When would you not use extended thinking?* For high-volume, latency-sensitive or simple steps (classification, extraction, routing), where it adds cost and latency with little accuracy gain. Use a mix: a fast model for easy steps and higher effort only where evals show it helps.

## Hands-on (30–45 min): measure the cache, catch a hallucination, pay for thinking

**Goal.** See the KV cache's speed-up and memory cost on a local model, provoke and then reduce a hallucination, and measure what extra reasoning costs and buys on your Gemini setup.

**Setup.** Nothing new to install. Part A reuses `torch`, `transformers` and the SmolLM2-135M-Instruct model from Day 3 (already in your Hugging Face cache). Parts B and C use the same `.env` as `check_setup.py` (`GEMINI_API_KEY`, `LLM_MODEL`) and make about 16 small API calls, well inside the free tier. Part C needs a thinking-capable model in `LLM_MODEL`; most current Gemini Flash models are. Put each part in its own file in a new `Day-4 exercise/` folder.

**Part A — the KV cache, locally (1.10), about 15 min.**

```python
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()
cfg = model.config

# 1) How big is the cache per token? Read the shape from the config.
def kv_bytes_per_token(layers, kv_heads, head_dim, bytes_per_value):
    return 2 * layers * kv_heads * head_dim * bytes_per_value      # 2 = K and V

head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
small = kv_bytes_per_token(cfg.num_hidden_layers, kv_heads, head_dim, 4)   # float32 on CPU
print(f"SmolLM2: layers={cfg.num_hidden_layers} q_heads={cfg.num_attention_heads} "
      f"kv_heads={kv_heads} head_dim={head_dim}")
print(f"  KV per token: {small/1024:.1f} KiB | 8k tokens: {small*8192/2**20:.0f} MiB")
llama = kv_bytes_per_token(32, 8, 128, 2)                                  # Llama 3.1 8B, fp16
print(f"Llama 3.1 8B: {llama/1024:.0f} KiB per token | 8k: {llama*8192/2**30:.1f} GiB | "
      f"128k: {llama*131072/2**30:.0f} GiB")
print(f"  without GQA (32 KV heads) it would be {kv_bytes_per_token(32, 32, 128, 2)/1024:.0f} KiB per token")

# 2) Same output, with and without the cache
text = tok.apply_chat_template([{"role": "user", "content": "Explain what an Orchestrator queue is."}],
                               tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt")
outs = {}
for use_cache in (True, False):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, min_new_tokens=150,
                             do_sample=False, use_cache=use_cache)
    outs[use_cache] = out
    print(f"use_cache={use_cache!s:5}  150 tokens in {time.perf_counter() - t0:5.1f}s")
print("identical output:", torch.equal(outs[True], outs[False]))

# 3) Prefill time (≈ time to first token) vs prompt length
filler = tok("The robot processed invoice 1234 and logged the result. " * 300,
             return_tensors="pt")["input_ids"]
for n in (128, 512, 1024, 2048):
    ids = filler[:, :n]
    with torch.no_grad():
        t0 = time.perf_counter()
        model(input_ids=ids, use_cache=True)
        dt = time.perf_counter() - t0
    print(f"prompt {n:>5} tokens: prefill {dt*1000:7.0f} ms | cache ≈ {n*small/2**20:5.1f} MiB")
```

**Part B — provoke, then reduce, a hallucination (1.11), about 10 min.** The first two questions are about things that do not exist; the third is real (a control); the fourth gives context that does not contain the answer.

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
MODEL = os.getenv("LLM_MODEL")

CONTEXT = ("Policy INV-7: Invoices above 50,000 INR need a second approver. "
           "Approvers must respond within 2 business days.")
QUESTIONS = [
    "Summarise the 2021 paper 'Recursive Queue Attention for Robotic Process Mining' by Hemant Vora.",
    "What is the default value of the UiPath Orchestrator setting 'Queue.AutoRetryJitterMs'?",
    "What does HTTP status code 429 mean?",
    f"Context: {CONTEXT}\n\nQuestion: What happens if an approver does not respond within 2 business days?",
]
GUARDED = ("Answer only if you are confident the thing asked about exists and you know the answer. "
           "If context is provided, use only that context. Otherwise reply exactly: I don't know. "
           "Never invent papers, settings, numbers or citations.")

def ask(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    r = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=300)
    return (r.choices[0].message.content or "").strip()

for q in QUESTIONS:
    print("\nQ:", q.splitlines()[-1][:90])
    print("  plain  :", ask(q)[:220].replace("\n", " "))
    print("  guarded:", ask(q, GUARDED)[:220].replace("\n", " "))
```

**Part C — what does thinking cost and buy? (1.12 + 1.10), about 15 min.**

```python
import os, time
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
MODEL = os.getenv("LLM_MODEL")

TASKS = [
    ("How many times does the letter r appear in 'strawberry raspberry'? "
     "End with a line 'ANSWER: <number>'.", "6"),
    ("An Orchestrator queue gets 7 items at minute 0, then 3 more items at minutes 5, 10, 15, 20, 25 and 30. "
     "Two robots start at minute 0; each takes exactly 4 minutes per item and picks the next item "
     "immediately when it finishes, if one is waiting. At minute 30, after that minute's arrivals and "
     "pickups, how many items are still waiting (not started)? End with a line 'ANSWER: <number>'.", "9"),
]

def reasoning_tokens(usage):
    details = getattr(usage, "completion_tokens_details", None)
    rt = getattr(details, "reasoning_tokens", None) if details else None
    if rt is None and usage.total_tokens:            # fallback: thinking often shows up as the gap
        rt = usage.total_tokens - usage.prompt_tokens - usage.completion_tokens
    return rt

print(f"{'effort':7} {'task':5} {'ok':3} {'secs':>5} {'answer_tok':>10} {'reason_tok':>10}")
for effort in ("low", "high"):
    for i, (prompt, expected) in enumerate(TASKS, 1):
        for run in range(2):
            t0 = time.perf_counter()
            try:
                r = client.chat.completions.create(model=MODEL, reasoning_effort=effort,
                                                   messages=[{"role": "user", "content": prompt}])
            except BadRequestError as e:
                print(f"{effort}: rejected ({e.message[:80]}) - is LLM_MODEL a thinking model?")
                break
            secs = time.perf_counter() - t0
            text = r.choices[0].message.content or ""
            got = text.rsplit("ANSWER:", 1)[-1].strip().split()[0] if "ANSWER:" in text else "?"
            print(f"{effort:7} {i:<5} {'Y' if got == expected else 'N':3} {secs:5.1f} "
                  f"{r.usage.completion_tokens:>10} {reasoning_tokens(r.usage)!s:>10}")
```

**Done when:** (1) Part A shows identical output with and without the cache, the cached run is clearly faster, prefill time rises with prompt length, and you can explain from the printed numbers why a 128k-token prompt on Llama 3.1 8B needs about as much memory for the cache as for the weights; (2) in Part B the plain prompt invents details for at least one of the two fake items (if not, try a smaller model or rephrase it more confidently), the guarded prompt says "I don't know" for the fakes, still answers 429, and does not make up a rule for the fourth question; (3) Part C prints a table where `high` uses more reasoning tokens and more seconds than `low`, and you have written one sentence in your Day-4 README on whether the extra cost bought correctness on these two tasks. Bonus: add a third, trivial task ("What is 2 + 2?") and check whether high effort wastes tokens on it.

## Go deeper (optional)

- [Why language models hallucinate (OpenAI, September 2025)](https://openai.com/index/why-language-models-hallucinate/) — the argument that rare facts and accuracy-only grading make guessing pay, with a proposed fix for evals.
- [Adaptive thinking (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking) — current controls for thinking and effort, and how thinking tokens are billed.

## Recall answers

1. Only on their relative offset (m − n). A query at 105 and a key at 102 score the same as a query at 5 and a key at 2.
2. Pretraining → supervised instruction tuning (SFT) → preference tuning (RLHF, RLAIF, or DPO). DPO removes the separate reward model and the RL (PPO) loop, training directly on chosen/rejected pairs.
3. Probabilities at T = 1 are about 0.67, 0.24 and 0.09. The top two sum to 0.91, which reaches 0.9, so the first two tokens survive and the third is cut.

