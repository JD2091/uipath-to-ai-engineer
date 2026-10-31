# Day 3 — From order to answer: positions, post-training, and picking the next token (2026-10-31)

Today's thread: attention by itself cannot see word order, so models add **position information**, which also sets the **context window** (1.7). A pretrained model is then **post-trained** into an assistant (1.8). Finally, at every step a **sampling** rule turns the model's probabilities into one actual token (1.9). Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. You switch your RAG system from embedding model A to model B. Why must you re-embed every stored document?
2. In a decoder-only transformer, what does the causal mask do, and why is it needed?
3. Why is QKᵀ divided by √d_k before the softmax?

## 1.7 Positional encoding and the context window

**The idea.** Self-attention (Day 2) compares every token with every other token, but nothing in that maths says which token came first. Shuffle the input and you get the same outputs, shuffled. So "the bot paid the vendor" and "the vendor paid the bot" would look alike. **Positional encoding** injects order. The **context window** is the maximum number of tokens the model can attend over in one call. It exists because models are trained on sequences up to some length, and attention cost grows with the square of that length.

**How it works.** Three main families:

```
Absolute, learned      one trained vector per slot 0..N-1, added to the embedding (GPT-2: 1,024 slots)
Absolute, sinusoidal   fixed sin/cos waves at many frequencies, added to the embedding (2017 paper)
Relative (RoPE)        rotate each query and key by an angle proportional to its position;
                       the q·k score then depends only on the OFFSET (m - n), not on m or n
                       -> used by Llama, Qwen, Mistral, SmolLM and most open models today
```

With RoPE, a query at position 105 and a key at 102 score exactly the same as a query at 5 and a key at 2, because only the gap of 3 matters. That makes the pattern "the token 3 back" reusable everywhere. Context extension tricks such as position interpolation and YaRN rescale those rotation angles so a model trained on shorter sequences can be stretched to longer ones with a short extra training run.

What the window has to hold on every call:

```
| system prompt | tool definitions | conversation history | retrieved docs | user msg || model output |
|<------------------------------- input tokens ---------------------------------->||<- max_tokens ->|
                          must fit within the model's limits on every single request
```

**Gotchas.**

- The API is stateless: each turn you resend the whole history, so a long chat fills the window and the bill on every call.
- Advertised is not effective. Models often use facts at the start and end of a long prompt better than facts buried in the middle ("lost in the middle").
- Input and output limits are usually separate numbers. Check both on the provider's model page; reasoning tokens may also count against output.
- Chat apps often truncate or summarise old turns silently. Your own app has to choose a strategy (trim, summarise, retrieve) on purpose.

**UiPath bridge.** When an Agent Builder agent pulls a long email thread or a large tool result into its prompt, that text spends the same fixed budget as its instructions and tool definitions.

**Interview angle.**

- *Why do transformers need positional encoding?* Attention is order-blind (permutation-equivariant). Without position signals the model cannot tell "A paid B" from "B paid A".
- *What is RoPE and why is it popular?* It rotates queries and keys by position-dependent angles so attention scores depend on relative distance. It adds no extra parameters, works across positions, and can be rescaled to extend the context window.

## 1.8 Pretraining → instruction tuning → RLHF/RLAIF/Constitutional AI

**The idea.** On Day 1 you saw that a base model continues text rather than answering. Pretraining gives it knowledge of language and the world. **Post-training** turns it into an assistant that follows instructions, uses a chat format, refuses harmful requests and stays honest. The stages exist because next-token prediction on internet text rewards imitating the internet, not being helpful.

**How it works.**

```
1. Pretraining         trillions of tokens of web, books, code      loss: next token          -> BASE model
                       (by far most of the compute)
2. Instruction tuning  ~10^4-10^6 (prompt, ideal reply) pairs        loss: next token, only    -> SFT model
   (SFT)               written in the chat template                  on the reply tokens
3. Preference tuning   people (RLHF) or an AI judge (RLAIF) pick     reward model + RL (PPO),  -> chat/assistant
                       the better of two replies                     or DPO directly on pairs     model
```

- **RLHF** (InstructGPT, 2022): train a reward model on human comparisons, then use RL to push the model toward high-reward replies. A **KL penalty** keeps it close to the SFT model so it does not drift into gibberish that fools the reward model. Labelers preferred the 1.3B-parameter InstructGPT over the 175B GPT-3.
- **DPO** skips the separate reward model and RL loop and trains directly on (chosen, rejected) pairs. It is simpler and common for open models.
- **RLAIF** replaces human labelers with an AI judge, which is cheaper and scales.
- **Constitutional AI** (Anthropic, 2022) is a form of RLAIF guided by a written list of principles. The model critiques and revises its own drafts against the principles (supervised phase), then AI preferences based on those principles drive the RL phase.
- Newer pipelines add **RL with verifiable rewards** on maths and code, where a checker grades the answer. This is how reasoning models are trained (topic 1.12).

**Gotchas.**

- Post-training mostly shapes behaviour and format. It adds little new knowledge; facts come from pretraining (or from your prompt).
- Optimising for human approval has side effects: sycophancy (agreeing with the user), verbosity, and over-hedging. This is a form of reward hacking.
- Each open model expects the exact chat template it was tuned on. Feed the wrong special tokens and quality drops sharply.
- Alignment training lowers the rate of bad outputs; it is not a guarantee. You still need guardrails (Module 11).

**UiPath bridge.** Document Understanding's loop, where humans correct extractions in Validation Station and the corrections retrain the ML extractor, is the same idea as SFT on curated examples, at a much smaller scale.

**Interview angle.**

- *Walk me through RLHF.* SFT on demonstrations, then collect human rankings of replies, train a reward model on them, then optimise the policy with RL (PPO) to maximise reward minus a KL penalty to the SFT model.
- *DPO vs RLHF?* DPO optimises directly on preference pairs with a classification-style loss; there is no reward model or RL loop, so it is cheaper and more stable. RLHF is more flexible, since the reward model can score fresh samples during training.

## 1.9 Sampling: temperature, top-p, top-k, determinism

**The idea.** The model outputs a probability for every token in its vocabulary. A **decoding strategy** picks one. Always picking the top token (greedy) tends to be bland and can loop; drawing from the full distribution occasionally picks a junk token from the long tail. Temperature, top-k and top-p are knobs between those extremes.

**How it works.** The pipeline is logits → divide by temperature T → optional top-k cut → softmax → optional top-p cut → renormalise → random draw. Example for "The invoice is ___":

```
token        logit   T=0.5   T=1.0   T=2.0   T=1.0 + top-p 0.8
overdue       3.0    0.67    0.50    0.40    0.62
paid          2.5    0.24    0.30    0.31    0.38
attached      2.0    0.09    0.18    0.24    0      (cut: the top two already reach 0.80)
purple       -1.0    0.00    0.01    0.05    0
```

- **Temperature** rescales: T < 1 sharpens toward the top token; T > 1 flattens; T = 0 is treated as greedy (argmax).
- **Top-k** keeps only the k most likely tokens.
- **Top-p (nucleus)** keeps the smallest set whose probabilities add up to p, so the set grows when the model is unsure and shrinks when it is confident.

**Determinism.** Greedy decoding is deterministic in theory. On hosted APIs, even T = 0 can vary between calls: requests are batched with others, GPU floating-point maths is not perfectly associative, and the serving stack changes over time. Seed parameters, where offered, are best-effort. Design for variation: validate outputs against a schema, store what you got, and run evals over several samples.

**Gotchas.**

- Low temperature does not fix hallucinations. A wrong answer at T = 0 is just wrong consistently.
- Tune temperature or top-p, not both at once; they interact.
- Some current models advise against touching these knobs. Google's Gemini 3 developer guide recommends leaving temperature, top-p and top-k at their defaults and using clear system instructions for consistency.
- Repetition loops are a classic greedy-decoding failure, especially in small models.

**UiPath bridge.** A published RPA workflow gives the same output for the same input (Day 1 Rosetta row). An LLM step is a random draw, so treat it like an unreliable selector: validate its output before the deterministic workflow consumes it.

**Interview angle.**

- *Temperature vs top-p?* Temperature reshapes the whole distribution; top-p truncates its tail adaptively, keeping the smallest set of tokens covering p of the probability mass.
- *Why isn't temperature 0 fully deterministic on an API?* Batching, non-associative floating-point arithmetic on GPUs, and infrastructure or model updates can flip near-tied argmax choices, and one flipped token changes everything after it.

## Hands-on (30–45 min): see positions, post-training and sampling in a real small model

**Goal.** Prove RoPE depends only on relative position, compare a base and an instruction-tuned version of the same small model, then write your own sampler and drive generation with it.

**Setup.** Part A needs only `numpy`. Parts B and C run locally and free with Hugging Face: `pip install torch transformers` (the default PyPI build of torch on Windows is CPU-only, which is fine here). The first run downloads two 135M-parameter models from the SmolLM2 family, about 270 MB each, into your Hugging Face cache. A CPU is fast enough. No API key needed.

**Part A — position encodings in numpy (1.7), about 10 min.**

```python
import numpy as np
rng = np.random.default_rng(0)
d = 8

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def attend(X):                                    # attention without a causal mask, Q = K = V = X for brevity
    return softmax(X @ X.T / np.sqrt(X.shape[-1])) @ X

# 1) Without position info, attention cannot see word order
X = rng.normal(size=(4, d))                       # 4 token embeddings
perm = [2, 0, 3, 1]                               # shuffle the "sentence"
print("no positions -> shuffled input just gives shuffled output:",
      np.allclose(attend(X)[perm], attend(X[perm])))

# 2) Sinusoidal positional encoding (original 2017 transformer): add a position vector to each embedding
def sinusoidal(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (10000 ** (i / d))
    PE = np.zeros((n_pos, d)); PE[:, 0::2] = np.sin(ang); PE[:, 1::2] = np.cos(ang)
    return PE
PE = sinusoidal(4, d)
print("with positions  -> order now changes the result:   ",
      not np.allclose(attend(X + PE)[perm], attend(X[perm] + PE)))

# 3) RoPE: rotate q and k by an angle proportional to their position
def rope(x, pos, base=10000):
    freqs = base ** (-np.arange(0, len(x), 2) / len(x))
    ang = pos * freqs
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out

q, k = rng.normal(size=d), rng.normal(size=d)
for m, n in [(5, 2), (105, 102), (5005, 5002), (5, 4), (50, 2)]:
    print(f"query at {m:>4}, key at {n:>4} (offset {m-n:>2}): score = {rope(q, m) @ rope(k, n):+.4f}")
```

**Part B — base vs instruction-tuned, same family (1.8 + 1.7), about 15 min.**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
question = "What is the capital of France?"

for name in (BASE, INSTRUCT):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    cfg = model.config
    print(f"\n=== {name}")
    print("context window (max_position_embeddings):", cfg.max_position_embeddings)
    print("RoPE settings:", getattr(cfg, "rope_theta", None) or getattr(cfg, "rope_parameters", None))

    if name == INSTRUCT:   # the instruct model was tuned on this exact chat template
        text = tok.apply_chat_template([{"role": "user", "content": question}],
                                       tokenize=False, add_generation_prompt=True)
    else:                  # a base model only ever saw raw text
        text = question
    print("exact text the model sees:", repr(text))

    inputs = tok(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)   # greedy
    print("->", tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Then try giving the **base** model the chat-template text and the **instruct** model the raw question. Note what changes.

**Part C — your own sampler driving generation (1.9), about 15 min.**

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME).eval()

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def next_token_dist(logits, temperature=1.0, top_k=None, top_p=None):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature == 0:                                   # greedy
        p = np.zeros_like(logits); p[logits.argmax()] = 1.0; return p
    z = logits / temperature
    if top_k:
        kth = np.sort(z)[-top_k]
        z = np.where(z < kth, -np.inf, z)
    p = softmax(z)
    if top_p:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        keep = order[: np.searchsorted(cum, top_p) + 1]    # smallest set reaching top_p
        mask = np.zeros_like(p, dtype=bool); mask[keep] = True
        p = np.where(mask, p, 0.0); p /= p.sum()
    return p

prompt = tok.apply_chat_template([{"role": "user", "content": "Describe an invoice-processing robot in one sentence."}],
                                 tokenize=False, add_generation_prompt=True)
prompt_ids = tok(prompt, return_tensors="pt")["input_ids"]

# 1) Look at ONE next-token distribution under different settings
with torch.no_grad():
    logits = model(input_ids=prompt_ids).logits[0, -1].double().numpy()
for cfg in [dict(temperature=0), dict(temperature=0.5), dict(temperature=1.0), dict(temperature=1.5),
            dict(temperature=1.0, top_k=20), dict(temperature=1.0, top_p=0.9)]:
    p = next_token_dist(logits, **cfg)
    top = np.argsort(-p)[:3]
    print(f"{str(cfg):38} tokens kept: {(p > 0).sum():>6} | top-3:",
          [(tok.decode([int(i)]), round(float(p[i]), 2)) for i in top])

# 2) Generate whole replies with YOUR sampler
def generate(max_new=30, seed=None, **cfg):
    rng = np.random.default_rng(seed)
    ids = prompt_ids.clone()
    for _ in range(max_new):
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, -1].double().numpy()
        nxt = int(rng.choice(len(logits), p=next_token_dist(logits, **cfg)))
        if nxt == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    return tok.decode(ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)

print("\ngreedy x2:")
for _ in range(2): print("  ", generate(temperature=0))
print("T=1.0, top_p=0.9, no seed x3:")
for _ in range(3): print("  ", generate(temperature=1.0, top_p=0.9))
print("T=1.0, top_p=0.9, seed=42 x2:")
for _ in range(2): print("  ", generate(seed=42, temperature=1.0, top_p=0.9))
print("T=2.0 (watch it degrade):")
print("  ", generate(temperature=2.0))
```

**Done when:** (1) Part A prints True twice and the same score for offset 3 at positions 5, 105 and 5005, and you can explain why offsets 1 and 48 give different scores; (2) in Part B the base model rambles or continues the question while the instruct model answers, you have noted the context window and the special tokens in the chat template, and you can say which training stage made the difference; (3) in Part C the "tokens kept" count shrinks under top-k and top-p, both greedy runs match, the two seed-42 runs match, the unseeded runs differ, and T = 2.0 produces visibly worse text. Bonus: write one sentence on why a hosted API at temperature 0 might still not reproduce your greedy result exactly.

## Go deeper (optional)

- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) — the InstructGPT paper; Figure 2 shows the SFT → reward model → PPO pipeline.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864) — the RoPE paper; section 3 derives why the score depends only on relative position.

## Recall answers

1. Each embedding model has its own vector space. Vectors from A and B are not comparable, so similarity scores across them are meaningless.
2. It blocks each position from attending to later positions (sets those scores to −∞). Without it the model could see the answer during training, and it would not learn next-token prediction.
3. Dot products grow with dimension. Unscaled, the softmax saturates to near one-hot and gradients vanish; scaling keeps scores in a useful range.
