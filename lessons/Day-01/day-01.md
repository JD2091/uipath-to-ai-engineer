# Day 1 — From text to next token: what an LLM really is (2026-10-30)

Today's thread: an LLM turns **tokens** (1.2) into a probability distribution over the **next token** (1.1), and training is nothing more than nudging weights to make that distribution less wrong (1.3). Reading ≈ 20 min, hands-on ≈ 40 min. No recall warm-up on Day 1 (progress was reset today).

## 1.1 What an LLM is: next-token prediction, base vs chat models

**The idea.** A large language model is a function that takes a sequence of tokens and returns a probability for every possible next token. That is the whole interface. It exists because predicting what comes next in text is a task with unlimited free training data (all text ever written), and doing it well forces the model to learn grammar, facts, style and some reasoning as a side effect.

**How it works.**

```
"The invoice total is"  ->  LLM  ->  { " $": 0.31, " 1": 0.12, " due": 0.08, ... }  (one probability per token in the vocabulary)
pick one (see sampling, topic 1.9)  ->  append  ->  repeat until a stop token or max length
```

Two kinds of model matter in practice. A **base model** is trained only on next-token prediction over huge text, so it *continues* text: ask it a question and it may continue with more questions, as if writing a quiz. A **chat (instruction-tuned) model** starts as a base model and is further trained on conversations and human preferences, so it *answers* and follows instructions (topic 1.8). The API you call day to day (Claude, GPT, Gemini) is a chat model.

**Gotchas.**

- It does not look things up or "know" it is right: it produces likely text. Facts come from patterns in training data (more in 1.11).
- It has no memory between calls. The only "memory" is the tokens you send each time (the context window).
- Output is generated one token at a time, left to right; it cannot go back and edit earlier tokens (which is why reasoning models think first, 1.12).
- "Training data cutoff" means it knows nothing newer unless you provide it or give it tools.

**UiPath bridge.** Unlike a classic RPA activity, which is deterministic (same input, same output), an LLM call is probabilistic. You design workflows around that: validation, retries and human review.

**Interview angle.**

- *What is an LLM in one sentence?* A neural network trained to predict the next token given previous tokens; chat behaviour comes from additional instruction and preference tuning on top.
- *Base vs chat model?* A base model continues text, a chat model follows instructions. Chat models are fine-tuned from base models with supervised examples and preference optimisation.

## 1.2 Tokens and tokenization (BPE), why token counts matter

**The idea.** Models do not read characters or words; they read **tokens**, integer IDs for chunks of text. Tokenization exists because a fixed vocabulary of subword pieces is a good compromise: whole-word vocabularies cannot handle new words and typos, while character-level input makes sequences far too long.

**How it works.** The most common method is **Byte Pair Encoding (BPE)**. Start with single bytes or characters, count the most frequent adjacent pair in a large corpus, merge it into a new token, and repeat tens of thousands of times. The merge list is the tokenizer.

```
corpus words: l o w , l o w e r , l o w e s t
step 1: most frequent pair (l,o) -> "lo"
step 2: (lo,w) -> "low"
...
"lowest" -> ["low", "est"]   |   " tokenization" -> [" token", "ization"]   (about 3-4 characters per token in English)
```

Why counts matter: you are billed per token (input and output priced differently), the context window is measured in tokens, rate limits are in tokens per minute, and latency grows with tokens. Rule of thumb for English: 1 token ≈ 4 characters ≈ 0.75 words. Code, non-English text, numbers and emoji usually cost more tokens per character.

**Gotchas.**

- Each model family has its own tokenizer; the same text gives different counts across vendors, and a newer model version can change it. Anthropic notes newer Claude models use a newer tokenizer that produces noticeably more tokens for the same text, so recount against the model you will use.
- Leading spaces are part of tokens (" the" and "the" differ), which matters for prompt formatting.
- Models struggle with character-level tasks (counting letters in a word, reversing strings) partly because they see tokens, not letters.
- To count Claude tokens use the API's token counting endpoint (free, but an estimate); a local library like tiktoken only matches OpenAI-style tokenizers.

**UiPath bridge.** Token budgets play the role that Orchestrator rate limits and licence units play: a resource you must estimate and cap before running at volume.

**Interview angle.**

- *Why do LLMs struggle to count the letters in "strawberry"?* They see subword tokens, not individual characters, so character-level structure is not directly visible.
- *Your prompt is 3x more expensive than expected; where do you look?* Count tokens per component (system prompt, retrieved context, history, output), check for verbose formatting, repeated context and a non-English or code-heavy payload, then trim, summarise history or cache the stable prefix.

## 1.3 Neural network essentials: weights, loss, gradient descent, training vs inference

**The idea.** A neural network is a big function with adjustable numbers called **weights**. Training finds weights that make the function's outputs match the data. It exists because writing rules by hand for language is impossible; instead we show examples and let optimisation find the rules.

**How it works.** One training step:

```
1. forward pass : tokens -> network -> predicted probabilities for the next token
2. loss         : cross-entropy = -log(probability given to the CORRECT token)   (low = good)
3. backward pass: compute the gradient = how much each weight would change the loss
4. update       : weight = weight - learning_rate x gradient         (gradient descent)
repeat over billions of examples
```

If the model gave the right token probability 0.9, loss is -log(0.9) ≈ 0.11; if it gave 0.01, loss is ≈ 4.6. Gradient descent is just walking downhill on that loss surface. **Training** changes the weights (expensive, done once by the lab, needs many GPUs); **inference** uses frozen weights to produce outputs (what you do with an API). A model with N parameters stores N weights; "70B" means 70 billion of them.

**Gotchas.**

- The model does not learn from your API calls; weights are frozen at inference. Fine-tuning is a separate training job (Module 9).
- Low training loss is not the goal on its own: a model can memorise (overfit). We care about loss on unseen data.
- Learning rate too high diverges; too low is painfully slow.
- "Parameters" are not "facts"; knowledge is distributed across weights and cannot be edited by hand easily.

**UiPath bridge.** Training vs inference mirrors building vs running a process: you develop and tune once, then execute many times in Orchestrator with the definition frozen.

**Interview angle.**

- *Explain gradient descent to a non-ML engineer.* Measure how wrong the model is (loss), work out which direction each weight should move to reduce it (gradient), take a small step that way, repeat.
- *What's the difference between training and inference, and why does it matter for cost?* Training updates weights and costs millions in compute upfront; inference only runs the forward pass and is paid per request, so product cost is dominated by inference tokens.

## Hands-on (30–45 min): tokenize, then train a tiny next-token model

**Goal.** Tokenize real text and see token counts, then train a tiny next-token predictor by gradient descent in plain numpy and watch loss fall. This is the whole LLM recipe in miniature.

**Setup.** Python 3.10+, `pip install tiktoken numpy`. Optional Part C also needs `pip install openai python-dotenv` and your Gemini key in `.env` (free tier, no card).

**Part A — tokenization (1.2).**

```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

for s in ["Hello world", "tokenization", "strawberry", "नमस्ते दुनिया", "def add(a, b): return a + b", "2026-10-30"]:
    ids = enc.encode(s)
    print(f"{s!r:40} chars={len(s):3} tokens={len(ids):3}  {[enc.decode([i]) for i in ids]}")
```

Note how Hindi, code and numbers use more tokens per character than plain English.

**Part B — a bigram next-token model trained with gradient descent (1.1 and 1.3).**

```python
import numpy as np

text = "the cat sat on the mat . the dog sat on the log . the cat saw the dog ."
words = text.split()
vocab = sorted(set(words)); idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
X = np.array([idx[w] for w in words[:-1]])      # current token
Y = np.array([idx[w] for w in words[1:]])       # next token (the label)
W = np.zeros((V, V))                            # weights: row = current token, columns = logits for next token

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

n, lr = len(Y), 1.0
for step in range(201):
    P = softmax(W[X])                                   # forward pass
    loss = -np.log(P[np.arange(n), Y]).mean()           # cross-entropy loss
    G = P.copy(); G[np.arange(n), Y] -= 1; G /= n       # gradient of loss wrt logits
    dW = np.zeros_like(W); np.add.at(dW, X, G)          # backward pass
    W -= lr * dW                                        # gradient descent update
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss:.3f}")

print("initial loss should be ln(V) =", round(float(np.log(V)), 3))

# inference: weights frozen, just sample
rng = np.random.default_rng(0)
cur, out = idx["the"], ["the"]
for _ in range(8):
    p = softmax(W[[cur]])[0]
    cur = rng.choice(V, p=p); out.append(vocab[cur])
print(" ".join(out))
print({vocab[i]: round(float(p), 2) for i, p in enumerate(softmax(W[[idx['the']]])[0]) if p > 0.01})
```

**Part C (optional) — base vs chat intuition (1.1) and real Gemini token counts (1.2).** Compare tiktoken's count with the provider's `prompt_tokens` for the same text. It uses the `openai` package pointed at Google's OpenAI-compatible endpoint, so there is no Anthropic key. Put `GEMINI_API_KEY=...` and `LLM_MODEL=<id>` (for example `gemini-3.1-flash-lite`, no `models/` prefix) in `.env`, then save this next to it. Example run: the one-sentence prompt in `check_setup.py` gave tiktoken 11 vs Gemini 12, and the 20x fox sentence gave 201 vs 202. A gap of about one token is normal: the tokenizers differ and the provider adds a little message-format overhead.

```python
import os, tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env (GEMINI_API_KEY, LLM_MODEL)
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
)
MODEL = os.environ["LLM_MODEL"]   # the ID you put in .env
text = "The quick brown fox jumps over the lazy dog. " * 20
r = client.chat.completions.create(model=MODEL, max_tokens=1, messages=[{"role": "user", "content": text}])
print("tiktoken:", len(tiktoken.get_encoding("cl100k_base").encode(text)), "| Gemini:", r.usage.prompt_tokens)
```

**Done when:** (1) you can explain from Part A why Hindi and code cost more tokens; (2) Part B loss starts near ln(V) (about 2.4 for this corpus) and drops steadily, and the printed distribution after "the" shows cat, dog, mat and log with sensible probabilities (cat and dog highest); (3) you can say in two sentences which part of your script is training and which is inference. Bonus: set `lr` to 100 and then 0.001 and describe what happens to the loss.

## Go deeper (optional)

- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al.)](https://arxiv.org/abs/1508.07909) — the paper that brought BPE to neural NLP.
- [Anthropic docs: token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting) — how to count Claude tokens before you pay for them.
- [Karpathy, Let's Build the GPT Tokenizer (write-up on fast.ai)](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers) — builds BPE from scratch.

## Recall answers

No warm-up on Day 1. Tomorrow's warm-up will cover today's three topics: next-token prediction and base vs chat models, BPE tokenization and why token counts matter, and loss, gradient descent and training vs inference.
