# Day 2 — From vectors to attention: how a transformer reads (2026-10-30)

Today's thread: every token becomes a **vector** (1.4), a **transformer** is a stack of blocks that refine those vectors (1.5), and **self-attention** is the step inside each block where tokens read from each other (1.6). Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. You send a base model the text "What is the capital of France?". What kind of output should you expect, and why?
2. Roughly how many tokens is a 3,000-word English document, and why would the same content in Hindi cost more?
3. The model gives the correct next token probability 0.5. What is the cross-entropy loss, and does your API call change the model's weights?

## 1.4 Embeddings intuition: meaning as vectors

**The idea.** A token ID is just an integer; 4,812 is not "closer" to 4,813 in meaning. An **embedding** turns each token into a list of numbers (a vector, say 4,096 of them) placed so that tokens used in similar ways end up near each other. It exists because a neural network can only do arithmetic, and it needs an input where distance carries meaning.

**How it works.** The model holds an embedding matrix with one learned row per vocabulary token. Looking up a token means taking its row. Nobody hand-sets these numbers; they are weights learned by gradient descent (1.3), like every other weight.

```
token id 2834 ("invoice") -> row 2834 of E (V x d) -> [0.12, -0.80, 0.33, ... d numbers]

similarity = cosine(a, b) = (a · b) / (|a| |b|)       1.0 same direction, ~0 unrelated
  "invoice" vs "bill"    high
  "invoice" vs "giraffe" low
```

There are two uses you will meet. Inside an LLM, the input embeddings are only the starting point; each layer rewrites them using context, so "bank" in "river bank" and "bank account" ends up as different vectors. Separately, **embedding models** turn a whole sentence or document into one vector for semantic search; that is the basis of RAG (Modules 4 and 5).

**Gotchas.**

- Single dimensions are not human-readable features ("dimension 17 = money"); meaning lives in directions across many dimensions.
- Vectors from different models live in different spaces. Never compare an embedding from model A with one from model B, and re-embed everything when you switch models.
- Similar is not the same as relevant or true. "I love this vendor" and "I do not love this vendor" often score as very similar because they share topic and words.
- The famous king − man + woman ≈ queen trick comes from older static word vectors (word2vec); it is a nice intuition, not a reliable property of modern models.

**UiPath bridge.** Context Grounding in UiPath indexes your documents as embeddings so an agent can find passages by meaning rather than by exact keyword.

**Interview angle.**

- *What is an embedding and why use cosine similarity?* A learned vector representation where semantic similarity becomes geometric closeness; cosine compares direction and ignores vector length, which mostly reflects things like frequency rather than meaning.
- *Static vs contextual embeddings?* Static gives one vector per word regardless of context (word2vec, GloVe); contextual vectors come out of transformer layers and change with the surrounding text.

## 1.5 Transformer architecture overview

**The idea.** The transformer (Vaswani et al., 2017) is the network design behind every major LLM. Before it, recurrent networks (RNNs, LSTMs) read text one token at a time, which made training slow and long-range connections weak. The transformer processes all positions of a sequence in parallel during training and lets any token draw directly on any earlier token, which is what made training on internet-scale data practical.

**How it works.** A modern LLM is a decoder-only transformer:

```
tokens -> embedding lookup (+ position info, topic 1.7)
       -> [ Block 1 ] -> [ Block 2 ] -> ... -> [ Block N ]      (N is often dozens)
            each block:
              x = x + Attention(norm(x))   tokens exchange information   (1.6)
              x = x + MLP(norm(x))         each token processed on its own
       -> final norm -> unembedding (d -> V) -> logits -> softmax -> next-token probabilities
```

Think of `x` as a **residual stream**: one vector per token flowing up through the stack, and each block adds an update to it rather than replacing it. The original 2017 paper had an encoder (reads the input) and a decoder (writes the output) for translation. Today you mostly see decoder-only models (GPT, Claude, Gemini, Llama) for generation and encoder-only models (BERT-style) for classification and embeddings.

**Gotchas.**

- The model produces next-token logits at every position in one pass. Training uses all of them; generation only uses the last one.
- A causal mask stops each position from seeing future tokens, which is what makes it a next-token predictor.
- Most parameters sit in the MLP (feed-forward) layers, not in attention.
- "Transformer" is an architecture, not a model. Models differ in size, data, tuning and tweaks such as normalization placement or mixture-of-experts layers.

**UiPath bridge.** The residual stream is like a shared arguments dictionary passed through a chain of invoked workflows, where each workflow reads it and adds to it rather than replacing it.

**Interview angle.**

- *Why did transformers replace RNNs?* They parallelise across the sequence during training and give every token a direct path to every other token, so they train faster on GPUs and handle long-range dependencies better.
- *Encoder-only vs decoder-only?* Encoder-only sees the whole input in both directions and suits classification and embeddings; decoder-only uses a causal mask and suits generation.

## 1.6 Self-attention and multi-head attention

**The idea.** In "The bot read the invoice because it was overdue", the vector for "it" needs to pull in information from "invoice". **Self-attention** is the mechanism that lets each token decide which other tokens matter to it and blend in their information. It exists because meaning depends on context, and fixed windows or step-by-step memory handle that badly.

**How it works.** Each token's vector is multiplied by three learned matrices to get a **query** (what am I looking for?), a **key** (what do I contain?) and a **value** (what do I hand over if chosen?). M is the causal mask (0 for allowed positions, −∞ for future ones).

```latex
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V
```

```
scores[i][j] = q_i · k_j / sqrt(d_k)        how well token i's question matches token j's label
mask        : set scores where j > i to -inf (no peeking at future tokens)
weights     = softmax(row)                   each row sums to 1
output_i    = sum_j weights[i][j] * v_j      a blend of earlier tokens' values

"it" row:  The 0.02 | bot 0.10 | read 0.05 | the 0.03 | invoice 0.71 | ... -> "it" mostly absorbs "invoice"
```

**Multi-head attention** runs several of these in parallel (often 8 to 128 heads), each with its own smaller Q, K and V matrices (d_k = d_model / heads). One head might track syntax, another which noun a pronoun refers to. The head outputs are concatenated and mixed by one more matrix, W_O, back to model width.

**Gotchas.**

- Cost grows with the square of sequence length: every token scores against every earlier token. That is a main reason long context is expensive (topic 1.10).
- Dividing by √d_k is not decoration. Without it, dot products grow with dimension, softmax saturates to near one-hot and gradients vanish.
- Attention weights are not a reliable explanation of why the model answered something; they are one internal signal among many.
- Many modern models share keys and values across groups of heads (grouped-query attention) to cut memory at inference; you will see why in 1.10.

**UiPath bridge.** Attention behaves like a fuzzy Lookup Data Table: the query is the lookup value and the keys are the lookup column, but instead of returning one matching row it returns a weighted blend of all rows' values.

**Interview angle.**

- *Explain Q, K and V.* They are three learned projections of each token: the query is what the token is looking for, the key is what it offers for matching, the value is the content it passes on. Scores come from query·key; outputs are score-weighted sums of values.
- *What is the complexity of self-attention, and why scale by √d_k?* O(n²·d) in sequence length n, because every pair of tokens is scored. Scaling keeps the dot products in a range where softmax still has useful gradients.

## Hands-on (30–45 min): embed sentences, then build attention and a transformer block in numpy

**Goal.** See real embeddings cluster by meaning, then write the attention equation and a full transformer block yourself, ending with a next-token distribution. Weights are random, so the predictions are meaningless; the point is the shapes and the data flow.

**Setup.** No new packages: `numpy`, `openai` and `python-dotenv` from Day 1. Part A uses your Gemini key through the same OpenAI-compatible endpoint. Optionally add `EMBED_MODEL=gemini-embedding-2` to `.env` (that is the default in the script). If your key cannot use it, `pip install sentence-transformers` and swap in `SentenceTransformer("all-MiniLM-L6-v2").encode(sentences)` as a free local fallback.

**Part A — real embeddings (1.4), about 10 min.**

```python
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")

sentences = [
    "The invoice is overdue.",
    "Payment for this bill is late.",
    "The robot clicked the Submit button.",
    "The bot pressed submit on the form.",
    "I love this vendor.",
    "I do not love this vendor.",
]
r = client.embeddings.create(model=EMBED_MODEL, input=sentences)
E = np.array([d.embedding for d in r.data])
print("vectors:", E.shape)                                  # (6, dimensions)

E = E / np.linalg.norm(E, axis=1, keepdims=True)           # unit length -> dot product = cosine
S = E @ E.T
np.set_printoptions(precision=2, suppress=True)
print(S)
for i, s in enumerate(sentences):
    j = int(np.argsort(-S[i])[1])                          # nearest neighbour other than itself
    print(f"{s!r:42} -> {sentences[j]!r}  ({S[i, j]:.2f})")
```

**Part B — self-attention and multi-head attention from scratch (1.6), about 15 min.**

```python
import numpy as np
rng = np.random.default_rng(0)

tokens = ["the", "bot", "read", "the", "invoice", "total"]
n, d_model, n_heads = len(tokens), 16, 4
d_k = d_model // n_heads

X = rng.normal(size=(n, d_model))            # stand-in token embeddings (1.4): one row per token

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T / np.sqrt(Q.shape[-1])                 # (n, n): how much token i "looks at" token j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)        # causal: no peeking at future tokens
    scores = np.where(mask, -np.inf, scores)
    A = softmax(scores)                                     # each row sums to 1
    return A @ V, A

# --- single head ---
Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))
out, A = attention(X @ Wq, X @ Wk, X @ Wv)
np.set_printoptions(precision=2, suppress=True)
print("attention weights (row = query token):\n", A)
print("rows sum to 1:", np.allclose(A.sum(axis=1), 1), "| output shape:", out.shape)

# --- multi-head ---
heads = []
for h in range(n_heads):
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))
    o, A_h = attention(X @ Wq, X @ Wk, X @ Wv)
    heads.append(o)
    print(f"head {h}: token 'total' attends most to '{tokens[int(A_h[-1].argmax())]}'")
Wo = rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
mha = np.concatenate(heads, axis=-1) @ Wo                   # (n, d_model): back to model width
print("multi-head output shape:", mha.shape)
```

**Part C — a full transformer block, stacked, ending in next-token probabilities (1.5), about 15 min.**

```python
import numpy as np
rng = np.random.default_rng(1)

vocab = ["<bos>", "the", "bot", "read", "invoice", "total", "is", "due", "."]
V, d_model, n_heads, n_layers, d_ff = len(vocab), 32, 4, 2, 128
d_k = d_model // n_heads

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def init(*shape):
    return rng.normal(size=shape) / np.sqrt(shape[0])

E = rng.normal(size=(V, d_model)) * 0.1          # embedding matrix (1.4): one learned row per token
layers = [dict(Wq=init(d_model, d_model),
               Wk=init(d_model, d_model), Wv=init(d_model, d_model), Wo=init(d_model, d_model),
               W1=init(d_model, d_ff), W2=init(d_ff, d_model)) for _ in range(n_layers)]
W_unembed = init(d_model, V)                     # d_model -> one logit per vocab token

def mha(x, L):
    n = x.shape[0]
    split = lambda m: m.reshape(n, n_heads, d_k).transpose(1, 0, 2)          # (heads, n, d_k)
    Q, K, Vv = split(x @ L["Wq"]), split(x @ L["Wk"]), split(x @ L["Wv"])
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    scores = np.where(np.triu(np.ones((n, n), bool), 1), -np.inf, scores)
    out = softmax(scores) @ Vv                                               # (heads, n, d_k)
    return out.transpose(1, 0, 2).reshape(n, d_model) @ L["Wo"]

def block(x, L):
    x = x + mha(layer_norm(x), L)                                            # attention: tokens share info
    x = x + np.maximum(0, layer_norm(x) @ L["W1"]) @ L["W2"]                 # MLP: each token processed alone
    return x

ids = [vocab.index(w) for w in ["<bos>", "the", "bot", "read", "the", "invoice"]]
x = E[ids]                                       # (n, d_model)  (positions skipped until 1.7)
for L in layers:
    x = block(x, L)
logits = layer_norm(x) @ W_unembed               # (n, V): a next-token guess at EVERY position
p = softmax(logits[-1])                          # generation uses only the last position
print("next-token distribution after '... the invoice':")
for i in np.argsort(-p)[:5]:
    print(f"  {vocab[i]:8} {p[i]:.2f}")
n_params = E.size + W_unembed.size + sum(m.size for L in layers for m in L.values())
print("parameters:", n_params)
```

**Done when:** (1) in Part A the two invoice sentences and the two bot sentences are each other's nearest neighbours, and you have noted how close the love / do-not-love pair scores; (2) in Part B every attention row sums to 1 and everything above the diagonal is 0, and you can say why; (3) Part C prints a distribution and 25,152 parameters, and you can account for that number (embedding 9×32, unembedding 32×9, and per layer four 32×32 attention matrices plus a 32×128 and a 128×32 MLP). Bonus: remove the `/ np.sqrt(Q.shape[-1])` in Part B, multiply `X` by 10, and watch the attention rows collapse toward one-hot.

## Go deeper (optional)

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) — the original transformer paper; sections 3.1 to 3.2 cover scaled dot-product and multi-head attention.
- [Gemini API: OpenAI compatibility](https://ai.google.dev/api/compatibility) — the embeddings example used in Part A, including the current embedding model ID.

## Recall answers

1. A base model continues text rather than answering, so it may produce more quiz-style questions or a list of other capitals. It was trained only on next-token prediction; following instructions comes from later tuning.
2. About 4,000 tokens (1 token ≈ 0.75 English words). Hindi uses more tokens per character because BPE merges were learned mostly from English-heavy text, so Devanagari breaks into smaller pieces.
3. −ln(0.5) ≈ 0.69. No: inference uses frozen weights; only a separate training or fine-tuning job changes them.
