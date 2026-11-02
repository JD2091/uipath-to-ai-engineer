# Week 1 Review — From raw text to a chosen token: the LLM pipeline end to end (2026-11-01)

Sunday review, no new topics. This week covered 1.1–1.9 (Days 1–3). Do the quiz cold (about 15 min), read the synthesis, then pick a weekend slot for the mini-project (1–2 hours).

## Quiz (10 questions)

1. **(MCQ, 1.1)** You send "Write a haiku about overdue invoices" to a *base* model with no chat template. What is the most likely result?
    a) A clean haiku  b) It continues the text as if it were part of a document, e.g. more prompt-like lines or a list  c) A polite refusal  d) An API error
2. **(Short, 1.2)** The same 1,000-word contract gives a different token count on two vendors, and the Gujarati version costs more tokens than the English one. Explain both.
3. **(MCQ, 1.3)** What happens to the model's weights during a normal chat API call?
    a) Updated by gradient descent from your messages  b) Nothing: only a forward pass runs, weights are frozen  c) Loss is computed on your prompt and backpropagated  d) The conversation is written into the weights as memory
4. **(MCQ, 1.4)** Which statement about embeddings is true?
    a) Cosine similarity compares direction and ignores vector length  b) Each dimension maps to one readable feature, like "dimension 17 = money"  c) Vectors from two different models can be compared if both have 768 dimensions  d) Static embeddings change with the surrounding sentence
5. **(Short, 1.5)** For a ticket classifier and for a reply generator, which transformer family fits each (encoder-only or decoder-only), and what masking difference explains why?
6. **(Short, 1.6)** In one line each: what are Q, K and V? Write the attention formula and say why it divides by √d_k.
7. **(MCQ, 1.6 + 1.7)** You double a prompt from 4k to 8k tokens. Roughly how much more work is the attention *score matrix* (QKᵀ)?
    a) The same  b) 2×  c) 4×  d) 8×
8. **(Short, 1.7)** With RoPE, why does a query at position 105 and a key at 102 give the same score as a query at 5 and a key at 2? And given "lost in the middle", where in a long prompt would you put the critical instruction?
9. **(Short, 1.8)** Name the three stages that turn a pretrained model into an assistant, and say in one sentence how DPO differs from classic RLHF with PPO.
10. **(Short, 1.9)** Logits for three candidate tokens are [2.0, 1.0, 0.0]. (a) What are the probabilities at T = 1? (b) With top-p = 0.9, which tokens survive? (c) What happens to the top token's share at T = 0.5 vs T = 2? (d) Why can T = 0 on a hosted API still give different outputs across runs?

## How the week connects

Every topic this week is one stage of a single pipeline that runs on every call. Text is split into **tokens** (1.2) and each token id is looked up as an **embedding** vector (1.4); because attention cannot see order, **position** is injected, nowadays usually by rotating queries and keys with RoPE (1.7). The **transformer** stack (1.5) then lets each token gather information from earlier tokens through **multi-head self-attention** (1.6), whose quadratic cost is why the **context window** is finite and long prompts are expensive. The final layer outputs a probability for every possible next token, which is all an LLM fundamentally is (1.1); those weights were learned by **gradient descent on next-token loss** (1.3) during pretraining, and then **post-training** (SFT, then preference tuning such as RLHF, DPO or Constitutional AI) taught the model to behave like an assistant without adding much new knowledge (1.8). Finally a **sampling** rule picks one token from that distribution (1.9), it is appended, and the loop repeats. Put that way, cost (tokens × calls), latency (one forward pass per output token), and quality problems such as hallucination and inconsistency each map to a specific stage, which is exactly where next week starts (1.10 KV cache, 1.11 hallucinations).

```
text ──► tokens ──► embeddings + positions ──► N × [ multi-head attention → MLP ] ──► logits ──► sampler ──► next token
 1.2       1.2           1.4       1.7                1.5 / 1.6                       1.1          1.9          │
                    weights learned by 1.3 (pretraining) and shaped by 1.8 (post-training)                     │
      ▲───────────────────────────────── append token, repeat until stop ◄──────────────────────────────────────┘
```

## Weekend mini-project (1–2 hours): a "glass-box" LLM inspector

**Goal.** Write one script, `inspect_llm.py`, that traces a single prompt through every stage of the pipeline above on a real small model and prints a short report. It reuses Day 3's SmolLM2 models (already in your Hugging Face cache) and your Day 3 sampler.

**Setup.** Same as Day 3: `pip install torch transformers` plus `tiktoken` and `numpy` from your requirements. No API key, runs on CPU. Put it in a new folder of your choice, for example `Week-1 project/`.

**Steps.**

1. **Tokens (1.2).** Tokenize the prompt with the SmolLM2 tokenizer and with tiktoken's `o200k_base`. Print both counts and SmolLM2's token pieces. Try one English sentence, one in Gujarati or Hindi, and one line of JSON.
2. **Architecture and window (1.5, 1.7).** Print layers, heads, hidden size, vocab size and `max_position_embeddings` from the model config.
3. **Attention (1.6).** Run one forward pass with `output_attentions=True`. Confirm the causal mask (every weight above the diagonal is about 0). For two heads in the last layer, print the 3 tokens the final position attends to most.
4. **Loss (1.3 + 1.7).** Pass `labels=input_ids` to get the next-token loss. Compare a normal sentence with the same words shuffled. The shuffled one should score worse, because order matters.
5. **Embeddings (1.4).** Mean-pool the last hidden state for 4 sentences (2 about invoices, 2 about weather) and print a cosine-similarity matrix. Note that mean-pooling a small decoder is a crude embedding; real embedding models come in Module 4.
6. **Next token and sampling (1.1, 1.9).** Print the top-10 next tokens with probabilities, then generate 20 tokens 3 times each at T = 0.2, 0.8 and 1.5 with your own sampler (top-p = 0.9).
7. **Post-training (1.8).** Ask the base and the instruct model the same question (instruct via its chat template) and print both answers side by side.

**Starter code.**

```python
import numpy as np, torch, tiktoken
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "The robot opened the invoice and checked the total amount"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, attn_implementation="eager")  # eager = attentions are returned
model.eval()

# 1) Tokens (1.2)
ids = tok(PROMPT, return_tensors="pt").input_ids
gpt = tiktoken.get_encoding("o200k_base")
print(f"SmolLM2: {ids.shape[1]} tokens | o200k_base: {len(gpt.encode(PROMPT))} tokens")
print(tok.convert_ids_to_tokens(ids[0]))

# 2) Architecture + context window (1.5, 1.7)
cfg = model.config
print({k: getattr(cfg, k) for k in
       ["num_hidden_layers", "num_attention_heads", "hidden_size", "vocab_size", "max_position_embeddings"]})

# 3 + 4) One forward pass: logits, attentions, hidden states and loss (1.1, 1.3, 1.6)
with torch.no_grad():
    out = model(ids, labels=ids, output_attentions=True, output_hidden_states=True)
print("next-token loss:", round(out.loss.item(), 3))
att = out.attentions[-1][0]                 # last layer -> shape (heads, seq, seq)
upper = torch.triu(att, diagonal=1)
print("max weight above diagonal (should be ~0):", upper.max().item())
# TODO: for heads 0 and 1, print the 3 tokens the LAST position attends to most (att[h, -1].topk(3))
# TODO: write loss_of(text) and compare PROMPT with a word-shuffled copy

# 5) Embeddings (1.4)
def embed(text):
    with torch.no_grad():
        hs = model(tok(text, return_tensors="pt").input_ids, output_hidden_states=True).hidden_states[-1][0]
    v = hs.mean(dim=0).numpy()
    return v / np.linalg.norm(v)
# TODO: embed 4 sentences, print the 4x4 cosine matrix (dot products of unit vectors)

# 6) Next token + your sampler (1.1, 1.9)
logits = out.logits[0, -1].numpy()
top = np.argsort(logits)[::-1][:10]
probs = np.exp(logits - logits.max()); probs /= probs.sum()
for i in top:
    print(f"{tok.decode([int(i)])!r:>14}  {probs[i]:.3f}")

def sample(logits, temperature=1.0, top_p=None, rng=np.random.default_rng()):
    # TODO: paste your Day 3 sampler: divide by T, softmax, nucleus cut, renormalise, draw
    ...

# TODO: loop: sample a token, append it to ids, run the model again, 20 times, for T in (0.2, 0.8, 1.5)

# 7) Base vs instruct (1.8)
# TODO: load INSTRUCT, build input with tok_i.apply_chat_template([{"role": "user", "content": q}],
#       add_generation_prompt=True, return_tensors="pt"), generate 40 tokens greedily from both, print side by side
```

**Done looks like.** One run prints all seven sections without errors; the above-diagonal attention maximum is about 0; the shuffled sentence has higher loss; the invoice sentences are more similar to each other than to the weather ones (if not, note why mean-pooling may fail); T = 0.2 runs look alike while T = 1.5 runs drift; and the base model rambles where the instruct model answers. Add a short `NOTES.md` with five observations in your own words. That file is good material for your first portfolio write-up (12.6).

## Answers

1. **b.** A base model only continues text. With no chat format it often treats your line as part of a document and continues it (more prompts, a list, a forum post), though it may sometimes produce a haiku by chance.
2. Each vendor uses its **own tokenizer** with a different learned vocabulary, so the same text splits differently. BPE merges are learned from training data that is mostly English, so English words become single tokens while Gujarati script is split into many smaller pieces, which means more tokens, more cost and less text per context window.
3. **b.** Inference is a forward pass only. Weights change only in a separate training or fine-tuning job; any "memory" is just the history you resend.
4. **a.** Meaning lives in directions spread across many dimensions; vectors from different models live in different spaces even with equal size; static embeddings (word2vec, GloVe) are fixed per word.
5. Classifier: **encoder-only** (e.g. BERT-style), which sees the whole input in both directions and gives strong representations for classification and embeddings. Generator: **decoder-only**, which uses a **causal mask** so each position sees only earlier tokens, matching left-to-right generation.
6. **Q** = what this token is looking for; **K** = what each token offers to be matched on; **V** = the content a token passes on if attended to. Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V. Dot products grow with the dimension d_k; without scaling, softmax saturates to near one-hot and gradients vanish.
7. **c.** Every token scores against every other token, so the score matrix grows with n²: 2× tokens gives about 4× the attention-score work. (Other parts of the model grow linearly, so the whole call grows by less than 4×.)
8. RoPE rotates q and k by angles proportional to their positions, so their dot product depends only on the **offset** (m − n = 3 in both cases), not on absolute positions. Models use facts at the start and end of a long prompt better than ones in the middle, so put critical instructions at the **start** (system prompt) and/or restate them at the **end**, next to the question.
9. **Pretraining** (next-token prediction on huge text), **supervised fine-tuning / instruction tuning** on demonstrations, then **preference tuning** (RLHF, RLAIF, Constitutional AI, DPO). RLHF trains a separate reward model and optimises the policy with RL (PPO) plus a KL penalty; DPO skips the reward model and RL loop and optimises directly on preferred-vs-rejected pairs with a classification-style loss.
10. (a) softmax([2, 1, 0]) ≈ **[0.665, 0.245, 0.090]**. (b) Cumulative: 0.665, then 0.910 ≥ 0.9, so the **top 2** survive and are renormalised to about [0.73, 0.27]. (c) T = 0.5 sharpens it: logits become [4, 2, 0], giving the top token ≈ **0.87**; T = 2 flattens it: [1, 0.5, 0] gives ≈ **0.51**. (d) Batching and non-associative floating-point maths on GPUs can flip near-tied choices, and providers update infrastructure or models; one flipped token changes everything after it.
