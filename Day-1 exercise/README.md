# Day 1: What a language model actually is

**Topics:** 1.1 What an LLM is (next-token prediction), 1.2 Tokens and tokenization (BPE), 1.3 Neural network essentials (weights, loss, gradient descent, training vs inference)  
**Date:** 2026-10-30

## What this exercise is about

A language model is a function that gives every possible next token a score. Tokens are what it reads, and gradient descent is how it learned the scores. This exercise makes all three concrete: split text into tokens, train a tiny next-token model from scratch in numpy, then check how a real provider counts tokens.

## Files

| File | What it does |
| --- | --- |
| `part_a_tokenize.py` | Runs six strings (English, a single long word, Hindi, code, a date) through `tiktoken`'s `cl100k_base` encoding and prints characters, token count and the pieces. |
| `part_b_bigram.py` | A next-word model over a 9-word vocabulary, trained with gradient descent in about 30 lines of numpy. No framework: forward pass, cross-entropy loss, gradient, update. |
| `part_c_compare_gemini_tokenizer.py` | Sends the same text to Gemini and compares the provider's `prompt_tokens` with `tiktoken`'s count. Needs your API key. |

## Run it

From the repo root, with your virtual environment active and `.env` set up (see the main README):

```bash
python "Day-1 exercise/part_a_tokenize.py"
python "Day-1 exercise/part_b_bigram.py"
python "Day-1 exercise/part_c_compare_gemini_tokenizer.py"
```

Parts A and B need no API key. Part C does, and uses a small number of tokens.

## What you should see

**Part B** (my run; the sampled sentence may differ slightly on your machine):

```text
step   0  loss 2.197
step  50  loss 0.906
step 100  loss 0.717
step 150  loss 0.659
step 200  loss 0.632
initial loss should be ln(V) = 2.197
the dog . sat dog sat on the log
{'cat': 0.32, 'dog': 0.32, 'log': 0.16, 'mat': 0.16}
```

## What to notice

1. **The starting loss is `ln(9) = 2.197`.** Before training, the model is guessing uniformly among 9 tokens. Loss is just "how surprised the model is by the real next token", and the baseline is the surprise of a fair 9-sided die.
2. **The loss never reaches zero, and that is correct.** After "the", the text continues with cat, dog, mat or log, so no model can be sure. The final probabilities after "the" (cat 0.32, dog 0.32, log 0.16, mat 0.16) are close to how often each word followed "the" in the text. With a lookup-table model, training amounts to learning those frequencies. A real LLM does the same job but computes the scores from the whole context instead of one previous word.
3. **Training and inference are different phases.** The loop with `W -= lr * dW` is training. The sampling loop at the end only reads `W`. The weights are frozen and the model just draws tokens.
4. **Part A: look at the pieces.** Compare how "strawberry" is split, and compare the tokens-per-character of the Hindi string with "Hello world". The model never sees letters, only these pieces, which is why letter-level questions are harder for it than they look.
5. **Part C: the two counts differ.** Token counts depend on the tokenizer, so you cannot reuse one model's count to budget another's.

## UiPath bridge

Training vs inference is like publishing a package version vs running jobs on it: after publishing, the logic is frozen. **Where it breaks:** a published workflow returns the same output for the same input, while sampling from a language model need not.

More in the [RPA ↔ AI Rosetta Stone](../RPA-AI-ROSETTA.md).

## Next

Day 2 moves from tokens to meaning: embeddings, then the transformer architecture.
