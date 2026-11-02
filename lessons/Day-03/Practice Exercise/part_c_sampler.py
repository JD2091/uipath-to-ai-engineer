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
