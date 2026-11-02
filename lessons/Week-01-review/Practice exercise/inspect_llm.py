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
# --- ADDED: solution for the TODO above ---
toks = tok.convert_ids_to_tokens(ids[0])
for h in (0, 1):
    w, idx = att[h, -1].topk(3)
    print(f"head {h}: last token attends to", [(toks[i], round(x.item(), 3)) for x, i in zip(w, idx)])

# TODO: write loss_of(text) and compare PROMPT with a word-shuffled copy
# --- ADDED: solution for the TODO above ---
def loss_of(text):
    t = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        return model(t, labels=t).loss.item()

words = PROMPT.split()
shuffled = " ".join(np.random.default_rng(0).permutation(words))
print(f"loss normal: {loss_of(PROMPT):.3f} | shuffled ({shuffled!r}): {loss_of(shuffled):.3f}")

# 5) Embeddings (1.4)
def embed(text):
    with torch.no_grad():
        hs = model(tok(text, return_tensors="pt").input_ids, output_hidden_states=True).hidden_states[-1][0]
    v = hs.float().mean(dim=0).numpy()      # .float(): the model loads in bf16, which numpy can't take
    return v / np.linalg.norm(v)
# TODO: embed 4 sentences, print the 4x4 cosine matrix (dot products of unit vectors)
# --- ADDED: solution for the TODO above ---
sentences = ["The invoice is overdue.", "Payment for this invoice is late.",
             "The robot clicked the Submit button.", "I love this vendor."]
E = np.stack([embed(s) for s in sentences])
np.set_printoptions(precision=2, suppress=True)
print(E @ E.T)                                  # unit vectors -> dot product = cosine

# 6) Next token + your sampler (1.1, 1.9)
logits = out.logits[0, -1].float().numpy()
top = np.argsort(logits)[::-1][:10]
probs = np.exp(logits - logits.max()); probs /= probs.sum()
for i in top:
    print(f"{tok.decode([int(i)])!r:>14}  {probs[i]:.3f}")

def sample(logits, temperature=1.0, top_p=None, rng=np.random.default_rng()):
    # TODO: paste your Day 3 sampler: divide by T, softmax, nucleus cut, renormalise, draw
    # --- ADDED: solution for the TODO above ---
    z = np.asarray(logits, dtype=np.float64) / temperature
    p = np.exp(z - z.max()); p /= p.sum()
    if top_p:
        order = np.argsort(-p)
        keep = order[: np.searchsorted(np.cumsum(p[order]), top_p) + 1]   # smallest set reaching top_p
        mask = np.zeros_like(p, dtype=bool); mask[keep] = True
        p = np.where(mask, p, 0.0); p /= p.sum()
    return int(rng.choice(len(p), p=p))

# TODO: loop: sample a token, append it to ids, run the model again, 20 times, for T in (0.2, 0.8, 1.5)
# --- ADDED: solution for the TODO above ---
for T in (0.2, 0.8, 1.5):
    gen = ids.clone()
    for _ in range(20):
        with torch.no_grad():
            lg = model(gen).logits[0, -1].float().numpy()
        gen = torch.cat([gen, torch.tensor([[sample(lg, temperature=T, top_p=0.9)]])], dim=1)
    print(f"T={T}:", tok.decode(gen[0, ids.shape[1]:]))

# 7) Base vs instruct (1.8)
# TODO: load INSTRUCT, build input with tok_i.apply_chat_template([{"role": "user", "content": q}],
#       add_generation_prompt=True, return_tensors="pt"), generate 40 tokens greedily from both, print side by side
# --- ADDED: solution for the TODO above ---
tok_i = AutoTokenizer.from_pretrained(INSTRUCT)
model_i = AutoModelForCausalLM.from_pretrained(INSTRUCT).eval()
q = "What is the capital of France?"
chat = tok_i.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
if not torch.is_tensor(chat):          # newer transformers versions may return a BatchEncoding
    chat = chat["input_ids"]
raw = tok(q, return_tensors="pt").input_ids
for name, m, inp in (("base", model, raw), ("instruct", model_i, chat)):
    with torch.no_grad():
        o = m.generate(inp, max_new_tokens=40, do_sample=False)
    print(f"[{name}]", repr(tok.decode(o[0, inp.shape[1]:], skip_special_tokens=True)))
