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
