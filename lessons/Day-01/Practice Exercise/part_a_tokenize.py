import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

for s in ["Hello world", "tokenization", "strawberry", "नमस्ते दुनिया", "def add(a, b): return a + b", "2026-10-30"]:
    ids = enc.encode(s)
    print(f"{s!r:40} chars={len(s):3} tokens={len(ids):3}  {[enc.decode([i]) for i in ids]}")
