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
np.set_printoptions(precision=2, suppress=True)
Q, K, V = X @ Wq, X @ Wk, X @ Wv
print("=== single head ===")
print("Q (n, d_k):\n", Q)
print("K (n, d_k):\n", K)
print("V (n, d_k):\n", V)
out, A = attention(Q, K, V)
print("attention weights (row = query token):\n", A)
print("rows sum to 1:", np.allclose(A.sum(axis=1), 1), "| output shape:", out.shape)

# --- multi-head ---
heads = []
for h in range(n_heads):
    Wq, Wk, Wv = (rng.normal(size=(d_model, d_k)) / np.sqrt(d_model) for _ in range(3))
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    print(f"\n=== head {h} ===")
    print("Q (n, d_k):\n", Q)
    print("K (n, d_k):\n", K)
    print("V (n, d_k):\n", V)
    o, A_h = attention(Q, K, V)
    heads.append(o)
    print(f"head {h}: token 'total' attends most to '{tokens[int(A_h[-1].argmax())]}'")
Wo = rng.normal(size=(d_model, d_model)) / np.sqrt(d_model)
mha = np.concatenate(heads, axis=-1) @ Wo                   # (n, d_model): back to model width
print("multi-head output shape:", mha.shape)
