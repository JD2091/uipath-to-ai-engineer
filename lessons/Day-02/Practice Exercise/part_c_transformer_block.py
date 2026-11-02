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
