import numpy as np
rng = np.random.default_rng(0)
d = 8

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True); e = np.exp(z); return e / e.sum(axis=-1, keepdims=True)

def attend(X):                                    # attention without a causal mask, Q = K = V = X for brevity
    return softmax(X @ X.T / np.sqrt(X.shape[-1])) @ X

# 1) Without position info, attention cannot see word order
X = rng.normal(size=(4, d))                       # 4 token embeddings
perm = [2, 0, 3, 1]                               # shuffle the "sentence"
print("no positions -> shuffled input just gives shuffled output:",
      np.allclose(attend(X)[perm], attend(X[perm])))

# 2) Sinusoidal positional encoding (original 2017 transformer): add a position vector to each embedding
def sinusoidal(n_pos, d):
    pos = np.arange(n_pos)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (10000 ** (i / d))
    PE = np.zeros((n_pos, d)); PE[:, 0::2] = np.sin(ang); PE[:, 1::2] = np.cos(ang)
    return PE
PE = sinusoidal(4, d)
print("with positions  -> order now changes the result:   ",
      not np.allclose(attend(X + PE)[perm], attend(X[perm] + PE)))

# 3) RoPE: rotate q and k by an angle proportional to their position
def rope(x, pos, base=10000):
    freqs = base ** (-np.arange(0, len(x), 2) / len(x))
    ang = pos * freqs
    x1, x2 = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2] = x1 * np.cos(ang) - x2 * np.sin(ang)
    out[1::2] = x1 * np.sin(ang) + x2 * np.cos(ang)
    return out

q, k = rng.normal(size=d), rng.normal(size=d)
for m, n in [(5, 2), (105, 102), (5005, 5002), (5, 4), (50, 2)]:
    print(f"query at {m:>4}, key at {n:>4} (offset {m-n:>2}): score = {rope(q, m) @ rope(k, n):+.4f}")
