import numpy as np

text = "the cat sat on the mat . the dog sat on the log . the cat saw the dog ."
words = text.split()
vocab = sorted(set(words)); idx = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
X = np.array([idx[w] for w in words[:-1]])      # current token
Y = np.array([idx[w] for w in words[1:]])       # next token (label)
W = np.zeros((V, V))                            # row = current token, columns = logits for next token

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

n, lr = len(Y), 1.0
for step in range(201):
    P = softmax(W[X])                                   # forward pass
    loss = -np.log(P[np.arange(n), Y]).mean()           # cross-entropy loss
    G = P.copy(); G[np.arange(n), Y] -= 1; G /= n       # gradient wrt logits
    dW = np.zeros_like(W); np.add.at(dW, X, G)          # backward pass
    W -= lr * dW                                        # gradient descent update
    if step % 50 == 0:
        print(f"step {step:3d}  loss {loss:.3f}")

print("initial loss should be ln(V) =", round(float(np.log(V)), 3))

# inference: weights frozen, just sample
rng = np.random.default_rng(0)
cur, out = idx["the"], ["the"]
for _ in range(8):
    p = softmax(W[[cur]])[0]
    cur = rng.choice(V, p=p); out.append(vocab[cur])
print(" ".join(out))
print({vocab[i]: round(float(p), 2) for i, p in enumerate(softmax(W[[idx['the']]])[0]) if p > 0.01})
