"""Day 12: similarity metrics, a from-scratch IVF index, and a real vector DB (Chroma).
Run:  python day12_vector_search.py a|b|c|all"""
import sys
import time

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOCS = [
    ("Invoice INV-2060 was paid on 3 November by bank transfer.", "finance"),
    ("The supplier asked us to update their bank account details.", "finance"),
    ("Payment for the October bill is scheduled for next Friday.", "finance"),
    ("Our VPN keeps disconnecting every ten minutes.", "it"),
    ("Please reset my password for the HR portal.", "it"),
    ("The laptop screen flickers after the latest driver update.", "it"),
    ("New joiners must finish safety training in week one.", "hr"),
    ("Annual leave requests need manager approval in the portal.", "hr"),
]
QUERY = "When will our bill be paid?"


# ---------- embeddings (same local model as Day 11) ----------
_tok = _model = None
def embed(texts, normalise=True):
    global _tok, _model
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
        _model = AutoModel.from_pretrained(EMBED_MODEL).eval()
    batch = _tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        tokens = _model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    vecs = ((tokens * mask).sum(1) / mask.sum(1)).numpy()
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True) if normalise else vecs


# ---------- Part A: the three metrics ----------
def cosine(q, M): return (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))
def dot(q, M): return M @ q
def euclid(q, M): return np.linalg.norm(M - q, axis=1)

def ranking(scores, higher_is_better=True):
    return list(np.argsort(-scores if higher_is_better else scores))

def part_a():
    texts = [d for d, _ in DOCS]
    raw = embed(texts, normalise=False)
    q_raw = embed([QUERY], normalise=False)[0]
    print("Vector lengths before normalising:", np.round(np.linalg.norm(raw, axis=1), 2))

    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    q = q_raw / np.linalg.norm(q_raw)
    r_cos, r_dot, r_l2 = ranking(cosine(q, unit)), ranking(dot(q, unit)), ranking(euclid(q, unit), False)
    print("\nUnit vectors -> same order for all three metrics?", r_cos == r_dot == r_l2)
    for i in r_cos[:3]:
        print(f"  cos={cosine(q, unit)[i]:.3f}  dot={dot(q, unit)[i]:.3f}  l2={euclid(q, unit)[i]:.3f}  {texts[i]}")
    # identity check: for unit vectors, ||a-b||^2 = 2 - 2*cos
    print("  ||a-b||^2 == 2-2cos ?", np.allclose(euclid(q, unit) ** 2, 2 - 2 * cosine(q, unit), atol=1e-5))

    # Break it: make the best off-topic (IT) doc 20x longer. Same direction = same meaning.
    it_docs = [i for i, (_, dept) in enumerate(DOCS) if dept == "it"]
    j = max(it_docs, key=lambda i: cosine(q, unit)[i])
    skewed = unit.copy(); skewed[j] *= 20
    print(f"\nOff-topic doc scaled x20 (cos to query = {cosine(q, unit)[j]:.3f}): {texts[j]}")
    print("  top by cosine:", texts[ranking(cosine(q, skewed))[0]])
    print("  top by dot   :", texts[ranking(dot(q, skewed))[0]])
    print("  top by L2    :", texts[ranking(euclid(q, skewed), False)[0]])


# ---------- Part B: exact search vs a hand-built IVF index ----------
def make_corpus(n=50_000, d=128, centres=100, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.normal(size=(centres, d))
    X = c[rng.integers(0, centres, n)] + 1.5 * rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = c[rng.integers(0, centres, 200)] + 1.5 * rng.normal(size=(200, d))
    return X.astype(np.float32), (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)

def exact_topk(X, q, k=10):
    s = X @ q
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]

class IVF:
    def __init__(self, X, nlist=200):
        from sklearn.cluster import MiniBatchKMeans
        t = time.perf_counter()
        km = MiniBatchKMeans(n_clusters=nlist, batch_size=4096, n_init=3, random_state=0).fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.lists = [np.where(km.labels_ == i)[0] for i in range(nlist)]
        self.X = X
        print(f"IVF built: {nlist} lists in {time.perf_counter() - t:.1f}s")

    def search(self, q, k=10, nprobe=4):
        nearest = np.argsort(-(self.centroids @ q))[:nprobe]       # 1) pick closest clusters
        cand = np.concatenate([self.lists[i] for i in nearest])    # 2) only scan their members
        s = self.X[cand] @ q
        top = np.argsort(-s)[:k]
        return cand[top], len(cand)

def part_b():
    X, Q = make_corpus()
    k = 10
    t = time.perf_counter()
    truth = [set(exact_topk(X, q, k)) for q in Q]
    exact_ms = (time.perf_counter() - t) * 1000 / len(Q)
    print(f"Exact (brute force) over {len(X):,} vectors: {exact_ms:.2f} ms/query, recall 1.000")

    ivf = IVF(X, nlist=200)
    print(f"{'nprobe':>6} {'recall@10':>10} {'ms/query':>9} {'% scanned':>10}")
    for nprobe in [1, 2, 4, 8, 16, 32]:
        t = time.perf_counter(); hits = scanned = 0
        for q, gt in zip(Q, truth):
            ids, n = ivf.search(q, k, nprobe)
            hits += len(gt & set(ids)); scanned += n
        ms = (time.perf_counter() - t) * 1000 / len(Q)
        print(f"{nprobe:>6} {hits / (k * len(Q)):>10.3f} {ms:>9.2f} {100 * scanned / (len(Q) * len(X)):>9.1f}%")


# ---------- Part C: the same idea in a real vector DB ----------
def part_c():
    try:
        import chromadb
    except ImportError:
        print("pip install chromadb   (if no wheel exists for your Python version, skip Part C)")
        return
    client = chromadb.PersistentClient(path="./chroma_day12")
    col = client.get_or_create_collection(
        name="day12_docs",
        configuration={"hnsw": {"space": "cosine", "ef_search": 50}},
    )
    texts = [d for d, _ in DOCS]
    col.upsert(
        ids=[f"doc-{i}" for i in range(len(DOCS))],
        documents=texts,
        embeddings=embed(texts).tolist(),            # bring your own vectors: same model for docs and queries
        metadatas=[{"dept": dept} for _, dept in DOCS],
    )
    q = embed([QUERY]).tolist()
    for where in [None, {"dept": "it"}]:
        res = col.query(query_embeddings=q, n_results=3, where=where)
        print(f"\nwhere={where}")
        for doc, dist in zip(res["documents"][0], res["distances"][0]):
            print(f"  distance={dist:.3f} (similarity={1 - dist:.3f})  {doc}")


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("a", part_a), ("b", part_b), ("c", part_c)]:
        if part in (name, "all"):
            print(f"\n===== Part {name.upper()} ====="); fn()
