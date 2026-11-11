# Day 12 — Measure closeness, store the vectors, find neighbours fast (2026-11-11)

Yesterday you turned text into vectors (4.1). Today answers the three questions that come next. How do you score "close"? (4.2, similarity metrics.) Where do millions of vectors live? (4.3, vector databases.) How do you search them in milliseconds without comparing against every one? (4.4, ANN indexes.) The hands-on scores real embeddings with all three metrics, builds an IVF index from scratch and measures its recall/speed curve, then does the same job in Chroma. Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. Why do you log the model id next to the prompt version on every call?
2. What are the three parts of Simon Willison's "lethal trifecta"?
3. You switch from one embedding model to another. Why can't you keep the old vectors and embed only new documents with the new model?

## 4.2 Similarity metrics

**The idea.** An embedding is useful only if you can say how close two vectors are. A similarity metric is that ruler. Pick the wrong one, or mix up "similarity" with "distance", and your search quietly returns the wrong documents. The good news: with normalised embeddings, the three common metrics agree on the ranking.

**How it works.** For vectors a and b:

| Metric | Formula (in words) | Range | Better is | Ignores length? |
| --- | --- | --- | --- | --- |
| Dot product | sum of a[i] × b[i] | any number | higher | No: longer vectors score higher |
| Cosine similarity | dot product ÷ (length of a × length of b) | −1 to 1 | higher | Yes: only the angle counts |
| Euclidean (L2) distance | square root of the sum of (a[i] − b[i])² | 0 to ∞ | lower | No |

A 2-D picture:

```
        ▲
        │      b (long)          cos(a,b) = cos(a,c)   (same angle)
        │     ╱                  dot(a,b) > dot(a,c)    (b is longer)
        │    ╱  c (short)        L2(a,b)  > L2(a,c)     (b sits further away)
        │   ╱ ╱
        │  ╱╱____ a
        └──────────────▶
```

The key identity: if both vectors have length 1, then L2² = 2 − 2 × cosine, and dot = cosine. So on unit vectors all three produce the **same order**. That's why most pipelines normalise once at write time and then use the dot product, which is the cheapest to compute. You saw `vecs / norm` at the end of yesterday's `embed()` for exactly this reason.

Two more you'll meet: **Manhattan (L1)**, the sum of absolute differences (pgvector supports it), and **Hamming distance** on binary vectors, used with binary quantization to shrink indexes (Module 9.6).

**Similarity vs distance.** Databases usually return *distances* (lower is better). Chroma's cosine "distance" is 1 − cosine similarity. pgvector's `<=>` is cosine distance and its `<#>` is the *negative* inner product, so that "order by ascending" works for every operator. Read the docs for each store before writing a threshold.

**Gotchas.**

- **Use the metric the model was trained for.** The model card says it (for example "cosine" or "dot product"). Most modern models output normalised vectors, so it rarely matters, but some don't.
- **Un-normalised dot product rewards length.** If vectors aren't normalised, a long vector can win on magnitude alone. Tonight's Part A shows this by scaling an off-topic document ×20.
- **Default metrics differ.** Chroma's default space is `l2`, not cosine. Set it explicitly when you create the collection.
- **Scores aren't probabilities.** A cosine of 0.62 doesn't mean "62% relevant", and the useful range differs by model (yesterday's gotcha). Calibrate thresholds on labelled pairs from your own data.

**UiPath bridge.** It's the same discipline as confidence thresholds in Document Understanding / IXP: a score only means something after you calibrate a cut-off per use case on real documents.

**Interview angle.**

- *Cosine similarity or dot product: which would you use?* If the vectors are normalised, they give identical rankings, so use dot product because it's cheaper. If they aren't, cosine ignores length and dot doesn't. Follow the metric the embedding model was trained with, and normalise once at ingestion.
- *Your search returns the least relevant documents first. What's a likely cause?* Treating a distance as a similarity (sorting the wrong way), for example `<#>` in pgvector returns negative inner product, or Chroma returns 1 − cosine.

## 4.3 Vector databases (pgvector, Chroma, Qdrant, Pinecone)

**The idea.** For 1,000 vectors, a NumPy array and `M @ q` is all you need (you did that yesterday). A vector database exists for everything around that one line once you go to production: persistence, fast approximate search over millions of vectors (4.4), inserts/updates/deletes while serving queries, filtering by metadata (tenant, department, date), hybrid keyword + vector search, replication, backups and access control.

**How it works.** Every vector DB stores the same kind of record:

```
record = { id: "inv-2060-chunk-3",
           vector: [0.021, -0.113, …] (384 floats),
           metadata: { tenant: "acme", dept: "finance", date: "2026-11-03" },
           document: "Invoice INV-2060 was paid on 3 November…" }

query(vector q, top_k=5, filter: tenant = "acme")
   → ANN index finds nearest candidates → filter applied → top 5 ids + scores
```

The four in the curriculum, compared (checked 2026-11-11):

| Store | What it is | Index types | Best fit |
| --- | --- | --- | --- |
| **pgvector** | Open-source extension for PostgreSQL. Vectors are a column type next to your normal tables. | HNSW, IVFFlat | You already run Postgres; you want SQL joins, transactions and one system to back up. |
| **Chroma** | Open-source, developer-first. Runs embedded in your Python process or as a server; also a hosted cloud. | HNSW | Prototypes, notebooks, small-to-medium apps. Tonight's Part C. |
| **Qdrant** | Open-source engine written in Rust; self-host or Qdrant Cloud. | HNSW (filter-aware), with quantization options | Heavy metadata filtering, multi-tenant apps, self-hosted at scale. |
| **Pinecone** | Fully managed, closed-source service. Serverless by default; "Dedicated Read Nodes" (launched late 2025) for steady high query loads. | Managed (you don't pick) | You want zero operations and pay per use. |

Others you'll see in job specs: Weaviate, Milvus, Elasticsearch/OpenSearch, Azure AI Search, MongoDB Atlas Vector Search. The concepts transfer.

A pgvector taste, because it's just SQL:

```sql
CREATE EXTENSION vector;
CREATE TABLE chunks (id bigserial PRIMARY KEY, tenant text, body text, embedding vector(384));
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
SELECT id, body, embedding <=> $1 AS distance      -- $1 = query vector
FROM chunks WHERE tenant = 'acme'
ORDER BY embedding <=> $1 LIMIT 5;
```

**Gotchas.**

- **Operator must match the index.** In pgvector a cosine index (`vector_cosine_ops`) only helps queries that use `<=>`. Use `<->` (L2) instead and Postgres silently falls back to a full scan.
- **Dimension limits.** pgvector indexes `vector` up to 2,000 dimensions and `halfvec` (16-bit floats) up to 4,000. A 3,072-dim model needs `halfvec` or truncated (Matryoshka) dimensions.
- **"We need a vector DB" is often premature.** Under a few hundred thousand vectors, pgvector or even a NumPy/FAISS file is enough. Pick on operations and filtering needs, not on benchmark charts.
- **Store the source text and the model name.** If you only keep vectors, you can't re-embed when you change models (yesterday's lesson) or show citations later (5.8).

**UiPath bridge.** A Context Grounding index is a managed vector store in the Pinecone sense: you point it at a source (a storage bucket or an Integration Service connector), it chunks, embeds and indexes, and agents query it. You just don't choose the engine or the knobs.

**Interview angle.**

- *How would you choose a vector database?* Start from constraints: data size and growth, query load, filtering and multi-tenancy needs, hybrid search, latency target, who operates it, data residency and cost. If the team already runs Postgres and the corpus is modest, pgvector keeps one system. Heavy filtering or very large scale pushes toward Qdrant/Milvus; no ops team pushes toward a managed service like Pinecone.
- *Why not just keep vectors in a NumPy array?* Fine for prototypes. In production you need persistence, concurrent updates, sub-linear search at millions of vectors, metadata filters, access control and backups.

## 4.4 ANN indexes (HNSW, IVF), recall vs speed

**The idea.** Exact nearest-neighbour search compares the query with every vector. That's linear in the corpus: 10 million chunks × 1,024 dims ≈ 10 billion multiply-adds per query, and about 40 GB of float32 to read. Approximate nearest neighbour (ANN) indexes look at a small, well-chosen fraction and accept that they'll sometimes miss a true neighbour. The quality measure is **recall@k**: of the true top-k (from exact search), what fraction did the index return? Every ANN knob trades recall for speed or memory.

**How it works.** Two families dominate.

**IVF (inverted file):** cluster first, then search only nearby clusters.

```
BUILD: k-means splits the vectors into nlist clusters (e.g. 1,000), each with a centroid
QUERY: 1) compare q with the 1,000 centroids
       2) keep the nprobe closest clusters (e.g. 10)
       3) exact search inside those ~1% of vectors
knobs: nlist (pgvector: lists), nprobe (pgvector: probes)
higher nprobe → better recall, slower
```

**HNSW (Hierarchical Navigable Small World):** a layered graph where each vector links to its near neighbours.

```
layer 2:  A ─────────────── F                 few nodes, long jumps
layer 1:  A ──── C ──── F ──── H
layer 0:  A─B─C─D─E─F─G─H─I─J─K               every node, short links
query: start at the top, greedily walk to the neighbour closest to q,
       drop a layer, repeat; at layer 0 keep a candidate list of size ef_search
knobs: M (links per node), ef_construction (build quality), ef_search (query effort)
```

It's like an expressway → main road → local street route: big jumps first, then fine steps (Malkov & Yashunin, 2016).

| | IVF | HNSW |
| --- | --- | --- |
| Speed / recall | Good; depends on nprobe | Usually the best at a given recall |
| Memory | Low (plus centroids) | Higher: stores the graph links |
| Build | Needs training (k-means) on representative data first | Slower build, no training step |
| Updates | Easy inserts; clusters drift as data changes, so rebuild | Inserts fine; deletes are harder |
| Default in | FAISS, pgvector (IVFFlat), Milvus | Chroma, Qdrant, pgvector, Weaviate |

Both are often combined with **quantization** (store vectors in 8 bits, 1 bit, or product-quantized codes) to cut memory, then re-score the top candidates with full vectors.

**Gotchas.**

- **Filters + ANN can return too few results.** If the index finds 40 candidates and the `WHERE tenant='acme'` filter then drops 38, you get 2 rows, not 5. pgvector 0.8 added *iterative index scans* (off by default) to keep searching; Qdrant builds filtering into the graph walk. This is 4.5's topic.
- **Measure recall on your data.** Vendor benchmarks use public datasets. Run exact search on a sample of your queries and compute recall@k for your settings. Tonight you do exactly that.
- **More effort isn't always used.** In Postgres, push `ef_search` or `probes` too high and the planner may decide a sequential scan is cheaper.
- **IVF built on too little data** gives bad centroids. Build after loading representative data, not on an empty table.

**UiPath bridge.** No direct equivalent you configure. This runs inside Context Grounding's index; knowing it explains why retrieval is fast but can occasionally miss a passage that exact search would find.

**Interview angle.**

- *Explain HNSW in one minute.* A multi-layer proximity graph. Upper layers are sparse with long links, the bottom layer has every vector. Search starts at the top, greedily moves toward the query, descends layer by layer, and at the bottom explores a candidate list of size ef_search. M and ef_construction set graph quality and memory; ef_search trades recall for latency at query time.
- *How do you tune an ANN index?* Fix a recall target (say 0.95 recall@10) and a latency budget. Build ground truth with exact search on a few hundred real queries, sweep ef_search or nprobe, and pick the cheapest setting that meets the target. Re-check after the data grows or changes.

## Hands-on (30–45 min): metrics, a home-made IVF index, and a real vector DB

**Goal.** See all three topics working on your machine. Part A scores real embeddings with dot, cosine and L2 and breaks the dot product on purpose (4.2). Part B builds an IVF index from scratch over 50,000 vectors and measures its recall@10 vs speed curve against exact search (4.4). Part C stores the same documents in Chroma with metadata and queries them with and without a filter (4.3).

**Setup.**

- Parts A and B use only what your `requirements.txt` already has (`numpy`, `scikit-learn`, `torch`, `transformers`). Part A reuses yesterday's local `all-MiniLM-L6-v2` (already downloaded). No API calls, no cost.
- Part C needs `pip install chromadb` in your `.venv`. Your venv runs Python 3.14; if pip finds no wheel for it, skip Part C tonight (or try it in a Python 3.12 venv). If you keep Chroma, add it to `requirements.txt` yourself.
- Save as `day12_vector_search.py` in your Day-12 exercise folder. Part C creates a `chroma_day12/` folder next to the script; add it to `.gitignore`.

```python
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
```

**Steps.**

1. Part A (10 min). Run `python day12_vector_search.py a`. Check: (a) raw vectors are *not* length 1 before normalising; (b) on unit vectors the three metrics give the same order, and the identity check prints `True`; (c) after scaling the IT document ×20, cosine still picks a finance sentence, dot product jumps to the IT sentence (if its cosine to the query is positive), and L2 pushes it further away. Write one sentence on why normalising at ingestion avoids this whole class of bug.
2. Part B (15 min). Run `python day12_vector_search.py b`. Find the smallest `nprobe` with recall@10 ≥ 0.95 and work out the speed-up over exact search (exact ms ÷ IVF ms). Then change `nlist` to 50 and to 800 and rerun. Fill in a small table: nlist, the nprobe you need for 0.95 recall, ms/query, % scanned. What happens to recall at `nprobe=1` as nlist grows, and why?
3. Part C (10 min, optional if Chroma won't install). Run `python day12_vector_search.py c`. Confirm the reported numbers are *distances* (1 − cosine) and that the `{"dept": "it"}` filter returns only IT documents, even though they're poor matches. Then delete the `chroma_day12` folder, change `"space"` to `"l2"` and rerun: the order stays the same (unit vectors), but the numbers change. Which would you rather put a threshold on?
4. Wrap-up (5 min). In a short `notes.md` in the exercise folder, answer: for a 2-million-chunk index with a 50 ms budget, which knob would you tune first in HNSW, and in IVF?

**What "done" looks like.** Part A prints `True` twice and shows dot product picking the scaled off-topic sentence. Part B prints a recall/latency table where recall climbs toward 1.000 as nprobe grows while still scanning only a small percentage of vectors (in a test run, nprobe=4 reached ≈ 0.99 recall scanning under 2% of the corpus). Part C prints two result lists with distances. Your notes table and 3–4 sentences are written.

## Go deeper (optional)

- Paper: [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320v1): the HNSW paper. Sections 1 and 4 give the layered-graph idea and the role of M and ef.
- [pgvector README (GitHub)](https://github.com/pgvector/pgvector): distance operators, HNSW vs IVFFlat options, `halfvec`, and the iterative index scans you'll need for 4.5.
- [Chroma: configure a collection](https://docs.trychroma.com/docs/collections/configure): the `space`, `ef_construction`, `ef_search` and `max_neighbors` settings used in Part C.

## Recall answers

1. The same prompt behaves differently on a different model, so prompt + model together is the release. Logging both lets you trace a bad answer to the exact pair and re-run the suite when either changes.
2. Access to private data, exposure to untrusted content, and a way to send data out (exfiltration channel). An agent with all three can leak data from one poisoned input; remove at least one leg.
3. Each model has its own vector space, so vectors from two models can't be compared. A query embedded with the new model would be scored against old vectors meaninglessly. Re-embed the whole corpus (keep the source text so you can).
