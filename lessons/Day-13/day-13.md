# Day 13 — Scope the search, pick the model, wire up RAG (2026-11-12)

Yesterday you learned how to score closeness and search millions of vectors fast. Today closes Module 4 and opens RAG. First, how to search only the vectors a user is allowed to see (4.5, metadata filtering and multi-tenancy). Second, how to choose an embedding model with evidence instead of a leaderboard screenshot (4.6). Third, how all the pieces join into a retrieval-augmented generation (RAG) system (5.1). The hands-on builds a two-tenant search, benchmarks two embedding models on your own labelled queries, then answers questions with citations. Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. Your vectors are normalised to length 1. Why do dot product, cosine and L2 give the same ranking?
2. In an IVF index, what does `nprobe` control, and what happens to recall and latency as you raise it?
3. In HNSW, which parameter do you raise *at query time* to get better recall, and what does it cost?

## 4.5 Metadata filtering and multi-tenancy

**The idea.** Vector search answers "what is most similar?". Real apps ask "what is most similar *among the things this user may see*?": this customer's documents, this department, contracts after 2024, files the user has permission to open. Metadata filtering adds those hard conditions to a similarity search. Multi-tenancy is the special case where the condition is "belongs to this customer". Get it wrong and you either return nothing useful or, worse, show one customer another customer's data.

**How it works.** Every vector is stored with a metadata payload (Day 12: `id`, `vector`, `text`, `metadata`). There are three ways to combine the filter with the search:

```
POST-FILTER     ANN top-k over ALL vectors  ──▶ drop rows that fail the filter ──▶ maybe < k left (or 0!)
PRE-FILTER      rows that pass the filter   ──▶ exact search over just those    ──▶ always k, slow if the subset is big
IN-INDEX        walk the HNSW graph but only count/visit nodes that pass the filter ──▶ k results, fast
(filterable ANN)   (Qdrant payload indexes, Weaviate, pgvector 0.8+ iterative scans, Pinecone metadata filters)
```

The trap is **post-filtering with a selective filter**. Say an HNSW search collects the 40 nearest candidates and your tenant owns 1% of the data. On average 0.4 of those 40 pass the filter, so the user gets an empty answer even though matching documents exist. Stores deal with this in different ways:

- **pgvector** applies a `WHERE` clause *after* the index scan, so filtered queries could return too few rows. Since version 0.8.0 you can turn on iterative index scans (`SET hnsw.iterative_scan = relaxed_order;` or `strict_order`). The scan then keeps going until it has enough matching rows or reaches `hnsw.max_scan_tuples` (default 20,000). IVFFlat has `ivfflat.iterative_scan` and `ivfflat.max_probes`. A B-tree index on the filter column (or a partial index per tenant) also helps the planner.
- **Qdrant** builds a *payload index* on the filter field and adds extra graph links so filtered walks stay connected. Its planner switches to a plain payload-index lookup when the filter is very selective.

**Multi-tenancy designs.** Each one trades isolation against cost:

| Design | Isolation | Cost / ops | Good for |
| --- | --- | --- | --- |
| Index (collection) per tenant | Strong; easy delete | Heavy when you have thousands of tenants (memory, connections) | A few large or regulated tenants |
| Namespace / partition per tenant (Pinecone namespaces, Postgres partitions) | Strong at query time | Moderate | Many tenants of mid size |
| Shared index + `tenant_id` metadata filter | Logical only; relies on your code | Cheapest | Many small, similar tenants |
| Tiered: shared by default, promote big tenants to their own shard | Mixed | Most moving parts | SaaS with a long tail of tenants plus a few whales |

Qdrant's guide recommends the shared-collection design for most cases. You create a keyword payload index on the tenant field with `is_tenant: true` (v1.11+) so each tenant's data is stored together. If queries are *always* tenant-scoped, you can set HNSW `m: 0` and `payload_m: 16`, which builds a graph per tenant instead of one global graph. Its tiered mode uses custom sharding with a `default` fallback shard and promotes large tenants to dedicated shards.

**The security rule.** The tenant filter is an *authorisation* control, not a search preference. Take `tenant_id` from the authenticated session on the server. Never take it from the user's message, a request body field the client controls, or an LLM tool argument. Wrap search in a function that *requires* the tenant, so nobody can forget it. Document permissions (ACLs) go into metadata at ingestion and must be re-synced when they change.

**Gotchas.**

- **Unindexed filter fields.** Filtering on a field with no payload or B-tree index often falls back to a full scan. Index every field you filter on.
- **Silent empty results.** Post-filtered queries that return 0–2 hits look like "the model doesn't know". Log the hit count per query.
- **Leaks outside the vector DB.** A shared semantic cache (Module 10.5), a shared reranker log or a prompt cache keyed only on the question can leak across tenants. Put the tenant in every cache key.
- **Deletes.** Right-to-erasure requests mean "delete every chunk where `tenant_id = X` or `source_doc = Y`". Design the metadata so that is one call.

**UiPath bridge.** It's Orchestrator folders and tenants for vectors. Context Grounding indexes live in a folder, and who can query them follows folder permissions, the same way assets and queues do.

**Interview angle.**

- *"You're building document Q&A for 2,000 B2B customers on one vector DB. How do you isolate them?"* Use a shared collection with an indexed `tenant_id` (marked as the tenant key where the store supports it). Inject the tenant on the server from the auth token inside a mandatory search wrapper. Use a filter-aware ANN path so selective filters don't return empty results. Promote the few huge or regulated tenants to their own shard or index. Add tests that try cross-tenant queries, and put `tenant_id` in every cache key.
- *"Filtered searches sometimes return nothing even though the documents exist. Why?"* It's post-filtering: the ANN step returned top candidates that the filter then removed. Fix it with in-index filtering, pgvector iterative scans, a larger candidate pool, or pre-filtering with exact search when the filtered subset is small.

## 4.6 Choosing and benchmarking embedding models

**The idea.** Retrieval quality is capped by the embedding model. If it doesn't put "reset my login" near "self-service password portal", no index, reranker or prompt can rescue the answer. There are dozens of models, and the public leaderboard changes monthly. You need a repeatable way to pick one for *your* data and limits, because switching later means re-embedding everything (Day 11).

**How it works: the public signal.** MTEB (Massive Text Embedding Benchmark) scores models on task families: retrieval, classification, clustering, reranking, semantic similarity and others. In 2025 it grew into MMTEB, covering 500+ tasks in 250+ languages, and the Hugging Face leaderboard now lets you filter by language, domain and task type. Read it carefully:

- Look at the **Retrieval** column, not the overall average. An overall average mixes in clustering and classification you may not care about.
- Many scores are self-submitted, and some models were trained on data close to the test sets. Treat small gaps (one or two points) as ties.
- Check the **max input tokens**, **dimensions**, **parameter count** and **license** columns before you get excited.

**How it works: the criteria.**

| Criterion | Why it matters | Example question |
| --- | --- | --- |
| Quality on *your* queries | Benchmarks are general; your domain isn't | Does it match "PO" to "purchase order"? |
| Languages | Many models are English-first | Do you need Hindi and German queries over English docs? |
| Max input length | Longer chunks get truncated silently | Is it 512 tokens or 8k+? |
| Dimensions and storage | Index RAM and cost grow linearly with dimensions | 1M chunks × 3,072 dims × 4 bytes ≈ 12.3 GB; at 768 dims ≈ 3.1 GB |
| Matryoshka support | Lets you truncate vectors (e.g. 3,072 → 768) with a small quality loss | Several current API models (OpenAI `text-embedding-3`, Google `gemini-embedding-001`, Voyage) support reduced dimensions |
| Latency and price | Query embedding sits on the hot path | API per-token price vs your own GPU or CPU |
| Hosting and data residency | Some data can't leave your cloud or region | Is an open-weights model (BGE-M3, Qwen3-Embedding, all-MiniLM) needed? |
| Query/document asymmetry | Some models expect a prefix or task type for queries | E5 uses `query:` / `passage:`; BGE v1.5 has an optional query instruction |

**Benchmarking your own.** It is a small eval (Module 8 will go deeper):

```
1. Collect 50–200 real queries (from logs, tickets, SMEs).
2. Label the relevant chunk id(s) for each one = your golden set.
3. For each candidate model: embed corpus + queries → search top-k.
4. Score:
   recall@k  = share of queries whose relevant chunk is in the top k
   MRR       = mean of 1 / (rank of first relevant hit)   (rank 1 → 1.0, rank 3 → 0.33)
   nDCG@10   = rewards putting the most relevant items highest (graded labels)
5. Also record: ms per query, vector size, $ per 1M tokens.
6. Pick the cheapest model within noise of the best on recall@k for the k you will actually pass on.
```

**Gotchas.**

- **Testing on the docs you wrote the queries from.** Paraphrase-heavy, real user queries are what break models. Include typos, acronyms and other languages.
- **Forgetting the query prefix** or the right pooling (mean vs CLS) for a model. That can cost several points and looks like "model B is worse".
- **Comparing at different k.** If your RAG passes 5 chunks to the LLM, recall@5 is the number that matters, not recall@1.
- **Ignoring re-embedding cost.** Changing models means re-embedding the whole corpus and rebuilding the index. Keep source text and a `embed_model` field in metadata.

**UiPath bridge.** It's the same discipline as checking a Document Understanding extractor on a sample of *your* invoices before trusting the vendor's accuracy figure.

**Interview angle.**

- *"How would you choose an embedding model for a new RAG system?"* Shortlist 3–4 models from the MTEB retrieval view, filtered by language, context length, license and hosting needs. Build a golden set of 100-ish real queries with labelled relevant chunks. Measure recall@k and MRR at the k you'll use, plus latency, storage and price. Choose the cheapest model within noise of the best, and record the model id with every vector so a later migration is planned, not accidental.
- *"Why not just take the #1 model on MTEB?"* The average mixes non-retrieval tasks, scores may be contaminated or self-reported, and it ignores your domain, languages, dimensions, cost, latency and data-residency limits.

## 5.1 RAG architecture end to end

**The idea.** An LLM only knows what was in its training data up to a cutoff. It has never seen your policies, contracts or tickets, and it makes things up when asked about them (Day 4, hallucinations). Retrieval-augmented generation fixes this at query time. It finds the few passages most relevant to the question and puts them in the prompt, so the model answers from evidence it can cite. Lewis et al. coined the term in 2020. Today "RAG" means any system that retrieves first and generates second. It's cheaper and fresher than fine-tuning for knowledge (you update the index, not the model), and it gives you citations and access control.

**How it works.** It's two pipelines that share one store:

```
OFFLINE: INGESTION (batch / on change)
  sources ─▶ load & parse ─▶ clean ─▶ chunk ─▶ embed ─▶ upsert (vector + text + metadata)
  (SharePoint,  (5.2: PDF,     (5.3)    (4.1, 4.6)   (4.3–4.5: tenant_id, ACL,
   S3, Confluence) tables, OCR)                       source, page, updated_at, embed_model)

ONLINE: QUERY (per request, latency budget ~1–3 s)
  user question
     │ (5.6 rewrite / multi-query)
     ▼
  embed query ─▶ retrieve top-k  ─▶ (5.5 rerank to top-n) ─▶ assemble prompt ─▶ LLM ─▶ answer + citations
                 (vector, 5.4 hybrid;                       system rules +        (5.8 grounding)
                  tenant filter 4.5)                         <source id=…> blocks
                                                             + question
  every stage logged ─▶ evals (5.9): retrieval metrics + answer faithfulness
```

The prompt at the end is just Module 3 work: the system prompt says to answer *only* from the sources and to cite their ids, and to say "I don't know" if they don't contain the answer. The sources go in XML-tagged blocks (3.4), with the question last.

A useful mental split: **retrieval quality** (did the right chunk reach the prompt?) and **generation quality** (did the model use it faithfully?). Most bad RAG answers are retrieval failures. Debug them separately: log the retrieved chunk ids for every request.

**Gotchas.**

- **RAG reduces hallucination; it doesn't remove it.** Models still blend sources, over-generalise or ignore the context. Grounding instructions, citations and faithfulness evals (5.8, 5.9) are part of the design, not extras.
- **"Naive RAG" is the baseline, not the product.** Fixed-size chunks + top-5 vector search + one prompt works for demos. Production systems add hybrid search, reranking and evals because users ask messy questions.
- **Stale or orphaned chunks.** If a policy is updated and the old chunks aren't deleted, retrieval serves both versions. Use stable document ids and replace all chunks of a document on re-ingest.
- **Too much context.** Stuffing 50 chunks costs tokens (Day 4: long context costs more) and can bury the right passage. Retrieve wide, rerank, pass a few.

**UiPath bridge.** Context Grounding *is* managed RAG. You point an index at a storage bucket or connector, it handles ingestion, and Agent Builder or Autopilot retrieves from it at run time. Today you see the parts it hides.

**Interview angle.**

- *"Walk me through a RAG system you'd build for internal policy Q&A."* Describe ingestion (connectors → parse → chunk with metadata → embed → vector store with tenant/ACL fields). Then the query path (auth → optional rewrite → hybrid retrieve with permission filter → rerank → grounded prompt with ids → answer with citations → "don't know" fallback). Close with operations: logging retrieved ids, a golden-set eval for retrieval and faithfulness, incremental re-indexing, cost and latency budget.
- *"Users say answers are wrong. How do you debug?"* Take the failing queries and check whether the right chunk was retrieved. If not, it's a retrieval problem: parsing, chunking, embedding model, filters, add hybrid search or reranking. If it was, it's a generation problem: prompt, context order, too much noise, model choice. Add the cases to the eval set so the fix stays fixed.

## Hands-on (30–45 min): tenant-safe search, a model bake-off, and a cited RAG answer

**Goal.** One script, three parts. Part A shows a cross-tenant leak and the post-filter "empty result" bug, then fixes both with a mandatory tenant filter (4.5). Part B benchmarks two local embedding models on a small golden set with recall@k, MRR and latency (4.6). Part C runs the full online RAG path for one tenant and produces an answer with citations, or "I don't know" (5.1).

**Setup.**

- Parts A and B use only what your `requirements.txt` already has (`numpy`, `torch`, `transformers`). Part B downloads `BAAI/bge-small-en-v1.5` (about 130 MB) the first time. No API cost.
- Part C uses your existing `.env` (`GEMINI_API_KEY`, `LLM_MODEL`) through the OpenAI-compatible client, as on Day 11. It makes 3 short chat calls. Run `python day13_rag.py c --dry` to print the prompts without calling the API.
- Save as `day13_rag.py` in your Day-13 exercise folder.

```python
"""Day 13: tenant-safe filtering, an embedding-model bake-off, and an end-to-end RAG answer.
Run:  python day13_rag.py a|b|c|all   [--dry]  (--dry = print the Part C prompt, no API call)"""
import os
import sys
import time

import numpy as np

# ---------- corpus: two tenants in ONE shared store ----------
DOCS = [  # (id, tenant, dept, text)
    ("a1", "acme", "finance", "Acme pays supplier invoices within 45 days of receipt, by bank transfer only."),
    ("a2", "acme", "finance", "At Acme, any change to a supplier's bank details must be confirmed by phone with a known contact."),
    ("a3", "acme", "it", "Acme staff reset passwords through the self-service portal; IT never asks for a password by email."),
    ("a4", "acme", "it", "Acme laptops must run the VPN client whenever they connect to public Wi-Fi."),
    ("a5", "acme", "hr", "Acme employees get 24 days of annual leave, approved by their line manager."),
    ("a6", "acme", "hr", "New Acme joiners complete safety training during their first week."),
    ("a7", "acme", "finance", "Acme expense claims above 500 euros need a director's sign-off."),
    ("a8", "acme", "it", "Acme keeps backups of the finance database for 90 days."),
    ("g1", "globex", "finance", "Globex pays supplier invoices within 30 days, and early payment earns a 2% discount."),
    ("g2", "globex", "finance", "Globex accepts payment by card or bank transfer for customer orders."),
    ("g3", "globex", "it", "Globex passwords expire every 60 days and must be at least 14 characters long."),
    ("g4", "globex", "it", "Globex uses hardware security keys instead of SMS codes for sign-in."),
    ("g5", "globex", "hr", "Globex staff receive 20 days of annual leave plus public holidays."),
    ("g6", "globex", "hr", "Globex offers four weeks of paid parental leave."),
    ("g7", "globex", "finance", "Globex reimburses expense claims in the next monthly payroll."),
    ("g8", "globex", "it", "Globex deletes chat logs after one year."),
]
# golden set: (tenant, query, relevant doc id). Add your own hard ones in step 3.
GOLDEN = [
    ("acme", "How long until a supplier gets paid?", "a1"),
    ("acme", "Someone emailed asking to switch the vendor's bank account. What do I do?", "a2"),
    ("acme", "I forgot my login, how do I get a new one?", "a3"),
    ("acme", "Do I need the VPN at the airport?", "a4"),
    ("acme", "How many holiday days do I get?", "a5"),
    ("acme", "Who must approve a large expense?", "a7"),
    ("globex", "Is there a discount for paying suppliers early?", "g1"),
    ("globex", "How often do I need to change my password?", "g3"),
    ("globex", "Can I use a text message code to log in?", "g4"),
    ("globex", "How much vacation do staff get?", "g5"),
    ("globex", "When do I get my expenses back?", "g7"),
    ("globex", "How long are chat messages kept?", "g8"),
]

# ---------- embedding models (local, CPU) ----------
MODELS = {
    # name: (hf id, pooling, query prefix)
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "cls", "Represent this sentence for searching relevant passages: "),
}
_cache = {}

def embed(texts, model="minilm", is_query=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    hf_id, pooling, prefix = MODELS[model]
    if model not in _cache:
        _cache[model] = (AutoTokenizer.from_pretrained(hf_id), AutoModel.from_pretrained(hf_id).eval())
    tok, net = _cache[model]
    if is_query:
        texts = [prefix + t for t in texts]
    batch = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        out = net(**batch).last_hidden_state
    if pooling == "cls":
        vecs = out[:, 0]
    else:
        mask = batch["attention_mask"].unsqueeze(-1).float()
        vecs = (out * mask).sum(1) / mask.sum(1)
    vecs = vecs.numpy()
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class Store:
    """A toy shared vector store. In search_postfilter, pretend the first step is an HNSW index that returns
    only its `ef` nearest candidates over the WHOLE store (like a real ANN index)."""
    def __init__(self, model="minilm"):
        self.model = model
        self.ids = [d[0] for d in DOCS]
        self.meta = [{"tenant": d[1], "dept": d[2]} for d in DOCS]
        self.text = [d[3] for d in DOCS]
        self.M = embed(self.text, model)

    def _scores(self, query):
        return self.M @ embed([query], self.model, is_query=True)[0]

    def search_unfiltered(self, query, k=3):                       # BUG 1: no tenant at all
        s = self._scores(query)
        return [(self.ids[i], float(s[i])) for i in np.argsort(-s)[:k]]

    def search_postfilter(self, query, tenant, k=3, ef=4):        # BUG 2: filter after ANN
        s = self._scores(query)
        candidates = np.argsort(-s)[:ef]                           # ANN returns ef candidates
        kept = [i for i in candidates if self.meta[i]["tenant"] == tenant]
        return [(self.ids[i], float(s[i])) for i in kept[:k]]

    def search(self, query, *, tenant, k=3, dept=None):           # FIX: tenant is mandatory
        if not tenant:
            raise ValueError("tenant is required")
        s = self._scores(query)
        allowed = [i for i, m in enumerate(self.meta)
                   if m["tenant"] == tenant and (dept is None or m["dept"] == dept)]
        allowed.sort(key=lambda i: -s[i])                          # pre-filter + exact search
        return [(self.ids[i], float(s[i])) for i in allowed[:k]]


# ---------- Part A: leaks and empty results ----------
def part_a():
    st = Store()
    q, tenant = "How many days of annual leave do I get?", "acme"
    print(f"Query from an {tenant} user: {q!r}")
    print("  unfiltered :", st.search_unfiltered(q))
    for ef in (2, 4, 8):
        print(f"  post-filter (ef={ef}) :", st.search_postfilter(q, tenant, ef=ef))
    print("  mandatory pre-filter :", st.search(q, tenant=tenant))
    print("  + dept='hr'          :", st.search(q, tenant=tenant, dept="hr"))
    try:
        st.search(q, tenant="")
    except ValueError as e:
        print("  empty tenant ->", e)


# ---------- Part B: embedding-model bake-off ----------
def evaluate(model, k_values=(1, 3)):
    t0 = time.perf_counter(); st = Store(model); build_s = time.perf_counter() - t0
    ranks, t_query = [], 0.0
    for tenant, q, gold in GOLDEN:
        t0 = time.perf_counter()
        hits = [h for h, _ in st.search(q, tenant=tenant, k=len(DOCS))]
        t_query += time.perf_counter() - t0
        ranks.append(hits.index(gold) + 1)
    ranks = np.array(ranks)
    row = {f"recall@{k}": float(np.mean(ranks <= k)) for k in k_values}
    row["MRR"] = float(np.mean(1.0 / ranks))
    row["ms/query"] = 1000 * t_query / len(GOLDEN)
    row["dims"] = st.M.shape[1]
    row["misses@1"] = [GOLDEN[i][2] for i in np.where(ranks > 1)[0]]
    return row

def part_b():
    for name in MODELS:
        r = evaluate(name)
        print(f"{name:10s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                         for k, v in r.items()))


# ---------- Part C: the online RAG path, with citations ----------
SYSTEM = ("You answer employee questions using ONLY the sources provided. "
          "Cite the source id in square brackets after each claim, e.g. [a1]. "
          "If the sources do not contain the answer, reply exactly: I don't know based on the available documents.")

def build_prompt(question, hits, st):
    blocks = "\n".join(f'<source id="{i}">{st.text[st.ids.index(i)]}</source>' for i, _ in hits)
    return f"<sources>\n{blocks}\n</sources>\n\n<question>{question}</question>"

def llm(system, user):
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
                    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
                    max_retries=3, timeout=90)
    r = client.chat.completions.create(model=os.getenv("LLM_MODEL", ""),
                                       messages=[{"role": "system", "content": system},
                                                 {"role": "user", "content": user}])
    time.sleep(4)
    return r.choices[0].message.content or ""

def answer(st, question, tenant, k=3, dry=False):
    hits = st.search(question, tenant=tenant, k=k)                 # tenant comes from "auth", not the user
    print(f"\n[{tenant}] Q: {question}\n  retrieved: {[(i, round(s, 3)) for i, s in hits]}")
    prompt = build_prompt(question, hits, st)
    print("  answer:", prompt if dry else llm(SYSTEM, prompt))

def part_c(dry=False):
    st = Store()
    answer(st, "How long do suppliers wait for payment, and how is it paid?", "acme", dry=dry)
    answer(st, "How long do suppliers wait for payment, and how is it paid?", "globex", dry=dry)
    answer(st, "What is the office Wi-Fi password?", "acme", dry=dry)


if __name__ == "__main__":
    args = sys.argv[1:]
    part = next((a for a in args if not a.startswith("--")), "all")
    dry = "--dry" in args
    for name, fn in [("a", part_a), ("b", part_b), ("c", lambda: part_c(dry))]:
        if part in (name, "all"):
            print(f"\n===== Part {name.upper()} ====="); fn()
```

**Steps.**

1. Part A (10 min). Run `python day13_rag.py a`. Check: (a) the unfiltered search puts a Globex document in an Acme user's top 3, which is a leak; (b) with post-filtering at small `ef` the Acme user gets fewer than 3 results, possibly zero; (c) the mandatory pre-filter always returns 3 Acme documents. Write one sentence on where `tenant` should come from in a real API, and why not the request body.
2. Part B (15 min). Run `python day13_rag.py b`. Record recall@1, recall@3, MRR, ms/query and dims for both models. Then delete the bge query prefix (set it to `""`) and rerun. Did the score move? Note which golden ids each model misses at rank 1 (`misses@1`).
3. Make the benchmark harder (10 min). Add 6 golden queries of your own that use *different words* from the documents. Examples: "pay-out timeline for vendors", "MFA via text", "PTO allowance", plus a typo or two. Rerun Part B. Which model holds up better, and is the gap big enough to matter at recall@3?
4. Part C (5–10 min). Run `python day13_rag.py c --dry` first and read the assembled prompts. Then run it for real. Check that the same question gives different, correctly cited answers for Acme ([a1]) and Globex ([g1]). The Wi-Fi question should get the exact "I don't know" line. If it doesn't, tighten `SYSTEM` and note what changed.
5. Wrap-up. In `notes.md` in the exercise folder, label each line of `answer()` with its box in the 5.1 diagram, and list the two boxes this script skips (hint: ingestion-side chunking, and reranking).

**What "done" looks like.** Part A shows a Globex id in the unfiltered list, a short or empty post-filter list at `ef=2`, and three Acme ids from `search()`. Part B prints one metrics row per model for the original set and again with your harder queries, and you can say whether the gap between models grew. Part C prints three retrieval lists, two answers with the right citation and one "I don't know". Your `notes.md` has the benchmark table and the diagram mapping.

## Go deeper (optional)

- [Qdrant: Configure Multitenancy](https://qdrant.tech/documentation/manage-data/multitenancy/): the `is_tenant` payload index, the `m: 0` / `payload_m` per-tenant graphs, and tiered multitenancy with shard keys.
- [pgvector: iterative index scans](https://github.com/pgvector/pgvector#iterative-index-scans): why filtered ANN queries return too few rows and the `hnsw.iterative_scan` settings that fix it.
- Paper: [MMTEB: Massive Multilingual Text Embedding Benchmark](https://arxiv.org/abs/2502.13595) and the live [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). Read the paper's section on task selection before trusting an average.
- Paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401): the paper that named RAG. Sections 1–2 cover the idea of combining a retriever with a generator.

## Recall answers

1. For unit vectors, the dot product equals the cosine, and squared L2 = 2 − 2 × cosine. All three are monotonic functions of the same number, so they sort documents in the same order.
2. `nprobe` is how many of the nearest clusters (out of `nlist`) are scanned per query. Raising it improves recall toward exact search, but latency grows roughly linearly because more vectors are scanned.
3. `ef_search` (the size of the candidate list during the graph walk). Higher values give better recall but cost more latency per query. `M` and `ef_construction` are build-time settings.
