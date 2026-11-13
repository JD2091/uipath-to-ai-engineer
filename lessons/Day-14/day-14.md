# Day 14 — From messy files to findable chunks: parse, chunk, hybrid search (2026-11-13)

Yesterday you drew the full RAG pipeline. Today you build its first three stages properly, because most bad RAG answers start here, not in the LLM. First, turning PDFs, tables and scans into clean text with structure (5.2). Second, cutting that text into chunks that make sense on their own (5.3). Third, searching those chunks with keywords and vectors together, so exact codes like "VPN-809" and paraphrases like "maternity leave" both work (5.4). The hands-on parses a PDF two ways, chunks a handbook three ways, then compares BM25, dense and hybrid retrieval on a golden set. Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. A tenant owns 0.5% of the vectors in a shared index. Why can post-filtering return zero results for them?
2. Why shouldn't you choose an embedding model from the MTEB leaderboard average alone?
3. Users report wrong RAG answers. What is the first check that tells you whether it's a retrieval or a generation problem?

## 5.2 Document loading and parsing (PDFs, tables, OCR)

**The idea.** Embeddings and LLMs work on text. Enterprise knowledge lives in PDFs, Word files, slides, spreadsheets, HTML pages and scanned images. Parsing turns those into clean text that keeps the structure a reader relies on: headings, reading order, lists and tables. If parsing scrambles a table or glues a page footer into every paragraph, no chunker, embedding model or prompt can fix it later. Garbage in, confident garbage out.

**How it works.** A PDF is not a document in the Word sense. It is a list of drawing instructions: "put these glyphs at x, y". There are no paragraphs, no table cells, sometimes not even spaces. Parsers rebuild structure from positions. Pick the tier by how hard your documents are:

```
TIER 1  text-layer extraction     pypdf, pdfminer          fast, free; reading order only, tables flattened
TIER 2  layout-aware parsing      PyMuPDF4LLM, Docling,    rebuilds headings, columns, tables -> Markdown/JSON
                                  Unstructured
TIER 3  OCR / vision              Tesseract, cloud OCR     for scans/photos: page image -> text (+ boxes)
                                  (Azure Document Intelligence, AWS Textract, Google Document AI),
                                  or a multimodal LLM reading the page image (Day 5, 1.13)

Pipeline:  file ─▶ detect type ─▶ has text layer? ──yes──▶ layout parse ─▶ Markdown + metadata
                                        │no (scan)                          (source, page, section)
                                        └──▶ OCR ─────────────────────────────▲
```

A table is the classic test. Plain extraction of a 3-column hotel-cap table gives one value per line (`Grade`, `Hotel cap per night`, `G1-G3`, `120 USD`, …). The row relationships are gone, so "hotel cap for G5" becomes a guess. A layout-aware parser emits `| G4-G6 | 180 USD | 55 USD |`, a row the LLM can read. Docling, an MIT-licensed toolkit from IBM Research, does this with a layout-analysis model plus a dedicated table-structure model, and exports Markdown or JSON. A cheap scan check is to look at characters per page: near zero means no text layer, so route to OCR.

Keep metadata from the start: source file, page number, section heading, last-modified date. Citations (5.8), permission filters (4.5) and re-indexing all depend on it.

**Gotchas.**

- **Headers, footers and page numbers.** "ACME CONFIDENTIAL – Page 3" ends up in every chunk and pollutes both keyword and vector search. Strip repeated lines.
- **Multi-column layouts and reading order.** Naive extraction can read across two columns line by line and interleave unrelated sentences.
- **OCR errors are silent.** "0" vs "O" or "1" vs "l" break exact matches on IDs and amounts. Spot-check OCR output on real samples, and keep the confidence scores.
- **Licences matter.** PyMuPDF (and so PyMuPDF4LLM) is AGPL or commercial. Check before you ship it inside a product.

**UiPath bridge.** This is the digitization step of Document Understanding: OCR engine choice, the Document Object Model, and why a bad scan sinks extraction. RAG needs the same step, but the output is Markdown for retrieval instead of fields for a form.

**Interview angle.**

- *"Your RAG system answers questions about financial reports badly, especially numbers. Where do you look first?"* Parsing. Open the parsed text for a failing page. Tables are usually flattened or split, so the numbers lose their row and column labels. Move to a layout-aware parser that emits Markdown tables or extract tables separately as structured records. Keep each table whole in one chunk with its caption and heading, and evaluate on table questions specifically.
- *"How would you handle a mix of digital PDFs and scanned faxes?"* Detect the text layer per page. Digital pages go through a layout parser; scanned pages go through OCR or a vision model. Normalise both to the same Markdown + metadata format, record which path each page took, and track OCR confidence so low-quality pages can be flagged or reviewed.

## 5.3 Chunking strategies

**The idea.** You can't embed a 200-page manual as one vector: the meaning gets averaged into mush, and you couldn't put the whole thing in a prompt cheaply anyway. So you split documents into chunks, embed each, and retrieve the few that matter. Chunk size is a trade-off. Too small and a chunk loses the context that makes it answerable ("It expires on 31 March" — what does?). Too big and one vector covers several topics, so similarity blurs, and you pay for irrelevant tokens in the prompt.

**How it works.** Four common strategies, from simplest to smartest:

| Strategy | How it splits | Good for | Weakness |
| --- | --- | --- | --- |
| Fixed-size (+ overlap) | Every N tokens, with 10–20% overlap | Uniform text, quick baseline | Cuts mid-sentence, mid-table, across sections |
| Recursive | Try paragraph breaks, then lines, then sentences, then words until under the limit (LangChain's RecursiveCharacterTextSplitter) | General prose | Still blind to headings |
| Structure-aware | Split on headings, keep tables and lists whole, prefix each chunk with its heading path | Manuals, policies, docs from Markdown/HTML | Needs good parsing (5.2) |
| Semantic | Embed sentences and cut where the topic shifts (similarity drop) | Long, unstructured transcripts | Costs embedding calls; gains often small |

A structure-aware chunk looks like this:

```
Acme Employee Handbook > 3. Travel reimbursement      <- breadcrumb: the chunk says what it's about
Submit claims on form FIN-114 within 30 days...
| Grade | Hotel cap per night | Daily meal allowance |  <- table kept whole
| G4-G6 | 180 USD             | 55 USD               |
```

Two refinements you'll meet: **parent-child (small-to-big)** indexes small chunks for precise matching but sends their larger parent section to the LLM. **Late chunking** embeds the whole document with a long-context model first, then pools token vectors per chunk so each chunk "remembers" its document. Contextual retrieval (5.7) attacks the same "lost context" problem from another angle.

Starting point for most text: 200–500 tokens, about 10–20% overlap for fixed-size, and structure-aware splitting whenever you have headings. Then tune with your golden set, not by intuition.

**Gotchas.**

- **No universal best size.** It depends on your documents, your questions and your embedding model's max input. Benchmark two or three settings.
- **Silent truncation.** Many embedding models cap input at 512 tokens. A 1,000-token chunk is cut, and the tail is never searchable.
- **Overlap isn't free.** It duplicates tokens in the index and can return near-identical chunks in the top-k. Deduplicate before the prompt.
- **Chunk ids must be stable.** Derive them from document id + position so re-ingestion replaces old chunks (Day 13's stale-chunk gotcha).

**UiPath bridge.** Context Grounding chunks your indexed files for you. These are the knobs it hides, and why the same bucket can answer well for policies and badly for spreadsheets.

**Interview angle.**

- *"How do you choose chunk size?"* Start from the document structure and the embedding model's input limit. Use structure-aware chunks of a few hundred tokens with heading breadcrumbs as a baseline. Then build a golden set of real questions and compare two or three chunking configurations on recall@k and answer quality. Pick the winner on data, and re-check when the document type changes.
- *"Retrieval finds the right section but the answer misses a key detail from the next paragraph. Fix?"* The chunk boundary cut the context. Options: structure-aware splitting so related paragraphs stay together, overlap, or parent-child retrieval that sends the whole section to the LLM once a small chunk matches.

## 5.4 Hybrid search (BM25 + vectors)

**The idea.** Dense vector search is great at meaning ("maternity leave" matches "birth parents receive 26 weeks"). It is weak on exact tokens it rarely saw in training: error codes, SKUs, form numbers, names, acronyms. Searching "VPN-809" by embedding may return any VPN paragraph. Keyword search (BM25) is the opposite: exact tokens are its strength, synonyms its blind spot. Hybrid search runs both and merges the results, so you get both strengths.

**How it works.** BM25 is the classic ranking function behind Lucene, Elasticsearch and OpenSearch. For each query term it adds up:

```
score(doc) = Σ over query terms  IDF(term) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × doclen / avg_doclen))

IDF  rare terms count more ("vpn-809" >> "the")
tf   more occurrences help, but saturate (k1 ≈ 1.2–2.0)
b    long documents are penalised for having more words (b ≈ 0.75)
```

The two scores live on different scales (BM25 is unbounded; cosine sits in −1..1), so don't just add them. The standard merge is **Reciprocal Rank Fusion (RRF)**, from Cormack, Clarke and Büttcher (SIGIR 2009). It uses only ranks:

```
RRF(doc) = Σ over result lists  1 / (k + rank_in_list)        k = 60 by convention

            BM25 rank   dense rank   RRF score
chunk A         1           4        1/61 + 1/64 = 0.0320   <- strong in both: wins
chunk B         2           -        1/62        = 0.0161
chunk C         -           1        1/61        = 0.0164
```

The alternative is a weighted sum of normalised scores (Weaviate exposes this as an `alpha` slider). It can beat RRF when tuned, but it needs tuning. Most vector databases now support hybrid natively: Qdrant's Query API (prefetch both, fuse with RRF), Weaviate, Elasticsearch/OpenSearch, and Postgres full-text search alongside pgvector. "Learned sparse" models such as SPLADE sit between the two: sparse vectors like BM25, but with learned term expansion.

**Gotchas.**

- **Tokenisation decides BM25's fate.** A tokenizer that splits "VPN-809" into "vpn" and "809" weakens exact matching. Check how your engine analyses codes and IDs.
- **Hybrid is not automatically better.** If one retriever is much weaker for your data, fusion can drag good results down. Measure all three on your golden set.
- **Fuse the candidates, not the whole corpus.** Take the top 20–50 from each retriever, fuse, then rerank (5.5) and send only the top few to the LLM.
- **BM25 statistics need the whole corpus.** IDF and average length change as you add documents. Engines handle this; a hand-rolled index must be rebuilt.

**UiPath bridge.** It's exact selectors vs fuzzy selectors. Exact matching is precise but brittle; fuzzy matching tolerates variation but can grab the wrong element. Good automations use both, and so does good retrieval.

**Interview angle.**

- *"Why use hybrid search instead of pure vector search?"* Embeddings capture meaning but handle rare exact tokens (IDs, codes, names, new jargon) poorly. BM25 does the reverse. Running both and fusing with RRF improves recall across both query types at low cost. I'd confirm the gain on a golden set split by query type.
- *"How do you combine BM25 and cosine scores?"* Not by adding raw scores, because the scales differ. Use Reciprocal Rank Fusion, which sums 1/(k + rank) with k ≈ 60 and needs no tuning. Or use a weighted sum of min-max-normalised scores with the weight tuned on evals. Then rerank the fused top candidates.

## Hands-on (30–45 min): parse a PDF two ways, chunk three ways, retrieve three ways

**Goal.** One script, three parts. Part A builds a two-page PDF with a header, footer and table, then extracts it with pypdf and with PyMuPDF4LLM, so you see the table survive or die (5.2). Part B chunks a small handbook with fixed-size, fixed-with-overlap and heading-aware strategies (5.3). Part C scores BM25, dense and hybrid (RRF) retrieval on a 10-query golden set, for each chunking (5.4).

**Setup.**

- New installs: `pip install pypdf pymupdf4llm reportlab`. Everything else is already in your `requirements.txt` (`numpy`, `torch`, `transformers`, `tiktoken`).
- Part C reuses `BAAI/bge-small-en-v1.5` from Day 13 (cached after the first download). BM25 is written by hand in about 15 lines so you see the formula working. No API calls, no cost.
- Save as `day14_ingest.py` in your Day-14 exercise folder. Run `python day14_ingest.py a`, then `b`, then `c`.

```python
"""Day 14: parse a PDF two ways, chunk a handbook three ways, then compare BM25 vs dense vs hybrid (RRF).
Run:  python day14_ingest.py a|b|c|all"""
import re
import sys
import time

import numpy as np

# ---------- shared corpus: a small IT/HR handbook in Markdown ----------
HANDBOOK = """# Acme Employee Handbook

## 1. Laptop and VPN
All staff use the GlobalProtect VPN. If the client shows error VPN-809, your certificate has expired: open the Self-Service portal and choose "Renew device certificate". Error VPN-812 means the gateway is overloaded; wait five minutes and reconnect. Never share your token with a colleague.

## 2. Password policy
Passwords must be at least 14 characters and are rotated every 180 days. Reuse of the last 10 passwords is blocked. Accounts lock after 5 failed attempts and unlock automatically after 30 minutes, or immediately through the service desk.

## 3. Travel reimbursement
Book travel through the corporate portal at least 14 days ahead. Submit claims on form FIN-114 within 30 days of returning. Receipts are required for any item above 25 USD.

| Grade | Hotel cap per night | Daily meal allowance |
|-------|---------------------|----------------------|
| G1-G3 | 120 USD | 40 USD |
| G4-G6 | 180 USD | 55 USD |
| G7+   | 250 USD | 75 USD |

## 4. Leave
Employees get 24 days of paid annual leave per calendar year. Up to 5 unused days carry over to the next year and expire on 31 March. Request leave in the HR portal using form HR-217 at least two weeks ahead for any block longer than three days.

## 5. Parental leave
Birth parents receive 26 weeks of fully paid leave; other parents receive 12 weeks. Leave can start up to 4 weeks before the expected date. Tell your manager and HR at least 8 weeks in advance.

## 6. Working from home
Staff may work remotely up to 3 days a week with manager approval. Home office equipment up to 300 USD is reimbursed once every three years through form FIN-120. Remote work from another country needs prior approval from Legal because of tax rules.

## 7. Expense card
Corporate cards have a monthly limit of 2,000 USD. A lost card must be reported within 24 hours by calling the bank and logging ticket type CARD-LOST in the service desk. Personal spending on the card is not allowed.

## 8. Software requests
Install software only from the Company Portal. Anything not listed needs a request with category SW-REQ and security review, which takes about 5 working days. Browser extensions count as software.

## 9. Incident reporting
Report a suspected phishing email with the "Report Phish" button. Report a lost or stolen laptop to the security team within one hour; the device is remotely wiped after confirmation. Security incidents get ticket priority P1.
"""

# golden set: (query, section number that answers it). Mix of exact identifiers and paraphrases.
GOLDEN = [
    ("what does VPN-809 mean", 1),
    ("FIN-114", 3),
    ("hotel limit for a G5 employee", 3),
    ("how do I apply for annual leave, which form", 4),
    ("how long is maternity leave", 5),
    ("can I claim money for a desk and chair at home", 6),
    ("I lost my company credit card", 7),
    ("someone sent me a suspicious email trying to steal my login", 9),
    ("my account got locked after wrong password attempts", 2),
    ("SW-REQ turnaround time", 8),
]


def count_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except ImportError:  # rough fallback: ~1.3 tokens per word
        return int(len(text.split()) * 1.3)


# ---------- Part A (5.2): parse a PDF two ways ----------
def make_pdf(path="policy.pdf"):
    """Build a 2-page PDF with a header/footer on every page and a real table."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

    styles = getSampleStyleSheet()

    def header_footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 815, "ACME CONFIDENTIAL - Internal use only")
        canvas.drawString(40, 20, f"Page {doc.page}")

    story = [
        Paragraph("3. Travel reimbursement", styles["Heading2"]),
        Paragraph("Submit claims on form FIN-114 within 30 days. Hotel caps by grade:", styles["BodyText"]),
        Table(
            [["Grade", "Hotel cap per night", "Daily meal allowance"],
             ["G1-G3", "120 USD", "40 USD"],
             ["G4-G6", "180 USD", "55 USD"],
             ["G7+", "250 USD", "75 USD"]],
            style=TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]),
        ),
        PageBreak(),
        Paragraph("4. Leave", styles["Heading2"]),
        Paragraph("Employees get 24 days of paid annual leave. Use form HR-217.", styles["BodyText"]),
    ]
    SimpleDocTemplate(path, pagesize=A4).build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return path


def part_a():
    path = make_pdf()
    print(f"Built {path}\n")

    from pypdf import PdfReader
    reader = PdfReader(path)
    print("=== 1) pypdf: plain text, reading order only ===")
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        print(f"--- page {i} ({len(text)} chars) ---\n{text}\n")
        if len(text.strip()) < 20:
            print("    ^ almost no text layer: this page is probably scanned -> needs OCR\n")

    print("=== 2) pymupdf4llm: layout-aware Markdown ===")
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(path)
        print(md)
    except ImportError:
        print("pip install pymupdf4llm   (then re-run)")

    print("TODO: answer in notes.md -> Which output keeps the table as rows you could chunk?"
          " Where did the header/footer end up, and why would that pollute every chunk?")


# ---------- Part B (5.3): three chunkers ----------
def chunk_fixed(text, size=60, overlap=0):
    """Fixed-size word windows (a stand-in for token windows). Ignores structure."""
    words = text.split()
    step = max(1, size - overlap)
    return [{"text": " ".join(words[i:i + size]), "section": None} for i in range(0, len(words), step)]


def chunk_by_headings(text, max_tokens=180):
    """Structure-aware: split on '## ' headings, keep tables whole, prefix each chunk with its heading."""
    chunks = []
    doc_title = text.strip().splitlines()[0].lstrip("# ").strip()
    for block in re.split(r"\n(?=## )", text):
        if not block.startswith("## "):
            continue
        heading, body = block.split("\n", 1)
        heading = heading[3:].strip()
        section = int(heading.split(".")[0])
        # split body into paragraphs; a Markdown table is one paragraph, so it never gets cut in half
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        groups, buf = [], []
        for p in paras:
            if buf and count_tokens("\n\n".join(buf + [p])) > max_tokens:
                groups.append(buf)
                buf = []
            buf.append(p)
        if buf:
            groups.append(buf)
        for g in groups:  # breadcrumb prefix so the chunk still says what it is about
            chunks.append({"text": f"{doc_title} > {heading}\n" + "\n\n".join(g), "section": section})
    return chunks


def label_fixed_chunks(chunks, text):
    """Give fixed-size chunks a section label = the section where the chunk's first word sits."""
    starts = [(m.start(), int(m.group(1))) for m in re.finditer(r"## (\d+)\.", text)]
    for c in chunks:
        pos = text.find(" ".join(c["text"].split()[:6]))
        c["section"] = max([s for p, s in starts if p <= pos], default=1) if pos >= 0 else None
    return chunks


def part_b(show=True):
    fixed = label_fixed_chunks(chunk_fixed(HANDBOOK, 60, 0), HANDBOOK)
    overlap = label_fixed_chunks(chunk_fixed(HANDBOOK, 60, 15), HANDBOOK)
    heads = chunk_by_headings(HANDBOOK)
    if show:
        for name, cs in [("fixed-60", fixed), ("fixed-60/overlap-15", overlap), ("heading-aware", heads)]:
            toks = [count_tokens(c["text"]) for c in cs]
            print(f"{name:22s} chunks={len(cs):3d}  avg tokens={np.mean(toks):6.1f}  max={max(toks)}")
        print("\nWhere did the travel table land in fixed-60?")
        for c in fixed:
            if "G4-G6" in c["text"] or "Hotel cap" in c["text"]:
                print("  ...", c["text"][:160], "...")
        print("\nHeading-aware chunk for section 3:\n", [c for c in heads if c["section"] == 3][0]["text"])
    return {"fixed-60": fixed, "fixed-60/ov15": overlap, "heading-aware": heads}


# ---------- Part C (5.4): BM25 vs dense vs hybrid (RRF) ----------
def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())  # keeps "vpn-809", "fin-114" whole


class BM25:
    """Okapi BM25 in ~15 lines. k1 = term-frequency saturation, b = length normalisation."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.avgdl = np.mean([len(d) for d in self.docs])
        n = len(self.docs)
        vocab = {t for d in self.docs for t in d}
        df = {t: sum(t in d for d in self.docs) for t in vocab}
        self.idf = {t: np.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in vocab}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        for i, d in enumerate(self.docs):
            for t in tokenize(query):
                if t not in self.idf:
                    continue
                tf = d.count(t)
                out[i] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
        return out


_model = {}

def embed(texts, is_query=False):
    """bge-small with CLS pooling, same as Day 13. Downloads ~130 MB on first run."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    hf_id = "BAAI/bge-small-en-v1.5"
    if not _model:
        _model["tok"], _model["m"] = AutoTokenizer.from_pretrained(hf_id), AutoModel.from_pretrained(hf_id).eval()
    if is_query:
        texts = ["Represent this sentence for searching relevant passages: " + t for t in texts]
    batch = _model["tok"](texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        v = _model["m"](**batch).last_hidden_state[:, 0].numpy()
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion: score(doc) = sum over lists of 1 / (k + rank). Uses ranks only, never raw scores."""
    fused = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)


def evaluate(chunks, k=1, verbose=False):
    texts = [c["text"] for c in chunks]
    bm25 = BM25(texts)
    doc_vecs = embed(texts)
    hits = {"bm25": 0, "dense": 0, "hybrid": 0}
    for query, section in GOLDEN:
        lex = list(np.argsort(-bm25.scores(query)))
        den = list(np.argsort(-(doc_vecs @ embed([query], is_query=True)[0])))
        hyb = rrf([lex[:20], den[:20]])
        row = []
        for name, ranking in [("bm25", lex), ("dense", den), ("hybrid", hyb)]:
            ok = any(chunks[i]["section"] == section for i in ranking[:k])
            hits[name] += ok
            row.append("Y" if ok else ".")
        if verbose:
            print(f"  {'  '.join(row)}   {query}")
    return {m: h / len(GOLDEN) for m, h in hits.items()}


def part_c():
    chunkings = part_b(show=False)
    print("Per-query hit@1 on heading-aware chunks   (bm25 dense hybrid)")
    evaluate(chunkings["heading-aware"], verbose=True)
    print(f"\n{'chunking':16s} {'bm25':>6s} {'dense':>6s} {'hybrid':>7s}   (hit@1 over {len(GOLDEN)} queries)")
    for name, chunks in chunkings.items():
        t = time.perf_counter()
        r = evaluate(chunks)
        print(f"{name:16s} {r['bm25']:6.0%} {r['dense']:6.0%} {r['hybrid']:7.0%}   ({time.perf_counter() - t:.1f}s)")
    print("\nTODO: in notes.md, name one query where BM25 beat dense and one where dense beat BM25, and why.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    if arg in ("a", "all"):
        part_a()
    if arg in ("b", "all"):
        part_b()
    if arg in ("c", "all"):
        part_c()
```

**Steps.**

1. Run Part A. Compare the two outputs: does the table come out as rows or one value per line? Where did "ACME CONFIDENTIAL" and "Page 1" land? Note in `notes.md` how you would strip them.
2. Run Part B. Look at where the travel table landed in the fixed-60 chunks versus the heading-aware chunk for section 3.
3. Run Part C. Read the per-query table (Y = the top-1 chunk is from the right section). Find one query BM25 gets and dense misses, and one the other way round. Explain each in `notes.md`.
4. Add two queries of your own to `GOLDEN`: one with an exact code (e.g. "CARD-LOST") and one pure paraphrase with no shared words. Re-run Part C.
5. Stretch: in `chunk_by_headings`, set `max_tokens=40` so sections split into smaller chunks, and see how hit@1 changes. Then try `rrf(..., k=10)` versus `k=60`.

**What "done" looks like.** Part A shows the table as separate lines in pypdf and as a Markdown table in PyMuPDF4LLM. Part B prints chunk counts and token stats for all three chunkers. Part C prints a hit@1 table for 3 chunkings × 3 retrievers. Heading-aware chunks should clearly beat both fixed-size ones (a fixed chunk that straddles two sections only counts for the section it starts in, which is the point: it's a poor retrieval unit). `notes.md` has your two explained BM25-vs-dense cases and the results for your own queries.

## Go deeper (optional)

- Paper: [Docling Technical Report (IBM Research, 2024)](https://arxiv.org/abs/2408.09869). Shows how layout analysis and table-structure models rebuild a PDF's structure. The project lives at [github.com/docling-project](https://github.com/docling-project).
- Paper: [An Analysis of Fusion Functions for Hybrid Retrieval (Bruch et al.)](https://arxiv.org/abs/2210.11934). Compares RRF with weighted score fusion and explains when each wins. Read sections 1–4.

## Recall answers

1. Post-filtering runs the ANN search over all vectors first, takes the top-k, then drops rows that fail the tenant filter. If the tenant owns 0.5% of the data, the global top-k will often contain none of their vectors, so nothing survives the filter. Use in-index (filterable) search or a mandatory pre-filter.
2. The average mixes many tasks, domains and languages that may not match yours. A model can rank high overall and still lose on your data. Build a small golden set from your own documents and queries, and compare candidates on recall@k/MRR, latency, dimensions (storage cost), max input length and licence.
3. Check whether the chunk that contains the correct answer was retrieved for the failing query. If it wasn't, it's a retrieval problem (parsing, chunking, embeddings, filters). If it was retrieved but the answer is still wrong, it's a generation problem (prompt, context order, noise, model).
