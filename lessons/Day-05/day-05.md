# Day 5 — Choosing and checking a model: what it can see, which one to pick, and how to score it (2026-11-03)

Today's thread: modern models take more than text. Images, PDFs and audio become tokens too, with their own costs and failure modes (1.13). Which model you send them to is a trade-off between quality, cost, latency, deployment and licence (1.14). And the only honest way to make that choice is to measure, which needs the classical ML vocabulary of train/test splits, overfitting, precision, recall and F1 (1.15). This closes Module 1. Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. A request has a 2,000-token prompt and a 1,000-token answer. Which phase dominates total latency, and why?
2. Name two hallucination mitigations that work better than lowering temperature.
3. Why is a reasoning model's visible chain of thought not a reliable audit trail?

## 1.13 Multimodality: vision, documents, audio

**The idea.** A multimodal model accepts inputs other than text (images, PDF pages, audio, sometimes video) and reasons over them in the same context as your text prompt. Some models also *output* images or speech. It exists because most enterprise information is not clean text: scanned invoices, screenshots, charts in a PDF, call recordings. The old approach chained OCR → text → LLM, which threw away layout, tables, stamps, handwriting and charts. A multimodal model reads the page roughly the way a person does.

**How it works.** Everything is turned into vectors in the same embedding space as text tokens, and the transformer attends over all of them together:

```
image ──► cut into patches (e.g. 28×28 px) ──► vision encoder ──► projection ──┐
                                                                               ├──► one sequence ──► transformer ──► text out
text  ──► BPE tokens ──► token embeddings ─────────────────────────────────────┘
audio ──► short frames ──► audio encoder ──► projection ───────────────────────┘
```

So images and audio are billed and limited as tokens, like text. Concrete numbers (checked 2026-11-03):

```
Claude images: tokens ≈ ceil(width/28) × ceil(height/28)
  1000×1000 px            -> 36 × 36 = 1,296 tokens
  large images are downscaled first: standard models cap at a 1,568 px long edge / 1,568 tokens;
  Claude 4.7 and later allow up to 2,576 px / 4,784 tokens ("high resolution", up to ~3× the tokens)
Claude PDFs: each page = extracted text (typically 1,500–3,000 tokens) + the page as an image
Gemini images: ≤384 px on both sides = 258 tokens; larger images are tiled at 768×768, 258 tokens per tile
Gemini audio: 32 tokens per second  -> a 10-minute call ≈ 19,200 tokens
```

How you send it differs by API. Claude's Messages API takes `image` and `document` content blocks (base64, URL or a Files API id). OpenAI-style chat APIs, including Gemini's OpenAI-compatible endpoint that your repo uses, take an `image_url` part (a `data:image/png;base64,...` URL works) and an `input_audio` part with base64 `data` and a `format` such as `wav`.

For voice there are two designs. A **pipeline** runs speech-to-text → LLM → text-to-speech: easy to log, test and swap parts, but adds latency and loses tone. A **native speech-to-speech** (realtime) model hears and speaks directly: lower latency and keeps emotion and interruptions, but is harder to audit and usually costs more per minute.

**Gotchas.**

- Downscaling happens silently. A full A4 scan shrunk to about 1,568 px can make small print hard to read, and the model will still answer confidently. Crop to the region you need instead of sending the whole page at higher resolution.
- Vision is weak at exact counting, precise coordinates, rotated or tiny (under ~200 px) images and dense tables. Validate extracted numbers with code (do line items sum to the total?).
- Images are expensive at volume. 50,000 pages a month at ~1,500 image tokens plus ~2,000 text tokens each is about 175M input tokens. Compare that against classic OCR plus a text-only call.
- Text inside an image is still text the model reads. A screenshot or PDF can carry instructions, which makes it a channel for indirect prompt injection (Module 11).
- Place images or documents before the question in the message; Anthropic's docs recommend this order.

**UiPath bridge.** Document Understanding's flow is digitize → classify → extract → validate. A vision LLM collapses digitize, classify and extract into one call (that is what DU's generative extraction does), but the validation step in Action Center is still needed.

**Interview angle.**

- *How does an LLM "see" an image?* The image is split into patches, a vision encoder turns each patch into a vector, and a projection maps those into the model's token-embedding space. The transformer then attends over image and text tokens together. That is why images cost tokens and are downscaled to fit a budget.
- *Design an LLM pipeline for 50k scanned invoices a month.* Cheap classification first; crop or downscale to the regions you need; extract to a strict JSON schema; check totals and vendor IDs against the ERP in code; route low-confidence or failed checks to human review; use the batch API for cost; and track field-level precision and recall on a labelled sample each release.

## 1.14 Model landscape: closed vs open weights, sizes, choosing a model

**The idea.** There is no single best model. There are dozens of strong ones that differ in quality, price, speed, context length, modalities, where they can run, and licence. Choosing a model is an engineering decision with constraints and measurements, not a leaderboard lookup.

**How it works.** The main axes:

```
Axis            Options and what they mean
--------------  ---------------------------------------------------------------------------
Access          Closed (API only: Anthropic Claude, OpenAI GPT, Google Gemini)
                Open weights (download and run: DeepSeek, Qwen, Mistral, GLM, Kimi, Llama, ...)
Licence         Apache 2.0 / MIT (very permissive) vs custom licences with use limits.
                "Open weights" is not "open source": training data and code usually stay private.
Size tier       Big / medium / small in each family (Opus/Sonnet/Haiku, Pro/Flash/Flash-Lite).
                Small: cheaper and faster, good for classify/extract/route.
Architecture    Dense: every parameter is used for every token.
                Mixture of Experts (MoE): many parameters in total, only some "active" per token,
                so you get the knowledge of a big model at the compute cost of a smaller one.
Hosting         Vendor API · the same model on a cloud (Bedrock / Vertex / Azure) · your own GPUs
```

Rule of thumb for running open weights: memory for weights ≈ parameters × bytes per parameter. An 8B model needs ~16 GB in fp16 and ~4–5 GB at 4-bit. A 70B model needs ~140 GB in fp16. Add the KV cache from Day 4 on top. For MoE models, *all* experts must fit in memory even though only some are active.

Snapshot as of early November 2026 (from public trackers; verify before you rely on it, since this changes monthly):

- **Closed frontier:** Anthropic's Claude 5.5 family (Opus, Sonnet, Haiku), OpenAI's GPT-6 / 6.1 family, Google's Gemini 4 and 3.x Flash models. All offer roughly 1M-token context windows and image input.
- **Open weights:** DeepSeek V4, Alibaba's Qwen 3.8 Flash, Z.ai's GLM-5.3 (MIT licence), Moonshot's Kimi K3, Mistral Large 4 and Mistral Small 4 (Apache 2.0). Trackers such as Artificial Analysis put the best open-weight models only a few points behind the closed leader on their composite index.

A practical selection process:

```
1. Hard constraints   data residency / on-prem? modality? context length? licence? -> removes most options
2. Shortlist          one model per tier (big, medium, small) from 1–2 vendors
3. Eval set           30–100 real examples with known answers (score them with the 1.15 metrics)
4. Measure            quality, p50/p95 latency, cost per task = in_tok × in_price + out_tok × out_price
5. Decide             the cheapest model that clears the quality bar; route hard cases to a bigger one
6. Pin and re-test    pin the exact model ID; re-run the eval before any upgrade
```

**Gotchas.**

- Leaderboards measure someone else's task. A model ranked third may be best on your invoices.
- Per-token prices are not directly comparable across vendors because tokenizers differ (Day 1), and reasoning tokens are billed as output (Day 4). Compare cost *per task*.
- Open weights are not free. You pay for GPUs, ops and scaling, and the same open model from two hosts can behave differently (quantization, different default settings).
- Model aliases move and old snapshots are retired. Pin versions in production and track deprecation notices.
- Bigger is not always better for narrow, high-volume steps: a small model with a good prompt and schema often matches it at a fraction of the cost and latency.

**UiPath bridge.** The AI Trust Layer is where an Automation Cloud admin governs which models agents may call, including bring-your-own LLM connections. That is model selection treated as policy, which is the same discipline as step 1 above.

**Interview angle.**

- *A bank wants an internal assistant over confidential documents. Open or closed model?* Start from constraints. If data may not leave its environment, use open weights on its own infrastructure, or a closed model through the bank's cloud tenancy with regional hosting and no-retention terms. Then compare quality on the bank's own eval set and weigh the ops cost of self-hosting against API spend.
- *How do you choose a model for a new feature?* Constraints first, then a shortlist across tiers, a task-specific eval set, and measurements of quality, latency and cost per task. Pick the cheapest that passes, pin the version, and add routing to a bigger model for the hard tail.

## 1.15 Classical ML refresher: supervised/unsupervised, overfitting, precision/recall/F1

**The idea.** Before LLMs, ML meant training a model for one task from data. Its vocabulary is still how AI engineers judge LLM systems: is this classifier, router, guardrail or retriever actually good, and will it stay good on new data?

**How it works.**

```
Supervised       labelled examples (input -> correct output)
                 classification (invoice / not invoice), regression (predict amount)
Unsupervised     no labels; find structure
                 clustering (group support tickets), dimensionality reduction (plot embeddings)
Self-supervised  labels come from the data itself: next-token prediction (LLM pretraining)
Reinforcement    learn from rewards: RLHF and reasoning training (Day 3, Day 4)
```

**Overfitting** means the model memorised its training data instead of learning the pattern, so it scores well on data it has seen and badly on new data. You detect it by holding data out: train on one split and report on a test split you never tuned on. The warning sign is a big gap between training and test scores. Fixes: more or more varied data, a simpler model, regularisation, early stopping, and cross-validation to get a steadier estimate.

The **confusion matrix** for a yes/no classifier, and the metrics built from it:

```
                     predicted YES        predicted NO
actual YES           TP (true pos.)       FN (missed)
actual NO            FP (false alarm)     TN

precision = TP / (TP + FP)    "when it says yes, how often is it right?"
recall    = TP / (TP + FN)    "of all real yeses, how many did it catch?"
F1        = 2 · P · R / (P + R)   harmonic mean: high only when both are high
```

Worked example: 100 documents, 40 are real invoices. The model flags 35, and 30 of those are right. TP = 30, FP = 5, FN = 10. Precision = 30/35 ≈ 0.86, recall = 30/40 = 0.75, F1 ≈ 0.80.

Most classifiers output a score, and a **threshold** turns it into yes/no. Raising the threshold usually increases precision and lowers recall. Which matters more depends on the cost of each error. Favour precision when a false positive is expensive (auto-paying an invoice). Favour recall when a miss is expensive (fraud screening, PII detection, safety filters). For several classes, **macro** averaging weights each class equally and **micro** averaging weights each example equally.

**Gotchas.**

- Accuracy lies on imbalanced data. If 1% of transactions are fraud, a model that always says "not fraud" is 99% accurate and useless.
- You can overfit a *prompt* too. Tweaking a prompt until it passes 20 dev examples is training on your test set; keep a held-out set you only check at the end.
- Leakage: if the same document (or a near-duplicate) is in both train and test, the test score is inflated.
- F1 hides which error you are making. Always look at the confusion matrix too.
- Precision is undefined when the model predicts no positives at all; scikit-learn's `zero_division` argument decides what it reports.

**UiPath bridge.** Document Understanding confidence thresholds are exactly this trade-off. Raising the auto-accept threshold means fewer wrong extractions go straight through (higher precision), but more documents go to Action Center (lower recall for straight-through processing).

**Interview angle.**

- *Your fraud model is 99% accurate. Is it good?* Not necessarily. With 1% fraud, a model that never flags anything scores 99%. Report precision and recall on the fraud class, look at the confusion matrix, and choose a threshold based on the cost of a missed fraud against the cost of a false alarm.
- *How do you spot overfitting, and does the idea apply to LLM apps?* A big gap between training (or dev) performance and held-out test performance. Yes: prompts, few-shot examples and retrieval settings tuned on a small dev set can overfit it, so keep a separate held-out eval set and refresh it with real production cases.

## Hands-on (30–45 min): a vision invoice classifier, scored like an ML model

**Goal.** Build a small multimodal classifier (invoice vs not invoice, plus the total), run it on two models, score both with precision, recall and F1, and decide which you would ship. Then see overfitting and the threshold trade-off directly in scikit-learn.

**Setup.**

- `pip install pillow scikit-learn` (Pillow 10.1 or newer, for `load_default(size=...)`). Add both to `requirements.txt` yourself if you want them tracked.
- Uses the same `.env` as `check_setup.py`. Optional: add `LLM_MODEL_SMALL=` with a smaller or cheaper vision-capable model from the list `check_setup.py` prints (for example a Flash-Lite variant) so Part B compares two tiers. With only `LLM_MODEL` set, Part B runs one model.
- About 24 API calls per model, each with one small image, paced for the free tier.
- Put each part in its own file in a new `lessons/Day-05/Practice Exercise/` folder.

**Part A — what an image costs (1.13), about 5 min, no API.**

```python
import math

def claude_image_tokens(w, h, max_edge=1568, max_tokens=1568):
    """Approximate Claude standard tier: downscale to fit, then 28x28 patches."""
    scale = min(1.0, max_edge / max(w, h))
    while True:
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        tokens = math.ceil(tw / 28) * math.ceil(th / 28)
        if tokens <= max_tokens:
            return tokens, (tw, th)
        scale *= 0.97

def gemini_image_tokens(w, h):
    """Approximate Gemini rule: small image = 258 tokens, else 768x768 tiles at 258 each."""
    if w <= 384 and h <= 384:
        return 258
    return math.ceil(w / 768) * math.ceil(h / 768) * 258

for name, (w, h) in {"thumbnail": (300, 300), "screenshot": (1280, 720),
                     "A4 scan 200dpi": (1654, 2339), "4K photo": (3840, 2160)}.items():
    ct, size = claude_image_tokens(w, h)
    print(f"{name:15} {w}x{h:<5} Claude≈{ct:>5} tokens (sent as {size[0]}x{size[1]})  "
          f"Gemini≈{gemini_image_tokens(w, h):>5} tokens")
print("10-min call on Gemini audio:", 10 * 60 * 32, "tokens")
```

**Part B — vision classifier on two models, scored properly (1.13 + 1.14 + 1.15), about 25 min.** The data is synthetic so you know the labels. Quotations and delivery notes look like invoices on purpose (they test precision), and every fourth invoice is shrunk until it is hard to read (it tests recall and honesty about unreadable totals).

```python
import base64, io, json, os, random, re, time
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix

random.seed(7)
load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
MODELS = [m for m in (os.getenv("LLM_MODEL"), os.getenv("LLM_MODEL_SMALL")) if m]
KINDS = ["INVOICE"] * 12 + ["QUOTATION"] * 6 + ["DELIVERY NOTE"] * 3 + ["INTERNAL MEMO"] * 3

def make_doc(kind, i):
    total = round(random.uniform(1200, 98000), 2)
    img = Image.new("RGB", (800, 1000), "white")
    d = ImageDraw.Draw(img)
    big, body = ImageFont.load_default(size=40), ImageFont.load_default(size=22)
    d.text((40, 40), kind, font=big, fill="black")
    d.text((40, 110), f"No. {kind[:3]}-{1000 + i}   Date: 2026-10-{i % 28 + 1:02d}", font=body, fill="black")
    d.text((40, 150), "Bill to: Acme Logistics Pvt Ltd, Pune", font=body, fill="black")
    y = 220
    for n in range(3):
        d.text((40, y), f"Item {n + 1}: Bot maintenance hours x {random.randint(1, 9)}", font=body, fill="black")
        y += 40
    if kind != "INTERNAL MEMO":
        d.text((40, y + 40), f"TOTAL: INR {total:,.2f}", font=big, fill="black")
    if kind == "QUOTATION":
        d.text((40, y + 120), "Valid for 30 days. Payment terms on acceptance.", font=body, fill="black")
    if kind == "INVOICE" and i % 4 == 0:
        img = img.resize((200, 250))                      # deliberately hard to read
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), kind == "INVOICE", (total if kind != "INTERNAL MEMO" else None)

DOCS = [make_doc(k, i) for i, k in enumerate(KINDS)]

PROMPT = ("Is this document an invoice, meaning a request for payment for goods or services already supplied? "
          'Reply with JSON only: {"is_invoice": true or false, "total": number or null}. '
          "Use null if the total is not clearly readable. Do not guess.")

def ask(model, png):
    b64 = base64.b64encode(png).decode()
    t0 = time.perf_counter()
    r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},   # image first
        {"type": "text", "text": PROMPT}]}])
    secs = time.perf_counter() - t0
    text = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    return out, secs, r.usage.prompt_tokens

for model in MODELS:
    y_true, y_pred, secs_all, toks, totals_ok, totals_n = [], [], [], [], 0, 0
    for png, is_inv, total in DOCS:
        out, secs, ptok = ask(model, png)
        y_true.append(int(is_inv))
        y_pred.append(int(bool(out.get("is_invoice"))))
        secs_all.append(secs)
        toks.append(ptok)
        if is_inv:
            totals_n += 1
            try:
                totals_ok += abs(float(str(out.get("total")).replace(",", "")) - total) < 0.01
            except ValueError:
                pass                                      # null or unreadable total
        time.sleep(4)                                     # stay under free-tier rate limits
    print(f"\n=== {model} ===")
    print(classification_report(y_true, y_pred, target_names=["not_invoice", "invoice"], zero_division=0))
    print("confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))
    print(f"exact totals on invoices: {totals_ok}/{totals_n}")
    print(f"mean latency {sum(secs_all) / len(secs_all):.1f}s | mean prompt tokens per call {sum(toks) / len(toks):.0f}")
```

**Part C — overfitting and the threshold, in pure scikit-learn (1.15), about 10 min.**

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=600, n_features=20, n_informative=5, weights=[0.85],
                           flip_y=0.05, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

print("1) Overfitting: tree depth vs F1")
for depth in (1, 2, 3, 5, 8, None):
    tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_tr, y_tr)
    print(f"  depth={str(depth):4}  train F1={f1_score(y_tr, tree.predict(X_tr)):.2f}  "
          f"test F1={f1_score(y_te, tree.predict(X_te)):.2f}")

print("\n2) Accuracy trap: always predict 0 ->", f"{(y_te == 0).mean():.2f} accuracy, recall 0.00")

print("\n3) Threshold trade-off (logistic regression)")
proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
for t in (0.2, 0.35, 0.5, 0.65, 0.8):
    pred = (proba >= t).astype(int)
    print(f"  threshold={t:.2f}  precision={precision_score(y_te, pred, zero_division=0):.2f}  "
          f"recall={recall_score(y_te, pred):.2f}  F1={f1_score(y_te, pred):.2f}")
```

**Done when:** (1) Part A prints token estimates and you can say why a 4K photo costs no more than an A4 scan on Claude (both are downscaled to the cap); (2) Part B prints a classification report, confusion matrix, totals score, latency and tokens for each model, and you can name which documents caused the false positives (likely quotations) and false negatives (likely the shrunk invoices); (3) Part C shows the training F1 reaching about 1.0 while the test F1 stalls or drops as depth grows, and precision rising while recall falls as the threshold goes up; (4) you have written three lines in a Day-5 README: which model you would ship for this task and why (quality, latency, cost per document), and whether you would tune for precision or recall if the output auto-posted invoices to an ERP. Bonus: add "This is a quotation, not an invoice." to only half of the quotations and see whether precision changes.

## Go deeper (optional)

- [Vision (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/vision) — the patch-based token formula, resolution tiers, limits and the known weak spots of image understanding.
- [classification_report (scikit-learn docs)](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) — precision, recall, F1 and the macro, weighted and micro averages explained alongside the function you use in Part B.

## Recall answers

1. Decode. The 1,000 output tokens are 1,000 sequential forward passes, each reading the whole KV cache. The 2,000 prompt tokens are processed in one parallel prefill pass.
2. Any two of: ground the answer in provided sources and say "answer only from them"; give an explicit "not found" path; require quotes or citations and verify them in code; constrain the output with a schema or enum; add a checker model or self-consistency; measure with an eval set that includes unanswerable questions.
3. The thinking is generated text, not a trace of the computation. Research found reasoning models often fail to mention hints they actually used, so the visible chain of thought can leave out what really drove the answer.
