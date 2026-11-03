import base64, io, json, os, random, re, sys, time
import openai
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
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},   # image first
            {"type": "text", "text": PROMPT}]}])
    except openai.RateLimitError:
        sys.exit("429: free-tier rate limit hit. Wait a minute and rerun, or try a Flash-Lite model.")
    except openai.NotFoundError:
        sys.exit(f"404: model '{model}' not found. Run check_setup.py with LLM_MODEL removed to list valid IDs.")
    except openai.APIStatusError as e:
        sys.exit(f"HTTP {e.status_code}: {e.message}")
    secs = time.perf_counter() - t0
    text = r.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    return out, secs, r.usage.prompt_tokens

if not MODELS:
    sys.exit("Set LLM_MODEL (and optionally LLM_MODEL_SMALL) in .env")

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
