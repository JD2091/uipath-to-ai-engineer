import json
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
MODEL = os.getenv("LLM_MODEL", "")
PACE = float(os.getenv("PACE_SECONDS", "4"))  # free tier: stay under the RPM limit

LABELS = ["auto_approve", "manager_approval", "director_approval", "reject"]
APPROVED = {"Acme Supplies", "Globex", "Initech"}
TAX = 0.18

# ---------- ground truth: the business rules as code ----------
def subtotal(inv):
    return round(sum(q * p for _, q, p in inv["lines"]), 2)

def computed_total(inv):
    return round(subtotal(inv) * (1 + TAX), 2)

def expected(inv):
    total = computed_total(inv)
    if abs(total - inv["stated"]) > 1.00:
        return "reject"
    if total > 25000:
        return "director_approval"
    if not inv["po"] or inv["vendor"] not in APPROVED or total > 5000:
        return "manager_approval"
    return "auto_approve"

def inv(id_, vendor, po, lines, stated):
    return {"id": id_, "vendor": vendor, "po": po, "lines": lines, "stated": stated}

TEST = [
    inv("INV-101", "Acme Supplies", "PO-7781", [("Printer toner", 6, 85.00), ("A4 paper (box)", 10, 32.50)], 985.30),
    inv("INV-102", "Globex", "", [("Laptop stand", 8, 45.00)], 424.80),
    inv("INV-103", "Acme Supplies Ltd", "PO-7790", [("Office chair", 4, 310.00)], 1463.20),
    inv("INV-104", "Initech", "PO-7802", [("Monitor 27in", 10, 430.00)], 5074.00),
    inv("INV-105", "Globex", "PO-7810", [("Docking station", 5, 189.99)], 1120.34),
    inv("INV-106", "Initech", "PO-7815", [("Software licence", 20, 240.00)], 4800.00),
    inv("INV-107", "Acme Supplies", "PO-7820", [("Server rack", 2, 12500.00), ("Install service", 1, 1800.00)], 31624.00),
    inv("INV-108", "Globex", "", [("Firewall appliance", 1, 23900.00)], 28202.00),
    inv("INV-109", "Initech", "PO-7833", [("Laptop", 25, 1150.00)], 28750.00),
    inv("INV-110", "Acme Supplies", "PO-7840", [("Ergonomic keyboard", 12, 389.50)], 5515.32),
    inv("INV-111", "Globex", "PO-7851", [("Webcam", 14, 299.00)], 4939.48),
    inv("INV-112", "Umbrella Corp", "PO-7860", [("Cleaning service", 1, 950.00)], 1121.00),
]

# Few-shot examples: NOT from the test set
EXAMPLES = [
    inv("EX-1", "Initech", "PO-7001", [("USB-C cable", 30, 12.00)], 424.80),
    inv("EX-2", "Globex", "PO-7002", [("Desk lamp", 10, 40.00)], 400.00),
    inv("EX-3", "Hooli", "PO-7003", [("Consulting day", 5, 1300.00)], 7670.00),
]

def render(i):
    lines = "\n".join(f"  - {d}: {q} x {p:,.2f}" for d, q, p in i["lines"])
    return (f"Invoice {i['id']}\nVendor: {i['vendor']}\nPO number: {i['po'] or '(none)'}\n"
            f"Line items:\n{lines}\nTax rate: 18%\nStated total: {i['stated']:,.2f}")

def worked(i):
    """A worked solution (problem -> method -> answer) for CoT examples."""
    parts = " + ".join(f"{q} x {p:,.2f}" for _, q, p in i["lines"])
    total, label = computed_total(i), expected(i)
    diff = abs(total - i["stated"])
    return (f"Subtotal = {parts} = {subtotal(i):,.2f}. With 18% tax = {total:,.2f}. "
            f"Stated {i['stated']:,.2f}, difference {diff:,.2f}. "
            f"Vendor approved: {i['vendor'] in APPROVED}. PO present: {bool(i['po'])}. "
            f"First matching rule gives {label}.")

# ---------- 3.1 clarity: context + rules + format ----------
CONTEXT = (
    "You triage supplier invoices for an accounts-payable automation. A bot reads your label and "
    "routes the invoice. auto_approve posts it to the ERP with no human review, so a wrong "
    "auto_approve costs real money; the human queues are the safe side."
)
RULES = """Apply these rules in order and stop at the first one that matches:
1. Compute the total yourself: the sum of quantity x unit price over all lines, plus 18% tax. If the stated total differs from your computed total by more than 1.00, the label is reject.
2. If the computed total is above 25,000.00, the label is director_approval.
3. If the PO number is missing, OR the vendor is not exactly one of Acme Supplies, Globex, Initech (exact name match; any variation is not approved), OR the computed total is above 5,000.00, the label is manager_approval.
4. Otherwise the label is auto_approve."""
FORMAT_LABEL = "Output exactly one label: auto_approve, manager_approval, director_approval or reject. No other text."
FORMAT_COT = ("First reason through rules 1-4 in order inside <reasoning> tags, writing out the arithmetic. "
              "Then give exactly one label inside <answer></answer> tags.")

def examples_block(cot=False):
    out = []
    for e in EXAMPLES:
        answer = (f"<reasoning>{worked(e)}</reasoning>\n<answer>{expected(e)}</answer>"
                  if cot else expected(e))
        out.append(f"<example>\n<invoice>\n{render(e)}\n</invoice>\n<output>\n{answer}\n</output>\n</example>")
    return "<examples>\n" + "\n".join(out) + "\n</examples>"

def build(version, i):
    """Return (system, user) for each rung of the ladder."""
    text = render(i)
    if version == "v0":   # vague: no context, no rules
        return None, f"What should we do with this invoice?\n\n{text}\n\nAnswer with one of: {', '.join(LABELS)}."
    if version == "v1":   # 3.1 clear and specific
        return f"{CONTEXT}\n\n{RULES}\n\n{FORMAT_LABEL}", f"<invoice>\n{text}\n</invoice>"
    if version == "v2":   # 3.2 + few-shot
        return f"{CONTEXT}\n\n{RULES}\n\n{examples_block()}\n\n{FORMAT_LABEL}", f"<invoice>\n{text}\n</invoice>"
    if version == "v3":   # 3.3 + chain-of-thought with worked examples
        return f"{CONTEXT}\n\n{RULES}\n\n{examples_block(cot=True)}\n\n{FORMAT_COT}", f"<invoice>\n{text}\n</invoice>"
    raise ValueError(version)

def parse(reply):
    m = re.search(r"<answer>\s*(\w+)\s*</answer>", reply or "")
    if m and m.group(1) in LABELS:
        return m.group(1)
    found = re.findall("|".join(LABELS), reply or "")
    return found[-1] if found else "PARSE_FAIL"

def run(version, effort):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=3, timeout=90)
    rows, correct, out_tokens = [], 0, 0
    for i in TEST:
        system, user = build(version, i)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        resp = client.chat.completions.create(model=MODEL, messages=msgs, reasoning_effort=effort)
        reply = resp.choices[0].message.content or ""
        got, want = parse(reply), expected(i)
        correct += got == want
        out_tokens += resp.usage.completion_tokens
        mark = "ok " if got == want else "XX "
        print(f"{mark}{i['id']}  want={want:<18} got={got:<18} out_tokens={resp.usage.completion_tokens}")
        rows.append({"id": i["id"], "want": want, "got": got, "reply": reply})
        time.sleep(PACE)
    print(f"\n{version} (effort={effort}): {correct}/{len(TEST)} correct, "
          f"{out_tokens / len(TEST):.0f} avg completion tokens")
    with open(f"results_{version}_{effort}.json", "w") as f:
        json.dump(rows, f, indent=2)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    effort = sys.argv[2] if len(sys.argv) > 2 else "low"
    if mode == "dry":   # no API calls: inspect labels and one prompt per rung
        for i in TEST:
            print(i["id"], computed_total(i), expected(i))
        for v in ["v0", "v1", "v2", "v3"]:
            s, u = build(v, TEST[3])
            print(f"\n===== {v} =====\n[system]\n{s}\n[user]\n{u}")
    else:
        for v in (["v0", "v1", "v2", "v3"] if mode == "all" else [mode]):
            run(v, effort)
