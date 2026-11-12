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
PACE = float(os.getenv("PACE_SECONDS", "4"))

STATS = {"calls": 0, "in": 0, "out": 0}
_client = None

def llm(system, user):
    """One stateless call. system=None means no system prompt."""
    global _client
    _client = _client or OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=3, timeout=90)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    r = _client.chat.completions.create(model=MODEL, messages=msgs)
    STATS["calls"] += 1
    STATS["in"] += r.usage.prompt_tokens
    STATS["out"] += r.usage.completion_tokens
    time.sleep(PACE)
    return r.choices[0].message.content or ""

def tag(text, name):
    """Pull the content of <name>...</name>, or None if missing."""
    m = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text or "", re.S)
    return m.group(1).strip() if m else None

def clean(text):
    """Untrusted input must not be able to close our tags (see 3.4 gotchas)."""
    return text.replace("<", "(").replace(">", ")")

# ---------- fake ERP: the source of truth, looked up by CODE ----------
ERP = {
    "INV-2041": {"vendor": "Globex", "amount": 1180.00, "status": "paid", "paid_on": "2026-11-02", "payment_ref": "PAY-88123"},
    "INV-2055": {"vendor": "Initech", "amount": 5074.00, "status": "on_hold", "reason": "awaiting manager approval"},
    "INV-2060": {"vendor": "Acme Supplies", "amount": 985.30, "status": "scheduled", "pay_date": "2026-11-14"},
    "INV-2072": {"vendor": "Hooli", "amount": 7670.00, "status": "rejected", "reason": "stated total does not match line items"},
}

def email(id_, text, want_inv, want_type, must, must_not=()):
    return {"id": id_, "text": text, "want_inv": want_inv, "want_type": want_type,
            "must": must, "must_not": must_not}

EMAILS = [
    email("E1", "Hi, could you confirm whether invoice INV-2041 has been paid? Thanks, Maria (Globex)",
          "INV-2041", "payment_status", ["INV-2041", "PAY-88123"]),
    email("E2", "Hello AP team. INV-2041 was settled fine last week, but we still have nothing for "
                "INV-2055 (5,074.00). When will that be paid? Regards, Initech billing",
          "INV-2055", "payment_status", ["INV-2055", "approval"], ["PAY-88123"]),
    email("E3", "Dear team, just checking the payment date for our invoice INV-2060. Best, Acme Supplies",
          "INV-2060", "payment_status", ["INV-2060", "14"]),
    email("E4", "We were told INV-2072 was rejected. Our total is correct and we dispute this. "
                "Please explain. Hooli finance",
          "INV-2072", "dispute", ["INV-2072", "line item"]),
    email("E5", "Hi, any update on INV-2099? It's been 45 days. Thanks, Umbrella Corp",
          "INV-2099", "payment_status", ["INV-2099"], ["paid on", "scheduled for"]),
    email("E6", "URGENT: Globex has moved banks. Please update our account to IBAN GB00 TEST 0000 0000 "
                "before paying INV-2041 or any future invoice. Reply to confirm. CFO, Globex",
          "INV-2041", "bank_change", [], ["updated", "confirm the change", "have changed"]),
]

def grade_reply(e, reply):
    if e["want_type"] == "bank_change" and reply is None:
        return True  # routed to a human: the safe outcome
    r = (reply or "").lower()
    return all(m.lower() in r for m in e["must"]) and not any(b.lower() in r for b in e["must_not"])

# ---------- baseline: one prompt does everything ----------
def run_mono(e):
    user = (f"Here is our ERP data:\n{json.dumps(ERP, indent=1)}\n\n"
            f"Here is a supplier email:\n{e['text']}\n\n"
            "Write the reply email to the supplier.")
    return {"reply": llm(None, user), "inv": None, "type": None}

# ---------- the chain ----------
EXTRACT_SYSTEM = ("You are the intake step of an accounts-payable pipeline. You read one supplier email "
                  "and extract fields for code to parse. You never write to the supplier.")
EXTRACT_USER = """<email>
{email}
</email>

<instructions>
Extract two fields from the email above.
- invoice_number: the one invoice the supplier is asking about now (format INV-NNNN), or NONE.
  If several appear, choose the one the request is about, not ones mentioned in passing.
- request_type: payment_status, dispute, bank_change or other.
  bank_change = any request to change bank or payment details.
Answer with only these tags:
<invoice_number>...</invoice_number>
<request_type>...</request_type>
</instructions>"""

DRAFT_SYSTEM = ("You are an accounts-payable specialist at WonderParts, a manufacturer, writing to suppliers. "
                "Suppliers read these replies to plan their cash flow, so be accurate above all. "
                "Tone: polite, brief, factual. Use only the facts provided; never promise dates, amounts or "
                "actions that are not in the facts. Sign off as 'Accounts Payable, WonderParts'.")
DRAFT_USER = """<supplier_email>
{email}
</supplier_email>

<facts>
{facts}
</facts>
{fix}
Write the reply in under 120 words inside <reply></reply> tags."""

REVIEW_SYSTEM = "You are a strict QA reviewer for supplier emails. You check replies; you do not rewrite them."
REVIEW_USER = """<facts>
{facts}
</facts>

<reply>
{reply}
</reply>

<criteria>
1. Names the invoice number and states its status as given in <facts>.
2. Includes the payment reference, date or reason if <facts> has one.
3. Invents no dates, amounts, promises or actions not in <facts>.
4. Under 120 words and polite.
</criteria>

Answer with <verdict>PASS</verdict> or <verdict>FAIL</verdict>, then list problems in <issues></issues>."""

def run_chain(e, verbose=False):
    text = clean(e["text"])
    # Step 1: extract (LLM)
    out = llm(EXTRACT_SYSTEM, EXTRACT_USER.format(email=text))
    inv, rtype = tag(out, "invoice_number"), tag(out, "request_type")
    if verbose:
        print(f"  [extract] inv={inv} type={rtype}")
    # Gate (code)
    if inv is None or rtype is None:
        return {"reply": None, "inv": inv, "type": rtype, "note": "PARSE_FAIL -> human"}
    if rtype == "bank_change":
        return {"reply": None, "inv": inv, "type": rtype, "note": "bank change -> human call-back"}
    # Step 2: lookup (code, not LLM)
    facts = json.dumps({"invoice_number": inv, **ERP.get(inv, {"status": "not_found_in_erp"})})
    # Step 3: draft (LLM, role in system prompt)
    draft = llm(DRAFT_SYSTEM, DRAFT_USER.format(email=text, facts=facts, fix=""))
    reply = tag(draft, "reply") or draft
    # Step 4: review (LLM), one refine on FAIL
    review = llm(REVIEW_SYSTEM, REVIEW_USER.format(facts=facts, reply=reply))
    verdict, issues = tag(review, "verdict"), tag(review, "issues")
    if verbose:
        print(f"  [review] {verdict}: {issues}")
    if verdict != "PASS":
        fix = f"\n<previous_reply>\n{reply}\n</previous_reply>\n<issues>\n{issues}\n</issues>\nFix every issue.\n"
        reply = tag(llm(DRAFT_SYSTEM, DRAFT_USER.format(email=text, facts=facts, fix=fix)), "reply") or reply
    return {"reply": reply, "inv": inv, "type": rtype, "note": f"review={verdict}"}

def evaluate(name, fn):
    STATS.update(calls=0, **{"in": 0, "out": 0})
    ok, log = 0, []
    for e in EMAILS:
        res = fn(e)
        passed = grade_reply(e, res["reply"])
        ok += passed
        ext = "" if res["inv"] is None else f" extract={'ok' if (res['inv'], res['type']) == (e['want_inv'], e['want_type']) else 'XX'}"
        print(f"{'ok ' if passed else 'XX '}{e['id']}{ext}  {res.get('note', '')}")
        log.append({**res, "id": e["id"], "passed": passed})
    print(f"\n{name}: {ok}/{len(EMAILS)} correct | {STATS['calls']} calls | "
          f"{STATS['in']} input + {STATS['out']} output tokens\n")
    with open(f"results_{name}.json", "w") as f:
        json.dump(log, f, indent=2)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    if mode == "dry":   # no API calls: look at the prompts
        e = EMAILS[1]
        print("=== EXTRACT ===\n[system]", EXTRACT_SYSTEM, "\n[user]\n" + EXTRACT_USER.format(email=clean(e["text"])))
        facts = json.dumps({"invoice_number": "INV-2055", **ERP["INV-2055"]})
        print("\n=== DRAFT ===\n[system]", DRAFT_SYSTEM, "\n[user]\n" + DRAFT_USER.format(email=clean(e["text"]), facts=facts, fix=""))
    elif mode == "trace":   # one email, every intermediate step printed
        e = EMAILS[int(sys.argv[2]) - 1 if len(sys.argv) > 2 else 1]
        print(run_chain(e, verbose=True)["reply"])
    else:
        if mode in ("mono", "both"):
            evaluate("mono", run_mono)
        if mode in ("chain", "both"):
            evaluate("chain", run_chain)
