# Day 9 — Prompts as specifications: clear instructions, examples, and room to reason (2026-11-07)

Module 2 taught you how to send a request. Module 3 is about what you put in it. Today covers the three levers you will use on almost every prompt: say exactly what you want and why (3.1), show what good looks like (3.2), and give the model room to work through hard steps before it answers (3.3). The hands-on builds a small "prompt ladder" and measures how much each lever helps. Reading ≈ 20 min, hands-on ≈ 40 min.

## Recall warm-up (2 min)

1. Batch results come back as a JSONL file. How do you match each result to its input, and why not by position?
2. You fire 200 async LLM calls with `asyncio.gather` and nothing else. What goes wrong, and what is the standard fix?
3. Name two Claude API features from Day 8 that you may lose by running Claude on Bedrock, Vertex AI or Foundry.

## 3.1 Clarity, specificity, context

**The idea.** A model knows a lot about the world and nothing about your situation. It has not read your process document, does not know who reads its output, and cannot ask you what you meant. When a prompt leaves something open, the model fills the gap with the most typical answer, which is often not yours. Most "the model is dumb" bugs are really "the prompt was ambiguous" bugs. Clarity is the cheapest fix in LLM engineering: no new infrastructure, just a better specification.

Anthropic's docs frame it as briefing a capable new hire who lacks your context. Their test: show the prompt to a colleague who knows nothing about the task. If they would be confused or need to ask questions, so will the model.

**How it works.** A clear prompt answers five questions. Here is the same task, vague then specified:

```
VAGUE:  "Should we approve this invoice?"

CLEAR:
  Context   who it's for and why      "A bot reads your label and routes the invoice.
                                       auto_approve skips human review, so a wrong
                                       one costs money."
  Task      the exact job             "Classify the invoice into one of 4 routes."
  Rules     the decision logic        "1. Recompute the total; mismatch > 1.00 → reject
                                       2. Total > 25,000 → director_approval ..."
  Format    what the output looks like "Exactly one label, no other text."
  Edges     what to do when unsure    "Vendor names must match exactly; when in
                                       doubt, prefer a human queue."
```

Three habits do most of the work:

- **Give the reason behind a rule.** "Never use ellipses" is a rule; "the output goes to a text-to-speech engine that can't pronounce ellipses" lets the model generalize to other things a speech engine can't read. Anthropic's current guidance uses exactly this example.
- **Be specific about the output.** Length, format, label set, language, what to leave out. If code will parse the result, say so, or use structured outputs (Day 7).
- **Ask for what you actually want.** Modern models follow instructions literally. If you want a thorough answer, ask for one; "suggest changes" and "make the changes" get different behaviour.

**Gotchas.**

- **Longer is not clearer.** A 3-page prompt full of hedges and repeated warnings buries the real rules. Every sentence should change the output; delete the ones that don't.
- **SHOUTING and "CRITICAL!!!" backfire on current models.** Older models needed emphasis to notice a rule; newer ones over-apply it, for example refusing borderline cases or overusing a tool. State the rule calmly and give the reason.
- **Say what to do, not only what not to do.** "Don't use markdown" works less well than "write in plain paragraphs."
- **The rules you forgot are the bugs.** Look for edge cases in real data (missing fields, near-duplicate names, values right on a threshold) and write a rule for each.
- **Clarity cannot replace missing data.** If the answer depends on facts the model doesn't have (today's exchange rate, your vendor list), put the facts in the prompt or give it a tool (Module 6).

**UiPath bridge.** It is the PDD problem: a bot developer handed a process document with no exception-handling section will guess, and so will a model. The system and user prompt fields in Agent Builder are where your "PDD for the model" lives.

**Interview angle.**

- *A prompt works in testing but gives inconsistent results in production. What do you check first?* Look at the failing inputs. Usually they hit a case the prompt doesn't cover, such as a missing field, an ambiguous category or a threshold value. Add explicit rules and the reason behind them, specify the output format, then add those inputs to the eval set so the fix stays fixed.
- *What makes a prompt "good"?* It is unambiguous to someone without your context, it states the task, the audience and purpose, the decision rules, the output format and how to handle edge cases, and it is measured against a test set rather than judged by eyeballing one output.

## 3.2 Few-shot examples

**The idea.** Some things are easier to show than to describe: a tone of voice, an exact output layout, where the line falls between two categories. Few-shot (also called multishot) prompting puts a handful of solved input → output pairs in the prompt. The model picks up the pattern without any training. The GPT-3 paper (Brown et al., 2020) made this famous as *in-context learning*: the model's weights do not change; the examples only steer the next-token predictions for this request. Zero-shot means no examples; one-shot means one; few-shot usually means 2–10.

**How it works.** Wrap the examples so the model can tell them apart from instructions and from the real input:

```
system:
  <instructions + rules>
  <examples>
    <example>
      <invoice> ...EX-1... </invoice>
      <output> auto_approve </output>
    </example>
    <example> ...EX-2: a total mismatch → reject... </example>
    <example> ...EX-3: unapproved vendor → manager_approval... </example>
  </examples>
user:
  <invoice> ...the real one... </invoice>
```

Anthropic's current guidance: 3–5 examples, *relevant* (they look like real inputs), *diverse* (they cover edge cases and different labels), and wrapped in `<example>` / `<examples>` tags. You can also put examples as earlier user/assistant turns in the messages list; both work.

What the model actually learns from examples is mostly the *format*, the *label set* and the *kind of input*. A well-known study (Min et al., 2022) found that even randomly wrong labels in demonstrations hurt accuracy surprisingly little on classification tasks, while removing the format or the input distribution hurt a lot. So examples are excellent for format and boundary cases, and weaker as a way to teach complex logic. Rules teach logic; examples show how to apply it.

**Gotchas.**

- **The model copies surface features.** If all your examples are short, the answers get short. If the example vendor is always "Acme", the model may treat Acme as special. Vary length, wording and labels.
- **Label balance and order bias.** Three `approve` examples and one `reject` nudge the model toward `approve`, and the last example has extra pull (Zhao et al., 2021 called these majority-label and recency bias). Balance the labels and shuffle the order.
- **Never reuse test cases as examples.** The model will get those right for the wrong reason, and your eval score becomes meaningless.
- **Examples cost tokens on every call.** Five 300-token examples add 1,500 input tokens to every request. Put them in the stable prefix so prompt caching (Day 7) makes them about 10% of the normal price after the first call.
- **Examples can fight the instructions.** If an example's answer breaks one of your written rules, the model has to choose, and it often follows the example. Generate example answers from the same logic you test against (the hands-on does this in code).

**UiPath bridge.** Like giving Document Understanding a few labelled samples, except nothing is trained: the examples travel inside every request, and you pay for their tokens every time.

**Interview angle.**

- *When would you use few-shot examples rather than fine-tuning?* Almost always first: examples cost nothing to change, take effect instantly and need only a handful of cases. Fine-tune only when you have hundreds or thousands of examples, the behaviour can't fit in a prompt, or you need to cut per-call tokens or latency at high volume (Module 9).
- *Your few-shot classifier over-predicts one class. Why, and what do you do?* Likely label imbalance or recency bias in the examples, or examples that are all too similar. Balance and shuffle them, add boundary cases, and move the decision logic into explicit rules. Then measure on a held-out set.

## 3.3 Chain-of-thought

**The idea.** A model produces its answer one token at a time, and each token gets a fixed amount of computation. If you force the answer immediately ("output one label"), all the arithmetic and rule-checking must happen within the few tokens of that label. Chain-of-thought (CoT) prompting lets the model write out intermediate steps first. Every step it writes becomes context it can read for the next step, so it effectively gets scratch paper and more computation. Wei et al. (2022) showed that adding worked reasoning to few-shot examples sharply improved large models on maths and multi-step problems. Kojima et al. (2022) found that simply adding "Let's think step by step" helped too (*zero-shot CoT*).

**How it works.** There are three ways to get reasoning, from oldest to newest:

```
1. Zero-shot CoT    "Think step by step, then answer."
2. Few-shot CoT     examples show  problem → worked reasoning → answer
                    (the model copies the reasoning style, not just the label)
3. Built-in thinking reasoning models (Day 4, topic 1.12) think in a separate
                    channel before answering; you control how much with an
                    effort/budget setting, not with prompt wording
```

For manual CoT, separate the reasoning from the answer so code can parse the result:

```
<reasoning>
Subtotal = 10 x 430.00 = 4,300.00. With 18% tax = 5,074.00.
Stated 5,074.00 → no mismatch. Not above 25,000. Approved vendor, PO present,
but 5,074.00 > 5,000.00 → rule 3.
</reasoning>
<answer>manager_approval</answer>
```

Your code reads only `<answer>`. The reasoning is for the model's benefit and for your debugging.

**Current state (checked 2026-11-07, Claude prompting best-practices docs).** On current Claude models, thinking is built in and adaptive: the model decides how much to think, guided by an `effort` setting. Thinking is always on for Opus 5.5 and on by default for Sonnet 5.5 and Haiku 5.5. Anthropic's advice for these models is to rely on built-in thinking (at lower effort if cost matters) rather than asking for visible reasoning in `<thinking>` tags, which these models may decline. A general instruction such as "think thoroughly" tends to beat a hand-written step-by-step plan. Manual CoT with `<answer>` tags remains the tool for models or settings where thinking is off. Gemini and OpenAI models similarly expose a reasoning-effort setting. So in 2026 "use CoT" mostly means "choose the right thinking setting for the task," plus knowing the manual technique for non-reasoning and small models.

**Gotchas.**

- **It costs tokens and latency.** Reasoning tokens are output tokens, the expensive kind (Day 6). A 5-token label can become 150 tokens. Use it where it changes accuracy, not everywhere.
- **It helps multi-step problems, not simple lookups.** Sentiment of a one-line review doesn't need reasoning. Arithmetic, rule chains, comparisons and planning do. In the original paper the gains appeared mainly in large models; small models could produce confident but wrong chains.
- **The written reasoning may not be the real reason.** Research, including Anthropic's 2025 work on reasoning models, shows stated reasoning is not always faithful to what drove the answer. Treat it as a useful debugging signal, not an audit trail.
- **Never show raw reasoning to end users by default,** and never parse it. Extract the answer tag. If the answer tag is missing (for example, output cut off by `max_tokens`), treat the call as failed.
- **Don't double up.** Asking a reasoning model to also "think step by step" in visible text wastes tokens and can be declined. Pick one: built-in thinking or manual CoT.

**UiPath bridge.** It is like adding a Log Message before each decision in a workflow so you can see why a transaction became a Business Exception, with one difference: the model's "log" helps it compute, but may not be the true cause of its decision.

**Interview angle.**

- *What is chain-of-thought and why does it work?* Prompting the model to write intermediate steps before the final answer. Each generated token adds computation and becomes context for later tokens, so multi-step problems get broken into easy steps instead of being solved in one jump. It helps most on maths, logic and multi-rule decisions.
- *With reasoning models, is CoT prompting still relevant?* Less as a prompt trick, more as a design choice. You choose thinking on or off and the effort level per task, trading accuracy against latency and cost. Manual CoT with answer tags still matters for non-reasoning or small models, and the idea underlies agent loops (Module 7).

## Hands-on (30–45 min): a prompt ladder for invoice triage

**Goal.** Route 12 supplier invoices into `auto_approve`, `manager_approval`, `director_approval` or `reject`, using four prompt versions that each add one of today's levers. Measure accuracy and output tokens for each version, and find which lever fixes which mistakes. The test set has deliberate traps: tax that pushes a total over a threshold, a stated total that left out tax, a near-miss vendor name ("Acme Supplies Ltd"), a 0.60 rounding difference that should be tolerated, and rule-precedence conflicts.

| Rung | Adds | Topic |
| --- | --- | --- |
| v0 | Bare question + label list | baseline |
| v1 | Context, ordered rules, exact output format | 3.1 |
| v2 | v1 + 3 labelled examples in `<examples>` | 3.2 |
| v3 | v2 with worked reasoning in the examples + `<reasoning>` / `<answer>` output | 3.3 |

**Setup.** Your existing `.env` (`GEMINI_API_KEY`, `LLM_MODEL`) and `requirements.txt` (`openai`, `python-dotenv`). Nothing new to install. Each full ladder is 48 short calls; the script waits `PACE_SECONDS` (default 4) between calls to stay under free-tier rate limits, so `all` takes about 4–5 minutes. Save as `day09_prompt_ladder.py` in your own Day-09 exercise folder. If your model rejects `reasoning_effort` or a value of it, remove the argument from the `create(...)` call. The ground-truth labels are computed in code from the same rules the prompt states, so there are no hand-labelling errors.

```python
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
```

**Steps.**

1. `python day09_prompt_ladder.py dry` (5 min). No API calls. Read the expected labels and the four prompts for INV-104. For each trap invoice (103, 104, 105, 106, 108, 109), write down which rule decides it.
2. `python day09_prompt_ladder.py all` (10 min). Record accuracy and average completion tokens per rung in a small table. Note which invoices v0 gets wrong. Expect it to approve INV-106, because nothing told it to check the stated total against the line items.
3. Diagnose (10 min). Open `results_v1_low.json` and `results_v2_low.json`. For every remaining `XX`, decide: is it a clarity gap (fix the rules), a format/boundary issue (fix with an example), or an arithmetic/precedence slip (needs reasoning)? Make one targeted edit (a rule wording or a fourth example) and re-run just that rung.
4. Prompted CoT vs built-in thinking (10 min). Run `python day09_prompt_ladder.py v2 high` and compare it with `v3 low`. Which gives better accuracy per output token? This is the 2026 version of the CoT question: prompt for visible reasoning, or turn up the model's own thinking.
5. Bias check (5 min). In `EXAMPLES`, change EX-2 and EX-3 so all three examples are `auto_approve`, re-run `v2`, and see whether errors shift toward `auto_approve`. Restore them afterwards.

**Done when:** (1) you have a table of accuracy and average completion tokens for v0–v3 (plus `v2 high`); (2) you can name, for at least three trap invoices, which lever fixed them; (3) you made one evidence-based prompt edit and showed its effect on the score; (4) you can say in one sentence when you would pay for reasoning tokens on this task in production.

## Go deeper (optional)

- [Prompting best practices (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): Anthropic's current guide. Covers clarity and context, how to write examples, and how to steer adaptive thinking and effort on current models.
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)](https://arxiv.org/abs/2201.11903): the original CoT paper. Read the abstract, Figure 1 and the section on scale.

## Recall answers

1. By `custom_id`, which you set on each request and which comes back with each result. Results arrive in any order, so zipping by position silently mislabels data.
2. All 200 start at once, burst past your RPM and token-per-minute limits, and trigger a storm of 429s (and retry pile-ups). Cap in-flight calls with an `asyncio.Semaphore` sized to your rate limit, and keep one retry layer with backoff.
3. Any two of: the Message Batches API, the Files API, the Models API, some server-side tools. New features also tend to arrive later on the cloud platforms.

