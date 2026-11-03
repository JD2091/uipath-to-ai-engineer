import os
import sys
import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
MODEL = os.getenv("LLM_MODEL")

CONTEXT = ("Policy INV-7: Invoices above 50,000 INR need a second approver. "
           "Approvers must respond within 2 business days.")
QUESTIONS = [
    "Summarise the 2021 paper 'Recursive Queue Attention for Robotic Process Mining' by Hemant Vora.",
    "What is the default value of the UiPath Orchestrator setting 'Queue.AutoRetryJitterMs'?",
    "What does HTTP status code 429 mean?",
    f"Context: {CONTEXT}\n\nQuestion: What happens if an approver does not respond within 2 business days?",
]
GUARDED = ("Answer only if you are confident the thing asked about exists and you know the answer. "
           "If context is provided, use only that context. Otherwise reply exactly: I don't know. "
           "Never invent papers, settings, numbers or citations.")

def ask(q, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": q}]
    try:
        r = client.chat.completions.create(model=MODEL, messages=msgs, max_tokens=300)
    except openai.RateLimitError:
        sys.exit("429: free-tier rate limit hit. Wait a minute and rerun, or try a Flash-Lite model.")
    except openai.NotFoundError:
        sys.exit(f"404: model '{MODEL}' not found. Run check_setup.py with LLM_MODEL removed to list valid IDs.")
    except openai.APIStatusError as e:
        sys.exit(f"HTTP {e.status_code}: {e.message}")
    return (r.choices[0].message.content or "").strip()

for q in QUESTIONS:
    print("\nQ:", q.splitlines()[-1][:90])
    print("  plain  :", ask(q)[:220].replace("\n", " "))
    print("  guarded:", ask(q, GUARDED)[:220].replace("\n", " "))
