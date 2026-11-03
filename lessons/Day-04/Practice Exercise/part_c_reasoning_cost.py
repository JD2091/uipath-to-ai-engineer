import os, time
from dotenv import load_dotenv
from openai import OpenAI, BadRequestError

load_dotenv()
client = OpenAI(api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
                base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
MODEL = os.getenv("LLM_MODEL")

TASKS = [
    ("How many times does the letter r appear in 'strawberry raspberry'? "
     "End with a line 'ANSWER: <number>'.", "6"),
    ("An Orchestrator queue gets 7 items at minute 0, then 3 more items at minutes 5, 10, 15, 20, 25 and 30. "
     "Two robots start at minute 0; each takes exactly 4 minutes per item and picks the next item "
     "immediately when it finishes, if one is waiting. At minute 30, after that minute's arrivals and "
     "pickups, how many items are still waiting (not started)? End with a line 'ANSWER: <number>'.", "9"),
]

def reasoning_tokens(usage):
    details = getattr(usage, "completion_tokens_details", None)
    rt = getattr(details, "reasoning_tokens", None) if details else None
    if rt is None and usage.total_tokens:            # fallback: thinking often shows up as the gap
        rt = usage.total_tokens - usage.prompt_tokens - usage.completion_tokens
    return rt

print(f"{'effort':7} {'task':5} {'ok':3} {'secs':>5} {'answer_tok':>10} {'reason_tok':>10}")
for effort in ("low", "high"):
    for i, (prompt, expected) in enumerate(TASKS, 1):
        for run in range(2):
            t0 = time.perf_counter()
            try:
                r = client.chat.completions.create(model=MODEL, reasoning_effort=effort,
                                                   messages=[{"role": "user", "content": prompt}])
            except BadRequestError as e:
                print(f"{effort}: rejected ({e.message[:80]}) - is LLM_MODEL a thinking model?")
                break
            secs = time.perf_counter() - t0
            text = r.choices[0].message.content or ""
            got = text.rsplit("ANSWER:", 1)[-1].strip().split()[0] if "ANSWER:" in text else "?"
            print(f"{effort:7} {i:<5} {'Y' if got == expected else 'N':3} {secs:5.1f} "
                  f"{r.usage.completion_tokens:>10} {reasoning_tokens(r.usage)!s:>10}")
