import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
)
MODEL = os.environ["LLM_MODEL"]
EXTRA = {"reasoning_effort": "low"}  # keeps thinking tokens small; remove if your model rejects it


def ask(messages):
    r = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=1024, **EXTRA)
    return r.choices[0].message.content, r.choices[0].finish_reason, r.usage


# 1) Two independent calls: nothing is shared between them
print(ask([{"role": "user", "content": "My name is JD and I build UiPath agents. Reply 'noted'."}])[0])
print(ask([{"role": "user", "content": "What is my name?"}])[0])

# 2) The same question with the history resent by us
history = [
    {"role": "system", "content": "You are a concise tutor. Answer in one sentence."},
    {"role": "user", "content": "My name is JD and I build UiPath agents. Reply 'noted'."},
    {"role": "assistant", "content": "Noted."},
    {"role": "user", "content": "What is my name, and what do I build?"},
]
text, finish, usage = ask(history)
print(text, "| finish:", finish, "| prompt:", usage.prompt_tokens, "completion:", usage.completion_tokens)
