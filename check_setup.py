import os
import sys

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env in the current folder

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
base_url = os.getenv("LLM_BASE_URL", GEMINI_URL)
model = os.getenv("LLM_MODEL")

if not api_key:
    sys.exit("No key found. Put GEMINI_API_KEY=... in the .env file (step 3).")

client = OpenAI(api_key=api_key, base_url=base_url)

try:
    if not model:
        print("LLM_MODEL is not set yet. Models available to your key:\n")
        for m in client.models.list():
            print("  ", m.id)
        print("\nPick a chat model, add LLM_MODEL=<id> to .env, then rerun.")
        print("IDs may print as 'models/xyz' -- use only the part after 'models/'.")
        sys.exit(0)

    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with one short sentence: what is a token?"}],
    )
    print("Reply:", r.choices[0].message.content)
    u = r.usage
    print(f"Tokens -> prompt: {u.prompt_tokens}, completion: {u.completion_tokens}, total: {u.total_tokens}")
    print("\nSetup OK.")

except openai.RateLimitError:
    sys.exit("429: free-tier rate limit hit. Wait a minute, or try a Flash-Lite model.")
except openai.NotFoundError:
    sys.exit(f"404: model '{model}' not found. Remove LLM_MODEL from .env to list valid IDs.")
except openai.APIStatusError as e:
    sys.exit(f"HTTP {e.status_code}: {e.message}\n(400/401/403 usually means a wrong, mistyped or deleted key.)")
except openai.APIConnectionError as e:
    sys.exit(f"Could not reach {base_url}: {e}. Check internet, VPN or proxy.")
