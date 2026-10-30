import os, tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads .env (GEMINI_API_KEY, LLM_MODEL)
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
)
MODEL = os.environ["LLM_MODEL"]   # the ID you put in .env
text = "The quick brown fox jumps over the lazy dog. " * 20
r = client.chat.completions.create(model=MODEL, max_tokens=1, messages=[{"role": "user", "content": text}])
print("tiktoken:", len(tiktoken.get_encoding("cl100k_base").encode(text)), "| Gemini:", r.usage.prompt_tokens)