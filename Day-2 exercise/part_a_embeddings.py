import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
)
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")

sentences = [
    "The invoice is overdue.",
    "Payment for this invoice is late.",
    "The robot clicked the Submit button.",
    "The bot pressed submit on the form.",
    "I love this vendor.",
    "I do not love this vendor.",
]
r = client.embeddings.create(model=EMBED_MODEL, input=sentences)
E = np.array([d.embedding for d in r.data])
print("vectors:", E.shape)                                  # (6, dimensions)

E = E / np.linalg.norm(E, axis=1, keepdims=True)           # unit length -> dot product = cosine
S = E @ E.T
np.set_printoptions(precision=2, suppress=True)
print(S)
for i, s in enumerate(sentences):
    j = int(np.argsort(-S[i])[1])                          # nearest neighbour other than itself
    print(f"{s!r:42} -> {sentences[j]!r}  ({S[i, j]:.2f})")
