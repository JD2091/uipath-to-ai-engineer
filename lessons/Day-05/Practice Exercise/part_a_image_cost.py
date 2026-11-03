import math

def claude_image_tokens(w, h, max_edge=1568, max_tokens=1568):
    """Approximate Claude standard tier: downscale to fit, then 28x28 patches."""
    scale = min(1.0, max_edge / max(w, h))
    while True:
        tw, th = max(1, int(w * scale)), max(1, int(h * scale))
        tokens = math.ceil(tw / 28) * math.ceil(th / 28)
        if tokens <= max_tokens:
            return tokens, (tw, th)
        scale *= 0.97

def gemini_image_tokens(w, h):
    """Approximate Gemini rule: small image = 258 tokens, else 768x768 tiles at 258 each."""
    if w <= 384 and h <= 384:
        return 258
    return math.ceil(w / 768) * math.ceil(h / 768) * 258

for name, (w, h) in {"thumbnail": (300, 300), "screenshot": (1280, 720),
                     "A4 scan 200dpi": (1654, 2339), "4K photo": (3840, 2160)}.items():
    ct, size = claude_image_tokens(w, h)
    print(f"{name:15} {w}x{h:<5} Claude~{ct:>5} tokens (sent as {size[0]}x{size[1]})  "
          f"Gemini~{gemini_image_tokens(w, h):>5} tokens")
print("10-min call on Gemini audio:", 10 * 60 * 32, "tokens")
