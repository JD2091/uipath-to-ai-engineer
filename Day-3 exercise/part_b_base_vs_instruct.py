from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
question = "What is the capital of France?"

for name in (BASE, INSTRUCT):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    cfg = model.config
    print(f"\n=== {name}")
    print("context window (max_position_embeddings):", cfg.max_position_embeddings)
    print("RoPE settings:", getattr(cfg, "rope_theta", None) or getattr(cfg, "rope_parameters", None))

    if name == INSTRUCT:   # the instruct model was tuned on this exact chat template
        text = tok.apply_chat_template([{"role": "user", "content": question}],
                                       tokenize=False, add_generation_prompt=True)
    else:                  # a base model only ever saw raw text
        text = question
    print("exact text the model sees:", repr(text))

    inputs = tok(text, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False)   # greedy
    print("->", tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# Then try giving the BASE model the chat-template text and the INSTRUCT model the raw question.
