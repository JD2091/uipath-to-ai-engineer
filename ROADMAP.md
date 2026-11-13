# Roadmap

106 topics across 12 modules, taught in this order. Ticked items are done, with the date.

## Module 1: LLM Foundations
- [x] 1.1 What an LLM is: next-token prediction, base vs chat models — 2026-10-30
- [x] 1.2 Tokens and tokenization (BPE), why token counts matter — 2026-10-30
- [x] 1.3 Neural network essentials: weights, loss, gradient descent, training vs inference — 2026-10-30
- [x] 1.4 Embeddings intuition: meaning as vectors — 2026-10-30
- [x] 1.5 Transformer architecture overview — 2026-10-30
- [x] 1.6 Self-attention and multi-head attention — 2026-10-30
- [x] 1.7 Positional encoding and the context window — 2026-10-31
- [x] 1.8 Pretraining, instruction tuning, RLHF/RLAIF/Constitutional AI — 2026-10-31
- [x] 1.9 Sampling: temperature, top-p, top-k, determinism — 2026-10-31
- [x] 1.10 KV cache, latency, and why long context costs more — 2026-11-02
- [x] 1.11 Hallucinations: causes and mitigations — 2026-11-02
- [x] 1.12 Reasoning models and extended thinking — 2026-11-02
- [x] 1.13 Multimodality: vision, documents, audio — 2026-11-03
- [x] 1.14 Model landscape: closed vs open weights, sizes, choosing a model — 2026-11-03
- [x] 1.15 Classical ML refresher: supervised/unsupervised, overfitting, precision/recall/F1 — 2026-11-03

## Module 2: Working with LLM APIs
- [x] 2.1 Messages API anatomy: roles, system prompt, multi-turn state — 2026-11-04
- [x] 2.2 Pricing, token budgeting, rate limits — 2026-11-04
- [x] 2.3 Streaming — 2026-11-04
- [x] 2.4 Structured outputs and JSON schemas — 2026-11-05
- [x] 2.5 Errors, retries, backoff, timeouts — 2026-11-05
- [x] 2.6 Prompt caching — 2026-11-05
- [x] 2.7 Batch processing — 2026-11-06
- [x] 2.8 Async and concurrency in Python for LLM calls — 2026-11-06
- [x] 2.9 Cloud-hosted models (Bedrock/Vertex/Azure) and AI gateways — 2026-11-06

## Module 3: Prompt Engineering
- [x] 3.1 Clarity, specificity, context — 2026-11-07
- [x] 3.2 Few-shot examples — 2026-11-07
- [x] 3.3 Chain-of-thought — 2026-11-07
- [x] 3.4 Structuring prompts with XML tags — 2026-11-09
- [x] 3.5 System prompts and roles — 2026-11-09
- [x] 3.6 Prompt chaining and decomposition — 2026-11-09
- [x] 3.7 Prompt templates, versioning, testing — 2026-11-10
- [x] 3.8 Prompt injection basics — 2026-11-10

## Module 4: Embeddings & Vector Search
- [x] 4.1 Sentence/document embeddings and embedding models — 2026-11-10
- [x] 4.2 Similarity metrics — 2026-11-11
- [x] 4.3 Vector databases (pgvector, Chroma, Qdrant, Pinecone) — 2026-11-11
- [x] 4.4 ANN indexes (HNSW, IVF), recall vs speed — 2026-11-11
- [x] 4.5 Metadata filtering and multi-tenancy — 2026-11-12
- [x] 4.6 Choosing and benchmarking embedding models — 2026-11-12

## Module 5: RAG
- [x] 5.1 RAG architecture end to end — 2026-11-12
- [x] 5.2 Document loading and parsing (PDFs, tables, OCR) — 2026-11-13
- [x] 5.3 Chunking strategies — 2026-11-13
- [x] 5.4 Hybrid search (BM25 + vectors) — 2026-11-13
- [ ] 5.5 Reranking
- [ ] 5.6 Query rewriting, multi-query, HyDE
- [ ] 5.7 Contextual retrieval
- [ ] 5.8 Grounding and citations
- [ ] 5.9 Evaluating RAG
- [ ] 5.10 GraphRAG and agentic RAG
- [ ] 5.11 RAG vs long context vs fine-tuning

## Module 6: Tool Use & MCP
- [ ] 6.1 Tool calling mechanics
- [ ] 6.2 Designing tool schemas and descriptions
- [ ] 6.3 The tool-use loop
- [ ] 6.4 Parallel tool calls and tool errors
- [ ] 6.5 MCP architecture (host, client, server)
- [ ] 6.6 MCP primitives (tools, resources, prompts)
- [ ] 6.7 Building an MCP server in Python
- [ ] 6.8 MCP transports, OAuth, security

## Module 7: Agents
- [ ] 7.1 Workflows vs agents
- [ ] 7.2 Workflow patterns: chaining, routing, parallelization
- [ ] 7.3 Orchestrator-workers and evaluator-optimizer
- [ ] 7.4 ReAct and the agent loop
- [ ] 7.5 Planning and task decomposition
- [ ] 7.6 Memory: short-term, long-term, episodic
- [ ] 7.7 Context engineering
- [ ] 7.8 Multi-agent systems and subagents
- [ ] 7.9 A2A protocol and interoperability
- [ ] 7.10 Human-in-the-loop and approvals
- [ ] 7.11 Agent frameworks (LangGraph, Claude Agent SDK, OpenAI Agents SDK, CrewAI)
- [ ] 7.12 Computer-use and browser agents
- [ ] 7.13 Agent failure modes and debugging
- [ ] 7.14 Skills, AGENTS.md, packaging agent knowledge

## Module 8: Evaluation
- [ ] 8.1 Why evals; eval-driven development
- [ ] 8.2 Eval datasets and golden sets
- [ ] 8.3 Code-based graders
- [ ] 8.4 LLM-as-judge
- [ ] 8.5 Human evaluation
- [ ] 8.6 Agent evals: trajectories and tool-use correctness
- [ ] 8.7 Regression testing and evals in CI
- [ ] 8.8 Public benchmarks and their limits

## Module 9: Model Customization
- [ ] 9.1 When to fine-tune (and when not to)
- [ ] 9.2 Full fine-tuning vs PEFT
- [ ] 9.3 LoRA and QLoRA
- [ ] 9.4 Dataset preparation
- [ ] 9.5 Distillation
- [ ] 9.6 Quantization
- [ ] 9.7 Open models and local inference (Ollama, vLLM, Hugging Face)

## Module 10: LLMOps & Production
- [ ] 10.1 Reference architecture of a production LLM app
- [ ] 10.2 Observability and tracing
- [ ] 10.3 Latency optimization
- [ ] 10.4 Cost optimization and model routing
- [ ] 10.5 Semantic caching
- [ ] 10.6 AI gateways and fallbacks
- [ ] 10.7 Deployment and serving
- [ ] 10.8 Versioning prompts, models, datasets

## Module 11: Safety, Security & Governance
- [ ] 11.1 Prompt injection deep dive (direct and indirect)
- [ ] 11.2 Guardrails
- [ ] 11.3 PII and data privacy
- [ ] 11.4 Jailbreaks and red-teaming
- [ ] 11.5 Agent permissions and least privilege
- [ ] 11.6 Responsible AI and regulation (EU AI Act basics)

## Module 12: System Design & Interview Prep
- [ ] 12.1 AI system design interview framework
- [ ] 12.2 Design: customer-support assistant
- [ ] 12.3 Design: enterprise document Q&A
- [ ] 12.4 Design: coding agent
- [ ] 12.5 Design: enterprise automation agent (RPA + LLM)
- [ ] 12.6 Portfolio projects and how to present them
