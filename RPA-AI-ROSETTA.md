# RPA ↔ AI Rosetta Stone

A running map from UiPath / RPA concepts to AI-engineering concepts. One row is added whenever a lesson produces a link that holds up.

**Rule for this page:** an analogy is a teaching aid, not an equivalence. Every row says where the analogy breaks, because that is where the real learning is.

| AI concept | UiPath / RPA analogue | Where the analogy breaks | Day |
| --- | --- | --- | --- |
| Training vs inference | Publishing a package version vs running jobs on it. After publish, the logic is frozen; running it changes nothing about it. | A published workflow gives the same output for the same input. LLM inference samples from probabilities, so it need not (sampling is covered later in the roadmap). | 1 |
| Token count as the unit of cost and limits | Estimating transaction volume and run time before a run, to plan capacity. | Counts are model-specific: the same text gets a slightly different count under another tokenizer (Day 1, Part C: 201 vs 202), so recount against the model you will call. | 1 |

## Planned rows

These are links I expect to be useful. They stay here until the matching topic is covered and I have tested whether the analogy holds.

- Orchestrator queues and batch processing (Module 2)
- Orchestrator assets / credentials and secrets management (Module 6, Module 11)
- UiPath AI Trust Layer and guardrails (Module 11)
- Workflows vs agents, in terms of deterministic vs LLM-decided control flow (Module 7)

Disagree with a row? Open an issue.
