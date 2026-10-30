# UiPath → AI Engineer

A public learning log: one RPA / UiPath developer working through the skills of an AI engineer, from "what is a token" to agents, MCP, evals and production LLM apps.

Every lesson ends in code you can run. Every concept gets a **UiPath bridge** (how it maps to something automation people already know) and, just as important, a note on **where that analogy breaks**.

## Who this is for

- RPA / UiPath developers who want to move into AI engineering and already think in queues, jobs, assets and workflows.
- Anyone who learns better by running small, honest experiments than by watching long courses.

## How it works

- **Nightly:** about 20 minutes of reading, then 30 to 45 minutes of hands-on Python. Each night's exercise lands in its own `Day-N exercise/` folder with a short README.
- **Sundays:** a review, a weekend mini-project, and a written article on what I learned and what went wrong.
- **Free to follow:** the exercises use the free tier of Google AI Studio (Gemini) through its OpenAI-compatible endpoint, plus small local libraries. Any other OpenAI-compatible provider works by changing two environment variables.

## Follow along

```bash
git clone <this repo>
cd <this repo>
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then edit .env and add your own key
python check_setup.py    # lists models your key can use; set LLM_MODEL in .env; run again
```

`check_setup.py` prints "Setup OK." when your key, model and network are working. Your `.env` is git-ignored, so keep your key there and never commit it.

## What's here

| Path | What it is |
| --- | --- |
| [`ROADMAP.md`](ROADMAP.md) | The full curriculum (12 modules, 106 topics) as a checklist. Ticked items are done. |
| [`RPA-AI-ROSETTA.md`](RPA-AI-ROSETTA.md) | A growing map from UiPath concepts to AI-engineering concepts, with the limits of each analogy. |
| [`lessons/`](lessons) | One file per lesson: concepts, gotchas, interview angles, the hands-on exercise and recall answers. |
| `Day-N exercise/` | The code for each night, with a README: goal, how to run, what to notice. |
| [`templates/`](templates) | The templates I use for each day's README and each weekly article. |
| `check_setup.py` | One-time check that your API key and model work. |

## Progress

Module 1 (LLM foundations) is in progress. See [`ROADMAP.md`](ROADMAP.md) for the checklist.

## About

I'm Jeet Doshi. I have 11+ years across intelligent automation, enterprise application development and cloud technologies, and I help enterprises move from traditional RPA toward agentic automation using UiPath Agents, Maestro and AI-powered orchestration. Day to day that means platform onboarding and adoption, automation strategy and architecture, agentic workflow design, and technical enablement for customers.

I started as a .NET developer and then an RPA developer, building enterprise applications and automation solutions from scratch on Microsoft technologies and cloud-native architectures. This repo is where I'm learning the AI engineering layer underneath agentic automation: models, retrieval, tool use, evaluation and production practice.

**Background:** UiPath platform, Maestro and AI agents; agentic automation strategy; RPA solution design; Azure and Azure DevOps; Agile and DevOps; SOLID design and scalable architecture; technical consulting and training.

**What I write about:** AI and agentic workflows, enterprise agent orchestration, prompt engineering, developer tooling, intelligent workflows and enterprise AI adoption.

- LinkedIn: https://www.linkedin.com/in/jeet-doshi-28408071/
- Articles (Hashnode): https://hashnode.com/@jdoshi

Questions, corrections and "this analogy is wrong" issues are welcome. If an explanation here is wrong, I'd rather fix it publicly.

## License

[MIT](LICENSE)
