# mem101:Simple,Small,Extensible Memory system.

This is a agent memory system project,spefically designed for all agents,like Pi,codex.We use Web server + Skills to make this possible.

And this project is highly extensible. 
You can replace our key components to better one easily.

# Memory Extraction.

We use LLM to extract facts.This is done by a specially desigend prompt.And you can add relation extraction in the future.The original events are also stored for reference in case.

TODO: is it worth adding recent messages to do this?

# Backend

we use sqlite as our backend. We also store each memory's vector to do search.
After normal distance comparision,an LLM conducted comparision is also added to improve accuracy.

# Retrive


# Quick start/how to wire this to you Agent 

The canonical agent skill is stored in `skills/memkernel-memory`. The setup
script installs it globally in `~/.agents/skills`, where Agent Skills-compatible
agents such as Codex and Pi can discover it.

Start the configured API application:

```bash
uv run python scripts/initialize_database.py
uv run fastapi dev src/memkernel/api.py
```

The initialization command creates or migrates `memkernel.db` and rebuilds
embeddings for all existing memories. Use `--database PATH` for another file.

Preview where the skill would be installed:

```bash
python3 scripts/setup_agent_skill.py --dry-run
```

Install it globally:

```bash
python3 scripts/setup_agent_skill.py
```

Set `MEMKERNEL_URL` when the server does not use the default
`http://127.0.0.1:8000`, then restart or reload the agent.

## Minimal LoCoMo benchmark

The runner follows the ingest, retrieve, answer, and judge pipeline used by
[mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks).
It downloads LoCoMo automatically and writes a unified JSON result.

Run a small smoke test (one conversation, five questions):

```bash
uv run python -m benchmarks.locomo
```

By default, the runner combines eight turns from the same session into one
extraction request. Override this with `--chunk-size`; use `--chunk-size 1`
to reproduce one-turn-at-a-time ingestion.

Run retrieval only, without the answer and judge calls:

```bash
uv run python -m benchmarks.locomo --predict-only
```

Run all ten conversations and all supported questions:

```bash
uv run python -m benchmarks.locomo --all-conversations --all-questions
```

The runner uses the project's existing `API_KEY`, `API_BASE`, and
`EMBEDDING_BASE` environment settings.

## performance 

Due to time limited,we only tested on Conversation 1.You can see full result at ./benchmarks/result.json.
This is our performance :

```json
      "questions_with_evidence": 30,
      "evidence_hit_rate": 0.8333,
      "mean_evidence_recall": 0.7644
```
Hit Rate: If a question's evidence is hit,then it counters.
hit question / all question
evidence_recall: Measure how much evidence do we recall.
If one question has 3 evidence,we only recall 2,then it is 2/3






# TODO

locomo 
