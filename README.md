<p align="center">
  <img src="docs/assets/memkernel-icon.png" alt="MemKernel project icon" width="180">
</p>

<h1 align="center">MemKernel</h1>

<p align="center">
  <strong>A simple, compact, and extensible memory system for AI agents.</strong>
</p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white">
  <img alt="MemKernel 0.1.0" src="https://img.shields.io/badge/MemKernel-0.1.0-6D4AFF">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.141%2B-009688?logo=fastapi&logoColor=white">
  <img alt="SQLite" src="https://img.shields.io/badge/SQLite-Vector_Search-003B57?logo=sqlite&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white">
  <img alt="OpenAI-compatible" src="https://img.shields.io/badge/OpenAI-Compatible-412991?logo=openai&logoColor=white">
  <img alt="Agent Skills compatible" src="https://img.shields.io/badge/Agent_Skills-Compatible-1F6FEB">
</p>

MemKernel gives agents such as Codex and Pi a persistent memory layer. It runs
as a web service and ships with an Agent Skill, allowing compatible agents to
store and retrieve memories through a small HTTP API.

The system is intentionally modular: the extraction model, embedding model,
retrieval strategy, reconciliation logic, and storage backend can be replaced
independently.

## Paper

For a concise description of the architecture and preliminary LoCoMo results,
see [MemKernel: A Simple and Extensible Memory Service for Language Agents](paper/memkernel_acl_short_paper.pdf).

## How it works

1. **Ingest:** MemKernel receives a message, tool result, or document.
2. **Extract:** An LLM converts the source into a small set of durable,
   evidence-backed facts.
3. **Reconcile:** Vector search finds related memories, and an LLM classifies
   each new fact as equivalent, superseding, or distinct.
4. **Store:** Memories, embeddings, source events, and evidence links are
   persisted in SQLite.
5. **Recall:** Semantic search returns relevant current memories and,
   optionally, superseded history.

## Memory extraction

MemKernel uses a dedicated prompt to extract durable facts from each source
event. Every extracted fact must include an exact evidence quote from the
source, which keeps memories traceable to their origin.

Recent conversation context can be supplied to resolve references, pronouns,
confirmations, and corrections. Context is used only for disambiguation; a fact
must still be supported by the current source event.

The original source events are stored alongside the derived memories for
provenance and debugging. Common secrets and authentication tokens are
redacted before extraction and storage.

## Storage and retrieval

MemKernel uses SQLite as its default storage backend. Each memory is stored
with an embedding for semantic search. When a new fact resembles an existing
memory, an LLM performs a second comparison to decide whether the facts are:

- **Equivalent:** the existing memory remains current and gains another source.
- **Superseding:** the old memory is preserved as history and replaced by the
  new fact.
- **Distinct:** both memories remain current.

This design preserves memory history while keeping normal recall focused on
the latest known information.

## Quick start

### Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- An OpenAI-compatible chat-completions endpoint
- An OpenAI-compatible embeddings endpoint

Configure the following environment variables, for example in a local `.env`
file:

```bash
API_KEY=your-api-key
API_BASE=https://your-chat-endpoint/v1
EMBEDDING_BASE=http://127.0.0.1:11434/v1
```

Install the dependencies:

```bash
uv sync
```

Create or migrate the database and rebuild embeddings for any existing
memories:

```bash
uv run python scripts/initialize_database.py
```

Use `--database PATH` to initialize a different SQLite database. The default
embedding model is `nomic-embed-text:latest`; select another model with
`--embedding-model MODEL`.

Start the API server:

```bash
uv run fastapi dev src/memkernel/api.py
```

By default, the service is available at `http://127.0.0.1:8000`.

## API

The main endpoints are:

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/v1/memories` | Extract and store memories from a source event |
| `POST` | `/v1/recall` | Retrieve relevant current and historical memories |
| `GET` | `/v1/memories/{memory_id}/history` | Return a memory's supersession chain |
| `GET` | `/v1/memories/{memory_id}/sources` | Return the source events linked to a memory |
| `GET` | `/debug/memories` | Inspect memories and provenance in a debug view |

FastAPI also exposes interactive API documentation at `/docs` while the server
is running.

## Connect MemKernel to an agent

The canonical Agent Skill is stored in `skills/memkernel-memory`. The setup
script installs it globally in `~/.agents/skills`, where Agent
Skills-compatible clients such as Codex and Pi can discover it.

Preview the installation path:

```bash
python3 scripts/setup_agent_skill.py --dry-run
```

Install the skill:

```bash
python3 scripts/setup_agent_skill.py
```

If the service is not running at `http://127.0.0.1:8000`, set
`MEMKERNEL_URL` to the correct base URL. Restart or reload the agent after
installing the skill.

## LoCoMo benchmark

The benchmark runner follows the ingest, retrieve, answer, and judge pipeline
used by
[mem0ai/memory-benchmarks](https://github.com/mem0ai/memory-benchmarks). It
downloads LoCoMo automatically and writes the results to a unified JSON file.

Run a smoke test with one conversation and five questions:

```bash
uv run python -m benchmarks.locomo
```

By default, the runner combines eight turns from the same session into a single
extraction request. Override this behavior with `--chunk-size`; use
`--chunk-size 1` for one-turn-at-a-time ingestion.

Run retrieval only, without answer generation or judging:

```bash
uv run python -m benchmarks.locomo --predict-only
```

Run all ten conversations and all supported questions:

```bash
uv run python -m benchmarks.locomo --all-conversations --all-questions
```

The runner uses the project's existing `API_KEY`, `API_BASE`, and
`EMBEDDING_BASE` settings.

## Current benchmark results

Due to limited evaluation time, the current results cover Conversation 1 only.
The full output is available in [`benchmarks/result.json`](benchmarks/result.json).

| Metric | Result |
| --- | ---: |
| Questions with evidence | 30 |
| Evidence hit rate | 0.8333 |
| Mean evidence recall | 0.7644 |

**Evidence hit rate** is the proportion of questions for which at least one
expected piece of evidence was retrieved.

**Mean evidence recall** measures how much of the expected evidence was
retrieved per question. For example, retrieving two of three expected evidence
items gives that question a recall of `2/3`.

## TODO

- Run the full LoCoMo evaluation.
