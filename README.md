# mem101:Simple,Small,Extensible Memory system.

This is a agent memory system project,spefically designed for all agents,like Pi,codex.We use Web server + Skills to make this possible.

And this project is highly extensible. 
You can replace our key components to better one easily.

# Memory Extraction.

We use LLM to extract facts.This is done by a specially desigend prompt.And you can add 关系抽取 in the future.The original events are also stored for reference in case.

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




# TODO

locomo 
