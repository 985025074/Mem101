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


# how to wire this to you Agent 


# TODO

locomo 
