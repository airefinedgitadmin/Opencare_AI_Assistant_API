"""Agents built on the shared LLM gateway, memory and validation layers.

Each agent lives in its own subpackage (e.g. `care_plan`); `base` holds what
they share. New use cases (RAG library, notes summaries, ...) are added as new
subpackages here rather than as new services.
"""
