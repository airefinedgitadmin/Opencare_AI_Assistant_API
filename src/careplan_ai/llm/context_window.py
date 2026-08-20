"""Keeps a generation request's prompt within the model's context window.

The Java backend deliberately does NOT cap how much history it fetches per
resident - previous care plans, review notes, and earlier completions of the
same assessment are all sent in full, so this is the only place prompt size
is controlled. A resident with an unusually deep history, or simply long
free-text entries, could still assemble a prompt too large for the model's
context window; rather than let that surface as an opaque LLM failure, this
trims the oldest entries first (all three lists already arrive ordered
most-recent-first) until the estimated prompt fits the configured limit,
logging exactly what was dropped so a heavy-history resident degrades
gracefully and visibly instead of failing.

This is deliberately NOT retrieval/relevance-based selection (that would be a
RAG-style approach - picking the most SEMANTICALLY RELEVANT items rather than
just the most recent). A resident's own history is small enough to reason
about directly; recency-first trimming is simpler and sufficient unless real
usage shows relevant-but-old items getting dropped.
"""

from dataclasses import dataclass, field

import tiktoken

from careplan_ai.api.schemas.request import CarePlanGenerationRequest
from careplan_ai.core.logging import get_logger
from careplan_ai.llm.prompts import SYSTEM_PROMPT, build_generation_prompt

logger = get_logger(__name__)

# A fixed, widely-compatible encoding used purely to ESTIMATE size - it doesn't
# need to exactly match the deployed model's real tokenizer, just be a
# consistent, reasonably close approximation given the configured limit
# already has headroom built in (see Settings.max_context_tokens).
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Trim priority, least essential first: review notes are the most
# "supplementary commentary" signal; previous completions of the same
# assessment are valuable but each entry can be large (a whole set of
# answers); previous care plans - the most directly actionable "what did we
# do" signal - are trimmed last.
_TRIMMABLE_FIELDS = ("review_notes", "previous_assessment_completions", "previous_care_plans")


@dataclass
class ContextWindowFit:
    payload: CarePlanGenerationRequest
    prompt: str
    dropped_counts: dict[str, int] = field(default_factory=dict)


def _count_tokens(*texts: str) -> int:
    return sum(len(_ENCODING.encode(text)) for text in texts)


def fit_to_context_window(
    payload: CarePlanGenerationRequest, max_context_tokens: int
) -> ContextWindowFit:
    """Returns a payload/prompt guaranteed to fit within `max_context_tokens`,
    trimming the oldest entries first (see _TRIMMABLE_FIELDS priority) if
    needed."""
    trimmed = payload.model_copy(deep=True)
    dropped_counts = {name: 0 for name in _TRIMMABLE_FIELDS}

    def current_size() -> int:
        return _count_tokens(SYSTEM_PROMPT, build_generation_prompt(trimmed))

    def has_trimmable_content() -> bool:
        return any(getattr(trimmed, name) for name in _TRIMMABLE_FIELDS)

    while current_size() > max_context_tokens and has_trimmable_content():
        for name in _TRIMMABLE_FIELDS:
            items = getattr(trimmed, name)
            if items:
                setattr(trimmed, name, items[:-1])
                dropped_counts[name] += 1
                break

    if any(dropped_counts.values()):
        logger.warning(
            "prompt_trimmed_for_context_window",
            assessment_template_user_id=payload.assessment_template_user_id,
            max_context_tokens=max_context_tokens,
            **{f"{name}_dropped": count for name, count in dropped_counts.items()},
        )

    return ContextWindowFit(
        payload=trimmed,
        prompt=build_generation_prompt(trimmed),
        dropped_counts=dropped_counts,
    )
