"""Fills the Open Care Buddy prompt and reads the model's answer (OPENCARE-102)."""

import json
import re
from datetime import datetime
from typing import Any

from careplan_ai.buddy import prompt_files
from careplan_ai.buddy.schemas import BuddyChatRequest, BuddySource, BuddyTurn
from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import ConfigurationError


def _json(value: Any) -> str:
    # Compact, like the prototype's JSON.stringify, and UK characters kept as written.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _today(now: str) -> str:
    """'24/11/2026 10:15' -> 'Tuesday 24/11/2026' (the prototype's "Today is ..." form)."""
    day = datetime.strptime(now, "%d/%m/%Y %H:%M")
    return f"{day:%A} {day:%d/%m/%Y}"


def build_prompt(request: BuddyChatRequest) -> str:
    """The chat prompt file with every ${...} replaced from the request and the guide file.

    Filled in one pass, so text inside the data that happens to look like a placeholder is
    never replaced (record text is data, not part of the prompt).
    """
    ctx = request.context
    my_development = dict(ctx.my_development)
    if ctx.guidance:
        # The prompt reads Best Practice Guidance from MY DEVELOPMENT, as the prototype did
        # ("best_practice_guidance = BEST PRACTICE GUIDANCE").
        my_development["best_practice_guidance"] = ctx.guidance
    values = {
        "user_name": request.user.first_name,
        "role": request.user.role,
        "home": request.user.home,
        "shift": request.user.shift,
        "today": _today(request.now),
        "now": request.now.split(" ")[1],
        "screen": request.screen,
        "resident_names": ", ".join(str(r.get("name")) for r in ctx.residents if r.get("name")),
        "traffic_lights": _json(ctx.traffic_lights),
        "my_development": _json(my_development),
        "residents": _json(ctx.residents),
        "discharged": _json(ctx.discharged),
        "tasks": _json(ctx.tasks),
        "home_documents": _json(ctx.home_documents),
        "open_care_guide": prompt_files.load(prompt_files.OPEN_CARE_GUIDE),
    }
    template = prompt_files.load(prompt_files.CHAT_PROMPT)
    unknown = set(prompt_files.PLACEHOLDER.findall(template)) - values.keys()
    if unknown:
        # A placeholder the request can't fill: refuse rather than send the model "${...}".
        raise ConfigurationError(f"Unknown placeholders in the Buddy prompt: {sorted(unknown)}")
    return prompt_files.PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def recent_history(history: list[BuddyTurn]) -> list[BuddyTurn]:
    """The last turns of this chat (CAREPLAN_AI_BUDDY_HISTORY_TURNS), starting with a question:
    an answer without its question is dropped."""
    turns = [t for t in history if t.text.strip()][-get_settings().buddy_history_turns :]
    while turns and turns[0].role != "user":
        turns.pop(0)
    return turns


_SOURCES_LINE = re.compile(r"^\s*\**Sources:?\**\s*(.*)$", re.IGNORECASE)
_NEXT_LINE = re.compile(r"^\s*\**Next:?\**", re.IGNORECASE)
_SOURCE_ITEM = re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE)


def _source(text: str) -> BuddySource | None:
    parts = [p.strip() for p in text.split("·")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return BuddySource(area=parts[0], item=parts[1], date=" · ".join(parts[2:]))


def split_answer(text: str) -> tuple[str, list[BuddySource]]:
    """The answer without its "Sources:" line, and those sources as {area, item, date}.

    Any "Next:" line is removed too: Open Care Buddy shows no follow-up questions (rule 17).
    At most CAREPLAN_AI_BUDDY_MAX_SOURCES sources are kept (the prompt's rule 3 asks for 5).
    """
    max_sources = get_settings().buddy_max_sources
    lines = text.replace("\r\n", "\n").split("\n")
    kept: list[str] = []
    sources: list[BuddySource] = []
    seen: set[tuple[str, str, str]] = set()
    for line in lines:
        if _NEXT_LINE.match(line):
            continue
        match = _SOURCES_LINE.match(line)
        if match:
            for item in _SOURCE_ITEM.findall(match.group(1)):
                source = _source(item)
                if source is None:
                    continue
                key = (source.area, source.item, source.date)
                if key not in seen and len(sources) < max_sources:
                    seen.add(key)
                    sources.append(source)
            continue
        kept.append(line)
    return "\n".join(kept).strip(), sources
