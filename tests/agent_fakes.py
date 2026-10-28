"""Helpers to script the care plan agent's model turns in tests.

The real agent loop, tools, code checks and graph all run; only the model's
replies (which tool it calls, with what) are scripted.
"""

import itertools
import re
from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from careplan_ai.agents.care_plan.schemas import DraftPlanOut

_ids = itertools.count(1)
_ID_GROUP = re.compile(r"^\[([A-Z]\d+(?:, [A-Z]\d+)*)\]", re.MULTILINE)


def _call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}"}])


def submit(draft: DraftPlanOut) -> AIMessage:
    return _call("SubmitPlan", draft.model_dump())


def search(query: str) -> AIMessage:
    return _call("SearchRecords", {"query": query})


def shown_ids(messages: list[BaseMessage]) -> list[str]:
    """Record ids in the agent's opening EVIDENCE block, in order."""
    opening = next(m for m in messages if isinstance(m, HumanMessage))
    evidence = str(opening.content).split("STYLE EXAMPLES")[0]
    return [i for group in _ID_GROUP.findall(evidence) for i in group.split(", ")]


def style_texts(messages: list[BaseMessage]) -> list[str]:
    opening = str(next(m for m in messages if isinstance(m, HumanMessage)).content)
    if "STYLE EXAMPLES" not in opening:
        return []
    blocks = re.split(r"--- Example \d+ ---\n", opening.split("STYLE EXAMPLES", 1)[1])[1:]
    return [b.rstrip("\n") for b in blocks]


def feedback(messages: list[BaseMessage]) -> str:
    """The last thing the agent was told: a tool result or the verifier's note."""
    last = messages[-1]
    return str(last.content) if isinstance(last, ToolMessage | HumanMessage) else ""


Script = Callable[[str, list[BaseMessage]], AIMessage | DraftPlanOut]


def fake_agent(monkeypatch: pytest.MonkeyPatch, script: Script) -> list[dict[str, Any]]:
    """Every agent turn calls script(role, messages); a DraftPlanOut means
    "submit this plan". Returns a log of turns."""
    turns: list[dict[str, Any]] = []

    async def _step(role: str, messages: list[BaseMessage], *, force: str | None = None
                    ) -> AIMessage:
        turns.append({"role": role, "evidence": shown_ids(messages), "feedback": feedback(messages),
                      "force": force, "examples": style_texts(messages)})
        reply = script(role, messages)
        return submit(reply) if isinstance(reply, DraftPlanOut) else reply

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.agent_step", _step)
    return turns
