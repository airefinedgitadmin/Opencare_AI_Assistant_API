"""The care plan workflow: two agents, with code around them.

    pack -> retrieve -> agent --> verify --(not supported, revisions left)--> agent
                                    |
                                    v
                                  apply -> done

* pack     (code, packs.py) every must-read record, plus each need's best
           keyword matches across ALL the resident's records, to a small
           budget. Evidence is passed in - never fetched or rewritten by a model.
* retrieve (code + Pinecone) this home's style examples for this pathway
           (OPENCARE-51). Optional: none = the neutral house style.
* agent    the CARE PLAN AGENT (agent.py): a tool loop - searches the
           resident's records when something could change the plan, then
           submits; code checks each submission and sends back the exact
           problems until it is clean or the revision limit is reached.
* verify   the VERIFIER AGENT (verify.py): every AI statement against only the
           quotes it cites. Objections go back to the same agent conversation.
* apply    (code) statements still unsupported are dropped (their sources come
           back word for word), overstated ones are flagged, each statement
           records its verdict.

The order is fixed: the models never choose what happens next.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from careplan_ai.agents.care_plan.agent import (
    AgentRun,
    require_plan,
    run_turns,
    seen_in_order,
    send_back,
)
from careplan_ai.agents.care_plan.packs import Pack, build_pack
from careplan_ai.agents.care_plan.prompts import build_agent_messages
from careplan_ai.agents.care_plan.schemas import (
    DraftPlanOut,
    EvidenceItem,
    Pattern,
    SourceType,
    VerdictOut,
)
from careplan_ai.agents.care_plan.validation import ValidationResult, validate_draft
from careplan_ai.agents.care_plan.verify import (
    by_draft_index,
    feedback_from,
    mark_cards,
    rejected_draft,
    run_verifier,
)
from careplan_ai.core.config import get_settings
from careplan_ai.core.logging import get_logger
from careplan_ai.library.models import StyleExample
from careplan_ai.library.service import find_examples

logger = get_logger(__name__)

# Turns the agent gets to answer the verifier (search if needed, then submit).
_VERIFIER_FIX_TURNS = 3


class DraftState(TypedDict, total=False):
    resident_name: str | None
    pathway: str | None
    organisation_id: int | None
    evidence: list[EvidenceItem]
    pack: Pack
    examples: list[StyleExample]
    run: AgentRun
    feedback: list[str]
    verdicts: dict[str, VerdictOut] | None
    final: ValidationResult
    rejected: int


def _pack(state: DraftState) -> DraftState:
    settings = get_settings()
    pack = build_pack(
        state["evidence"],
        pathway=state.get("pathway"),
        max_must_read_tokens=settings.agent_max_draft_evidence_tokens,
        pack_tokens=settings.agent_pack_tokens,
        matches_per_need=settings.agent_pack_matches_per_need,
    )
    logger.info("care_plan_pack_built", records=len(state["evidence"]), shown=len(pack.shown))
    return {"pack": pack}


def _retrieval_query(evidence: list[EvidenceItem]) -> str:
    lines = [
        f"{e.label}: {e.text}" if e.source_type == SourceType.ASSESSMENT else e.text
        for e in evidence
        if e.source_type == SourceType.ASSESSMENT or (e.is_note and e.linked_to_pathway)
    ]
    return "\n".join(lines)


async def _retrieve(state: DraftState) -> DraftState:
    name = state.get("resident_name")
    examples = await find_examples(
        state.get("organisation_id"),
        state.get("pathway"),
        _retrieval_query(state["pack"].shown),
        resident_names=[name] if name else [],
    )
    return {"examples": examples}


def _validator(state: DraftState):  # type: ignore[no-untyped-def]
    def validate(seen: list[EvidenceItem], draft: DraftPlanOut) -> ValidationResult:
        return validate_draft(
            seen,
            draft,
            resident_name=state.get("resident_name"),
            pathway=state.get("pathway"),
            examples=[e.text for e in state.get("examples", [])],
            resident_sources=[e.text for e in state["evidence"]],
            with_outcomes=False,
        )

    return validate


async def _agent(state: DraftState) -> DraftState:
    settings = get_settings()
    run = state.get("run")
    if run is None:
        pack = state["pack"]
        run = AgentRun(
            messages=build_agent_messages(
                resident_name=state.get("resident_name"),
                pathway=state.get("pathway"),
                evidence=pack.shown,
                examples=state.get("examples", []),
            ),
            seen={e.id: e for e in pack.shown},
        )
        max_turns = settings.agent_max_turns
    else:
        send_back(run, state["feedback"])
        max_turns = _VERIFIER_FIX_TURNS
    await run_turns(
        run,
        evidence=state["evidence"],
        index=state["pack"].index,
        validate=_validator(state),
        max_turns=max_turns,
        max_revisions=settings.agent_max_revisions,
        search_k=settings.agent_search_results,
    )
    require_plan(run)
    logger.info("care_plan_agent_done", turns=run.turns, searches=run.searches,
                revisions=run.revisions, seen=len(run.seen))
    return {"run": run}


async def _verify(state: DraftState) -> DraftState:
    if not get_settings().agent_verify:
        return {"verdicts": None, "feedback": []}
    run = state["run"]
    _, best = require_plan(run)
    verdicts = await run_verifier(
        resident_name=state.get("resident_name"),
        result=best,
        evidence=seen_in_order(run, state["evidence"]),
    )
    feedback = feedback_from(verdicts, best) if verdicts else []
    return {"verdicts": verdicts, "feedback": feedback}


def _after_verify(state: DraftState) -> str:
    run = state["run"]
    if state["feedback"] and run.revisions < get_settings().agent_max_revisions:
        return "agent"
    return "apply"


def _apply(state: DraftState) -> DraftState:
    run = state["run"]
    draft, best = require_plan(run)
    verdicts = state.get("verdicts")
    if verdicts is None:
        mark_cards(best, None)
        return {"final": best, "rejected": 0}
    kept, rejected, index_map = rejected_draft(draft, best, verdicts)
    keyed = by_draft_index(best, verdicts, index_map)
    # Re-checking the plan without the rejected statements brings back, word
    # for word, any required source only they covered.
    final = _validator(state)(seen_in_order(run, state["evidence"]), kept) if rejected else best
    mark_cards(final, keyed)
    if rejected:
        logger.info("care_plan_statements_rejected", rejected=rejected)
    return {"final": final, "rejected": rejected}


@lru_cache
def build_draft_graph() -> CompiledStateGraph:
    graph = StateGraph(DraftState)
    graph.add_node("pack", _pack)
    graph.add_node("retrieve", _retrieve)
    graph.add_node("agent", _agent)
    graph.add_node("verify", _verify)
    graph.add_node("apply", _apply)
    graph.set_entry_point("pack")
    graph.add_edge("pack", "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_edge("agent", "verify")
    graph.add_conditional_edges("verify", _after_verify, {"agent": "agent", "apply": "apply"})
    graph.add_edge("apply", END)
    return graph.compile()


@dataclass
class DraftOutcome:
    result: ValidationResult
    model_id: str
    fallback_used: bool
    omitted: list[str] = field(default_factory=list)
    examples: list[StyleExample] = field(default_factory=list)
    patterns: list[Pattern] = field(default_factory=list)
    revisions: int = 0
    rejected: int = 0
    verifier_ran: bool = False
    searches: list[str] = field(default_factory=list)


async def run_draft(
    *,
    resident_name: str | None,
    pathway: str | None,
    organisation_id: int | None,
    evidence: list[EvidenceItem],
) -> DraftOutcome:
    final = await build_draft_graph().ainvoke(
        {
            "resident_name": resident_name,
            "pathway": pathway,
            "organisation_id": organisation_id,
            "evidence": evidence,
        }
    )
    run: AgentRun = final["run"]
    _, best = require_plan(run)
    return DraftOutcome(
        result=final["final"],
        model_id=run.best_model_id,
        fallback_used=run.best_role == "fallback",
        # Records the agent never saw - neither in its pack nor in a search.
        omitted=[e.id for e in evidence if e.id not in run.seen],
        examples=final.get("examples", []),
        patterns=best.patterns,
        revisions=run.revisions,
        rejected=final.get("rejected", 0),
        verifier_ran=final.get("verdicts") is not None,
        searches=run.queries,
    )
