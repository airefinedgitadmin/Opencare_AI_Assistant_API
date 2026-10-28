"""The care plan agent: a tool-using loop.

Each turn the model calls one or more tools:

    SearchRecords(query)  keyword search over ALL this resident's records
                          (search.py) - only this resident, never another's
    GetRecords(ids)       read records by id
    SubmitPlan(plan)      hand in the whole plan; code checks it at once
                          (validation.py) and answers "Accepted." or "Not
                          accepted. Fix these: ..." with the exact problems

The last turn is forced to SubmitPlan. A plan may be sent back at most
agent_max_revisions times (code checks and verifier together); the best plan
seen (fewest problems) is the one kept. Quotes always come from the records
the code holds - the model only ever sees and cites them by id.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from careplan_ai.agents.base.llm import ModelRole, has_fallback_model, model_for_role
from careplan_ai.agents.care_plan import llm
from careplan_ai.agents.care_plan.prompts import format_grouped
from careplan_ai.agents.care_plan.schemas import DraftPlanOut, EvidenceItem
from careplan_ai.agents.care_plan.search import RecordIndex
from careplan_ai.agents.care_plan.validation import ValidationResult
from careplan_ai.core.exceptions import LlmGenerationError
from careplan_ai.core.logging import get_logger

logger = get_logger(__name__)

SUBMIT = "SubmitPlan"
_MAX_PROBLEMS_SHOWN = 40
_MAX_IDS_PER_READ = 20


@dataclass
class AgentRun:
    messages: list[BaseMessage]
    # Every record the agent has been shown (pack + search / read results).
    seen: dict[str, EvidenceItem]
    best_draft: DraftPlanOut | None = None
    best: ValidationResult | None = None
    best_role: ModelRole = "primary"
    best_model_id: str = ""
    revisions: int = 0
    turns: int = 0
    searches: int = 0
    queries: list[str] = field(default_factory=list)


Validate = Callable[[list[EvidenceItem], DraftPlanOut], ValidationResult]


def seen_in_order(run: AgentRun, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    return [e for e in evidence if e.id in run.seen]


async def run_turns(
    run: AgentRun,
    *,
    evidence: list[EvidenceItem],
    index: RecordIndex,
    validate: Validate,
    max_turns: int,
    max_revisions: int,
    search_k: int,
) -> None:
    """Runs the agent until a plan is accepted or the turns run out."""
    order = {e.id: i for i, e in enumerate(evidence)}
    by_id = {e.id: e for e in evidence}

    for turn in range(max_turns):
        last = turn == max_turns - 1
        # A stronger model, if configured, takes over once a plan was sent back.
        role: ModelRole = "fallback" if run.revisions and has_fallback_model() else "primary"
        ai = await llm.agent_step(role, run.messages, force=SUBMIT if last else None)
        run.turns += 1
        run.messages.append(ai)
        accepted = False

        for call in ai.tool_calls:
            name, args = call["name"], call.get("args") or {}
            if name == "SearchRecords":
                query = str(args.get("query", ""))
                run.searches += 1
                run.queries.append(query)
                hits = [item for item, _ in index.search(query, k=search_k, exclude=set(run.seen))]
                hits.sort(key=lambda e: order[e.id])
                run.seen.update({e.id: e for e in hits})
                content = format_grouped(hits) if hits else "No new matching records."
            elif name == "GetRecords":
                ids = [str(i).strip() for i in (args.get("ids") or [])][:_MAX_IDS_PER_READ]
                found = [by_id[i] for i in ids if i in by_id]
                run.seen.update({e.id: e for e in found})
                content = format_grouped(found) if found else "No records with those ids."
            elif name == SUBMIT:
                content, accepted = _submit(run, args, role, validate, evidence,
                                            last=last, max_revisions=max_revisions)
            else:
                content = f"There is no tool called {name}."
            run.messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

        if accepted:
            return
    logger.info("care_plan_agent_out_of_turns", turns=run.turns)


def _submit(
    run: AgentRun,
    args: dict,  # type: ignore[type-arg]
    role: ModelRole,
    validate: Validate,
    evidence: list[EvidenceItem],
    *,
    last: bool,
    max_revisions: int,
) -> tuple[str, bool]:
    try:
        draft = DraftPlanOut.model_validate(args)
    except ValidationError as exc:
        return f"Not accepted - the plan doesn't match the required format: {exc}", False
    result = validate(seen_in_order(run, evidence), draft)
    logger.info(
        "care_plan_draft_checked",
        turn=run.turns,
        cards=len(result.cards),
        problems=len(result.problems),
        backfilled=result.backfilled_count,
        unsupported_details=result.unsupported_count,
        copied_from_examples=result.copied_count,
        possibly_missing=result.missing_detail_count,
    )
    # A revision replaces the plan unless it made things worse.
    if run.best is None or result.problem_score <= run.best.problem_score:
        run.best, run.best_draft, run.best_role = result, draft, role
        run.best_model_id = model_for_role(role)[1]
    if result.problems and run.revisions < max_revisions and not last:
        run.revisions += 1
        problems = result.problems[:_MAX_PROBLEMS_SHOWN]
        return (
            "Not accepted. Fix every one of these, change nothing else, and call "
            "SubmitPlan again with the whole corrected plan:\n"
            + "\n".join(f"- {p}" for p in problems)
        ), False
    return "Accepted.", True


def send_back(run: AgentRun, feedback: list[str]) -> None:
    """The verifier's objections, as the next message in the same conversation."""
    run.revisions += 1
    run.messages.append(
        HumanMessage(
            content="VERIFIER - these statements are not supported by the quotes they cite:\n"
            + "\n".join(f"- {f}" for f in feedback)
            + "\nFix them (reword to say only what the quotes say, or cite the records "
            "that do) and call SubmitPlan with the whole corrected plan."
        )
    )


def require_plan(run: AgentRun) -> tuple[DraftPlanOut, ValidationResult]:
    if run.best_draft is None or run.best is None:
        raise LlmGenerationError("The care plan agent did not hand in a valid plan")
    return run.best_draft, run.best
