"""The agents' model calls - care plan, verifier, readers - and the
staff-requested redraft.

Kept in one small module so tests can replace exactly these and exercise
everything else (agent loop, graph, validation, memory, API) for real.
"""

from langchain_core.messages import AIMessage, BaseMessage

from careplan_ai.agents.base.llm import ModelRole, invoke_structured, invoke_with_tools
from careplan_ai.agents.care_plan.prompts import (
    build_outcome_messages,
    build_read_messages,
    build_redraft_messages,
    build_verify_messages,
)
from careplan_ai.agents.care_plan.schemas import (
    AGENT_TOOLS,
    EvidenceItem,
    OutcomesOut,
    ReadOut,
    RedraftOut,
    VerifyOut,
)
from careplan_ai.library.models import StyleExample


async def agent_step(
    role: ModelRole, messages: list[BaseMessage], *, force: str | None = None
) -> AIMessage:
    """One turn of the care plan agent: it calls SearchRecords, GetRecords or
    SubmitPlan (`force` = must call that one)."""
    return await invoke_with_tools(role, messages, AGENT_TOOLS, force=force)


async def write_outcomes(
    *,
    resident_name: str | None,
    actions: list[tuple[str, str, list[EvidenceItem]]],
    examples: list[StyleExample] | None = None,
    feedback: dict[str, str] | None = None,
) -> OutcomesOut:
    """Care plan agent's outcome step: one outcome per added action. `feedback`
    = ref -> the rejected outcome and the verifier's reason, for a rewrite."""
    messages = build_outcome_messages(
        resident_name=resident_name, actions=actions, examples=examples, feedback=feedback
    )
    return await invoke_structured("primary", OutcomesOut, messages)


async def verify_statements(
    *, resident_name: str | None, statements: list[tuple[str, str, list[EvidenceItem]]]
) -> VerifyOut:
    """Verifier agent: each statement judged against only the quotes it cites."""
    messages = build_verify_messages(resident_name=resident_name, statements=statements)
    return await invoke_structured("primary", VerifyOut, messages)


async def read_records(
    *,
    resident_name: str | None,
    pathway: str | None,
    context: list[EvidenceItem],
    records: list[EvidenceItem],
) -> ReadOut:
    """A reader agent: classifies one chunk of records for this pathway."""
    messages = build_read_messages(
        resident_name=resident_name, pathway=pathway, context=context, records=records
    )
    return await invoke_structured("reader", ReadOut, messages)


async def generate_redraft(
    *,
    resident_name: str | None,
    target: str,
    current_text: str,
    earlier_wordings: list[str],
    evidence: list[EvidenceItem],
    action_text: str | None,
    examples: list[StyleExample] | None = None,
) -> str:
    messages = build_redraft_messages(
        resident_name=resident_name,
        target=target,
        current_text=current_text,
        earlier_wordings=earlier_wordings,
        evidence=evidence,
        action_text=action_text,
        examples=examples,
    )
    result = await invoke_structured("redraft", RedraftOut, messages)
    return result.text.strip()
