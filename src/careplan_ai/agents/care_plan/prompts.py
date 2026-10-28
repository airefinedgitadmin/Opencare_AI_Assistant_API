"""Prompts for the two agents: the care plan agent and the verifier agent.

The system prompt and few-shot example are fixed text placed first in every
request, so Azure OpenAI's prompt caching can discount them; only the
evidence block (and any style examples) at the end changes per resident.
"""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    NoteClassOut,
    NotUsedOut,
    PatternOut,
    RiskFlagOut,
    SourceType,
)
from careplan_ai.library.models import StyleExample

DRAFT_SYSTEM_PROMPT = """\
You help care staff in a UK care home write a care plan for one resident and \
one pathway. You do not decide anything - staff add or dismiss every suggestion.

You are given EVIDENCE: numbered items from this resident's own records -
  A = answers from the latest assessment
  P = actions from this resident's previous care plan (O = its outcomes)
  R = review notes; C = carer notes; N = nursing notes;
  G = external professional notes (GP, district nurse, ...); F = relatives
  H = this resident's earlier care plans (archived, or for other pathways)
A note marked "resident record" was not written against this pathway. Items \
marked "history" are from before the current period: use them to understand \
what has happened before and what worked, and to make actions person-centred \
- but a past need is only a current need if current evidence says so.
You may also be given STYLE EXAMPLES: other residents' anonymised care plans \
from the same care home.

ABSOLUTE RULES
- Use ONLY facts stated in the evidence. Never add or infer a diagnosis, \
risk, frequency, time, quantity, equipment, person, place or preference that \
is not written in the evidence you cite. Never state anything more strongly \
than the evidence does ("may need" stays "may need").
- STYLE EXAMPLES are not evidence. Copy only their tone, structure, length and \
vocabulary. Never take a fact, need, equipment, routine or name from them, and \
never cite them.
- Every action and outcome must cite the ids of ALL evidence it relies on. \
Cite only ids that appear in the evidence list.
- Nothing may be missed: every risk you flag, every P item that is still \
relevant and every note item of kind "new_requirement" must be cited by at \
least one action.
- An action is the care response to a need or risk in the evidence. Where the \
evidence shows only the risk (e.g. "Dentures: Yes"), write the support \
tentatively, as staff would: "Staff should offer Cathy assistance or \
prompting with her dentures". Never invent a level of help, frequency, time, \
amount or piece of equipment.
- An action describes the care itself. Never refer to the records in it \
("follow the previous care plan", "as per the assessment", "the care plan \
item"). If a P item states no care you can write as an action without \
guessing, put its id in carry_over instead - staff see it exactly as written.
- If items need the same or a similar action, combine them into ONE action \
that cites all of them - and keep every detail of each item (e.g. "upper \
dentures", "after meals"). Combining must never drop information.

TASKS
1. risks: for each assessment answer (A) that shows a risk or a need for \
support, give its id and a short reason using the answer's own words. Answers \
showing no problem are not risks.
2. note_classes: classify EVERY note item (R, C, N, G, F) as new_requirement, \
update, status or not_relevant (to this pathway).
3. not_used: a P item belongs here ONLY if a current (not "history") note item \
says it no longer applies or has changed; give that note's id as \
superseded_by. Never cite a not_used P item in an action.
4. patterns: trends over time that matter for this pathway - something \
repeating, getting worse or better, or coming back after an earlier care plan. \
Each cites at least two item ids, in one short sentence using the items' own \
words, with no cause, diagnosis or prediction. Empty if there are none.
5. actions: write the suggested actions. Each is one or two plain sentences in \
UK English, person-centred, and starts with "Staff should". Refer to the \
resident by the name given after "Resident:". Leave outcome_text empty - \
outcomes are written later, only for the actions staff add.
6. carry_over: ids of P items that are still current but state no care you \
can write as an action (see ABSOLUTE RULES). Usually empty.

HOW YOU WORK (tools)
- EVIDENCE holds every must-read record and, for each need, the resident's \
records that matched it best. The resident has MORE records than shown. Use \
SearchRecords (keywords) when something could change the plan - e.g. has this \
happened before, did a note change a previous action, what did a professional \
advise - and GetRecords to read ids you were shown. Never guess an id; cite \
only ids you have seen.
- Ids in one bracket, e.g. [C3, C9, C14], are the same note written on \
several dates - cite any or all of them.
- When ready, call SubmitPlan with the WHOLE plan. Code checks it. If it is \
not accepted you are told exactly what to fix: fix every problem, change \
nothing else, and submit the whole plan again.
- Search only when it can change the plan; don't search for what EVIDENCE \
already shows.
"""

REDRAFT_SYSTEM_PROMPT = """\
You reword one sentence (or two) of a care plan for UK care staff.

RULES
- Keep exactly the same facts and every point in the current wording. Do not \
add, remove or change any fact, number, frequency, equipment or person.
- Use only the evidence given. Refer to the resident by the name given after "Resident:".
- STYLE EXAMPLES, if given, are other residents' anonymised care plans: copy \
only their tone and vocabulary, never a fact from them.
- Write it differently from every earlier wording listed.
- Keep the same form: an ACTION starts with "Staff should"; an OUTCOME \
describes the result for the resident (e.g. "Cathy will ...").
- Plain, person-centred UK English, one or two sentences.
"""

OUTCOME_SYSTEM_PROMPT = """\
You write the intended OUTCOME for care plan actions that staff in a UK care \
home have added to a resident's plan.

For each action (ref) write one sentence describing the result for the \
resident, e.g. "Cathy will ...". Use only facts in that action's quotes - the \
outcome is the goal of the action, not new information. Never add a \
frequency, time, amount, equipment, person or diagnosis. Cite the ids of the \
quotes it relies on. Write exactly one outcome per ref given. Plain, \
person-centred UK English.
"""

VERIFY_SYSTEM_PROMPT = """\
You verify statements in a UK care home resident's care plan. Nothing else.

Each statement comes with ONLY the quotes it cites from the resident's \
records. Judge each statement against its own quotes and nothing else - not \
general knowledge, not what is usual in care.

Separate two things in every statement:
  FACTS about the resident (what she has, needs, does, prefers, how she is) - \
each must be stated in the quotes.
  The CARE RESPONSE (what staff should do) - this is the plan's job, so it is \
NOT expected to be in the quotes. It is fine when it responds to a need or \
risk the quotes show and stays proportionate: e.g. quote "Dentures: Yes" -> \
"Staff should offer Cathy assistance or prompting with her dentures" is \
supported. An OUTCOME is the intended result of its action, stated as a goal \
- fine when it follows from the action and adds no new fact.

Verdicts:
  supported   - every fact is in the quotes, and any care response answers a \
quoted need or risk proportionately. Rewording is fine.
  overstated  - it goes further than the quotes: a stronger level of need \
("must" for "may", "full assistance" for "some help", "always"), or a \
frequency, time, amount or duration the quotes don't give.
  unsupported - it states a fact about the resident (need, condition, \
equipment, person, place, routine, preference, diagnosis) that no quote \
contains, or a care response to a need no quote shows.
  contradicts - it conflicts with a quote (e.g. "encourage independence" where \
a quote says she now needs full assistance).

Give a verdict for EVERY ref. For anything but supported, the reason names the \
exact words at fault and what the quote actually says.
"""

_SOURCE_NAMES = {
    SourceType.ASSESSMENT: "Assessment",
    SourceType.PREVIOUS_CARE_PLAN: "Previous care plan",
    SourceType.REVIEW_NOTE: "Review note",
}


def format_evidence_line(item: EvidenceItem) -> str:
    details = [item.date] if item.date else []
    if item.source_type == SourceType.ASSESSMENT:
        when = f" ({details[0]})" if details else ""
        return f"[{item.id}] Assessment{when} - Q: {item.label} | A: {item.text}"
    if item.is_note:
        name = _SOURCE_NAMES.get(item.source_type, item.label)
        if not item.linked_to_pathway:
            details.append("resident record")
    elif item.source_type == SourceType.RELATED_CARE_PLAN:
        name = item.label
    else:
        name = _SOURCE_NAMES[item.source_type] + (" outcome" if item.id.startswith("O") else "")
    if item.historical:
        details.append("history")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"[{item.id}] {name}{suffix}: {item.text}"


def format_evidence(items: list[EvidenceItem]) -> str:
    return "\n".join(format_evidence_line(item) for item in items)


def format_grouped(items: list[EvidenceItem]) -> str:
    """Like format_evidence, but the same note written on several dates is
    shown once with all its ids and dates - lossless and much shorter."""
    groups: dict[tuple[str, str, bool, bool, str], list[EvidenceItem]] = {}
    for item in items:
        if not item.is_note:
            groups[(item.id, "", False, False, "")] = [item]
            continue
        key = (item.source_type.value, item.label, item.linked_to_pathway, item.historical,
               " ".join(item.text.split()).casefold())
        groups.setdefault(key, []).append(item)
    lines = []
    for group in groups.values():
        first = group[0]
        if len(group) == 1:
            lines.append(format_evidence_line(first))
            continue
        ids = ", ".join(e.id for e in group)
        dates = ", ".join(e.date for e in group if e.date)
        details = [f"x{len(group)}" + (f": {dates}" if dates else "")]
        if not first.linked_to_pathway:
            details.append("resident record")
        if first.historical:
            details.append("history")
        name = _SOURCE_NAMES.get(first.source_type, first.label)
        lines.append(f"[{ids}] {name} ({'; '.join(details)}): {first.text}")
    return "\n".join(lines)


def format_style_examples(examples: list[StyleExample]) -> str:
    if not examples:
        return ""
    blocks = [f"--- Example {n} ---\n{e.text}" for n, e in enumerate(examples, start=1)]
    return (
        "\n\nSTYLE EXAMPLES (tone, structure, length and vocabulary ONLY - not evidence, "
        "never cite, never copy a fact)\n" + "\n".join(blocks)
    )


def _draft_user_prompt(
    resident: str,
    pathway: str | None,
    evidence: list[EvidenceItem],
    examples: list[StyleExample] | None = None,
) -> str:
    return (
        f"Resident: {resident}\nPathway: {pathway or 'Not specified'}\n\n"
        f"EVIDENCE\n{format_evidence(evidence)}{format_style_examples(examples or [])}"
    )


# --- the deck's oral health example (slides 6-13), made strictly grounded ---
_EXAMPLE_EVIDENCE = [
    EvidenceItem(id="A1", source_type=SourceType.ASSESSMENT, label="Dentures", text="Yes",
                 date="14/10/2026"),
    EvidenceItem(id="A2", source_type=SourceType.ASSESSMENT, label="Oral cleanliness",
                 text="Some particles/tartar/plaque in a few areas", date="14/10/2026"),
    EvidenceItem(id="A3", source_type=SourceType.ASSESSMENT, label="Dental pain", text="No",
                 date="14/10/2026"),
    EvidenceItem(id="P1", source_type=SourceType.PREVIOUS_CARE_PLAN,
                 label="Previous care plan - action",
                 text="Cathy is independent with her oral care needs.", date="20/05/2026"),
    EvidenceItem(id="P2", source_type=SourceType.PREVIOUS_CARE_PLAN,
                 label="Previous care plan - action",
                 text="She needs prompting from staff.", date="20/05/2026"),
    EvidenceItem(id="P3", source_type=SourceType.PREVIOUS_CARE_PLAN,
                 label="Previous care plan - action",
                 text="Cathy has upper dentures and natural bottom teeth.", date="20/05/2026"),
    EvidenceItem(id="O1", source_type=SourceType.PREVIOUS_CARE_PLAN,
                 label="Previous care plan - outcome",
                 text="Cathy will not experience any oral discomfort.", date="20/05/2026"),
    EvidenceItem(id="R1", source_type=SourceType.REVIEW_NOTE, label="Review note",
                 text="Cathy is still using her dentures and will require further prompting "
                      "from staff to carry out regular oral hygiene.", date="08/10/2026"),
    EvidenceItem(id="R2", source_type=SourceType.REVIEW_NOTE, label="Review note",
                 text="Staff may be required to assist with providing oral hygiene.",
                 date="08/10/2026"),
    EvidenceItem(id="N1", source_type=SourceType.NURSING_NOTE, label="Nursing note",
                 text="Cathy said her gums felt sore when brushing this morning.",
                 date="12/10/2026", linked_to_pathway=False),
]

_EXAMPLE_OUTPUT = DraftPlanOut(
    risks=[
        RiskFlagOut(evidence_id="A1", reason="Has dentures"),
        RiskFlagOut(evidence_id="A2", reason="Some particles/tartar/plaque in a few areas"),
    ],
    note_classes=[
        NoteClassOut(evidence_id="R1", kind="update"),
        NoteClassOut(evidence_id="R2", kind="new_requirement"),
        NoteClassOut(evidence_id="N1", kind="new_requirement"),
    ],
    not_used=[
        NotUsedOut(evidence_id="P1", superseded_by="R2",
                   reason="Review note says staff may need to assist with oral hygiene."),
    ],
    patterns=[
        PatternOut(summary="Cathy has gone from needing prompting with oral hygiene to "
                           "staff possibly needing to assist with it.",
                   evidence_ids=["P2", "R1", "R2"]),
    ],
    actions=[
        ActionOut(
            text="Staff should offer Cathy assistance or prompting with her upper dentures "
                 "and support her to keep using them.",
            evidence_ids=["A1", "P3", "R1"],
        ),
        ActionOut(
            text="Staff should prompt Cathy to carry out oral hygiene to keep particles, "
                 "tartar and plaque from building up, and may need to assist her with it, "
                 "taking care as her gums have felt sore when brushing.",
            evidence_ids=["A2", "P2", "R1", "R2", "N1"],
        ),
    ],
)


AGENT_SYSTEM_PROMPT = (
    DRAFT_SYSTEM_PROMPT
    + "\nWORKED EXAMPLE (another resident)\n"
    + _draft_user_prompt("Cathy", "Oral Health Assessment", _EXAMPLE_EVIDENCE)
    + "\n\nSubmitPlan:\n"
    + json.dumps(_EXAMPLE_OUTPUT.model_dump())
)


def build_agent_messages(
    *,
    resident_name: str | None,
    pathway: str | None,
    evidence: list[EvidenceItem],
    examples: list[StyleExample] | None = None,
) -> list[BaseMessage]:
    """The care plan agent's opening messages. The system prompt (rules, tools,
    worked example) is identical for every request - the cacheable prefix."""
    return [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Resident: {resident_name or 'the resident'}\n"
            f"Pathway: {pathway or 'Not specified'}\n\n"
            f"EVIDENCE\n{format_grouped(evidence)}{format_style_examples(examples or [])}"
        ),
    ]


def build_redraft_messages(
    *,
    resident_name: str | None,
    target: str,
    current_text: str,
    earlier_wordings: list[str],
    evidence: list[EvidenceItem],
    action_text: str | None,
    examples: list[StyleExample] | None = None,
) -> list[BaseMessage]:
    parts = [
        f"Resident: {resident_name or 'the resident'}",
        "",
        f"EVIDENCE\n{format_evidence(evidence)}",
        "",
    ]
    if target == "outcome" and action_text:
        parts += [f"This is the intended outcome of the action: {action_text}", ""]
    parts.append(f"CURRENT {target.upper()} WORDING:\n{current_text}")
    if earlier_wordings:
        parts += ["", "EARLIER WORDINGS (do not repeat):"]
        parts += [f"- {w}" for w in earlier_wordings]
    style = format_style_examples(examples or [])
    if style:
        parts.append(style)
    return [
        SystemMessage(content=REDRAFT_SYSTEM_PROMPT),
        HumanMessage(content="\n".join(parts)),
    ]


def build_outcome_messages(
    *,
    resident_name: str | None,
    actions: list[tuple[str, str, list[EvidenceItem]]],
    examples: list[StyleExample] | None = None,
    feedback: dict[str, str] | None = None,
) -> list[BaseMessage]:
    """actions: (ref, action text, the quotes behind it). feedback: ref -> a
    rejected earlier outcome and why, so the rewrite avoids it."""
    feedback = feedback or {}
    blocks = []
    for ref, text, quotes in actions:
        block = f"[{ref}] ACTION: {text}\nQUOTES:\n{format_evidence(quotes)}"
        if ref in feedback:
            block += (f"\nREJECTED EARLIER OUTCOME (not supported by the quotes): "
                      f"{feedback[ref]}\nWrite one that says only what the quotes support.")
        blocks.append(block)
    return [
        SystemMessage(content=OUTCOME_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Resident: {resident_name or 'the resident'}\n\n" + "\n\n".join(blocks)
            + format_style_examples(examples or [])
        ),
    ]


def build_verify_messages(
    *, resident_name: str | None, statements: list[tuple[str, str, list[EvidenceItem]]]
) -> list[BaseMessage]:
    """statements: (ref, text, the quotes it cites)."""
    blocks = []
    for ref, text, quotes in statements:
        blocks.append(f"[{ref}] {text}\nQUOTES:\n{format_evidence(quotes)}")
    return [
        SystemMessage(content=VERIFY_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Resident: {resident_name or 'the resident'}\n\n" + "\n\n".join(blocks)
        ),
    ]
