"""Prompts for the two agents: the care plan agent and the verifier agent.

The system prompt and few-shot example are fixed text placed first in every
request, so Azure OpenAI's prompt caching can discount them; only the
evidence block (and any style examples) at the end changes per resident.
"""

import json
import re

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from careplan_ai.agents.care_plan.schemas import (
    ActionOut,
    DraftPlanOut,
    EvidenceItem,
    NoteClassOut,
    NotUsedOut,
    PatternOut,
    RiskFlag,
    RiskFlagOut,
    SourceType,
)
from careplan_ai.library.models import StyleExample, pathway_key

DRAFT_SYSTEM_PROMPT = """\
You help care staff in a UK care home write a care plan for one resident and \
one pathway. You do not decide anything - staff add or dismiss every suggestion.

You are given EVIDENCE: numbered items from this resident's own records -
  A = answers from the latest assessment
  P = actions from this resident's previous care plan (O = its outcomes)
  R = review notes; C = carer notes; N = nursing notes;
  G = external professional notes (GP, district nurse, ...); F = relatives
  H = this resident's earlier care plans (archived, or for other pathways)
  V = observation chart entries (fluid, food, TPR, falls diary, ...), one per entry
  T = one chart's summary line, computed from all its entries
  K = a care standard step: the routine care the recorded score calls for \
(e.g. the steps for the MUST band in the assessment), or how often to offer \
drinks for a chart's daily fluid target
A note marked "resident record" was not written against this pathway; an \
observation marked "resident record" is from a chart not linked to it. Items \
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
relevant, every note item of kind "new_requirement" and every K item must be \
cited by at least one action.
- K items are the care standard for what the records show. Write each K step \
as an action for the resident, citing the K item AND the A, V or T items it \
responds to (e.g. the MUST score answer, the Food chart). Keep its frequency \
and its trigger exactly ("weekly", "every 1-2 hours", "less than half of each \
meal for 3 days"); add nothing a K item does not say.
- Every chart linked to this pathway (V or T, not "supporting") with entries \
needs an action that cites it - e.g. a Food chart showing how much of each \
meal was eaten needs a food action that cites it.
- An action is the care response to a need or risk in the evidence. Where the \
evidence shows only the risk (e.g. "Dentures: Yes"), write the support \
tentatively, as staff would: "Staff should offer Cathy assistance or \
prompting with her dentures". Never invent a level of help, frequency, time, \
amount or piece of equipment.
- An action describes the care itself. Never refer to the records in it \
("follow the previous care plan", "as per the assessment", "the care plan \
item"). If a P item states no care you can write as an action without \
guessing, put its id in carry_over instead - staff see it exactly as written.
- ONE ACTION PER NEED. If several items show the SAME need, write ONE action \
that cites all of them - and keep every detail of each item (e.g. "upper \
dentures", "after meals"). Combining must never drop information. Never put \
different needs in one action (walker, sensor mat and toileting are three \
actions), and never write two actions for the same need (taking her to the \
bathroom at night is one action, however many items mention it).
- PREVENT, DON'T WAIT. Write what staff do routinely to stop a risk happening \
("walk with Cathy to the toilet before bed and during the night"), not only \
what to do once it has happened ("if Cathy says she feels light-headed, offer \
support"). Use only the steps and details the evidence gives.
- An action says only WHAT staff do. It never explains why ("Following \
review, he is no longer independent...", "... because she was coughing", \
"... as she is no longer safe to walk alone", "... noting she needed help \
twice", "... since she fell" - the reason is in the cited evidence, which \
staff see under View). Never use "as she / as he", "because", "since", \
"noting" or "given that" in an action. It is never \
paperwork about the records ("document that a dentist appointment has been \
requested", "document that she was weighed and file it with the chart"): write \
the care it leads to, or leave it out. Recording on a chart IS care when the \
evidence asks for it ("record every drink on the Fluid chart").
- P and H items are evidence of what was planned before, not wording to copy. \
Where an earlier plan's line gives a reason, is paperwork or breaks any other \
rule here, write the care it describes by these rules instead.
- An action about equipment says how staff use it, not its status ("place the \
sensor mat by Cathy's bed at night and respond as soon as it alerts", not \
"arrange a sensor mat as requested"), using only details the evidence gives.
- STAY IN THE PATHWAY. Every action is about the pathway named after \
"Pathway:". Content that belongs to another pathway (e.g. a dentist \
appointment in a MUST plan) is used ONLY where it directly affects this \
pathway, and is then worded for this pathway (e.g. "Staff should check \
Cathy's denture is in and comfortable before meals").
- Each action is ONE thing staff do, stated clearly, with how often or when \
if the evidence gives it ("record every drink on the Fluid chart", not "keep \
the charts that have been started"; "weigh her weekly" only if the evidence \
says weekly).
- Where the assessment records a score or risk level (e.g. "MUST score 1 - \
Medium risk"), the actions must respond to it, using the score and level in \
the evidence's own words.
- Where a chart linked to this pathway (V or T, not "supporting") shows \
amounts eaten or drunk, readings or a target, write the action that responds \
to it and cite those V/T items - e.g. a Fluid chart showing intake below its \
recorded target needs a fluid action that cites it.
- V items are readings staff recorded on a chart; T items summarise a chart. \
Quote their figures exactly as written: never calculate a total, average or \
difference, and never turn a reading into a diagnosis or a risk the evidence \
does not state. A reading can support an action or a pattern, but never makes \
a P item not_used. V and T items marked "supporting" are context for this \
pathway only: cite them alongside other evidence, never as the only evidence \
for an action.

TASKS
1. risks: for each assessment answer (A) that shows a risk or a need for \
support, give its id and a short reason using the answer's own words. Answers \
showing no problem are not risks. Every answer marked [FLAGGED RISK] is a \
risk: list it here and cover it with an action (e.g. a medicines answer \
needs a medicines action, a footwear answer a footwear action).
2. note_classes: classify EVERY note item (R, C, N, G, F - not V) as \
new_requirement, update, status or not_relevant (to this pathway).
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

# Splits the outcome rules from their per-pathway measures (see outcome_wording).
_MEASURES_HEADING = "WHAT A GOOD OUTCOME MEASURES"

OUTCOME_SYSTEM_PROMPT = """\
You write the intended OUTCOMES of a UK care home resident's care plan: the \
results the actions staff have added should achieve, for the pathway named \
after "Pathway:".

How many: one outcome for each need or flagged risk the actions answer - not \
one per action, and no fixed number. Combine two needs into one outcome only \
when they share the same result. An action that only supports another outcome \
(recording on a chart, a referral if things get worse, a check before meals) \
gets no outcome of its own: list it under the outcome it supports. So does an \
action whose need is not one this pathway measures (see below).

For each outcome give:
- ref: the action it measures most directly. No two outcomes share a ref.
- also_measures: the refs of the other actions it measures or depends on.
- text and evidence_ids.

Every outcome must:
1. Be a RESULT for the resident, not a staff task: "Cathy drinks the daily \
amount her Fluid chart sets", never "Cathy will be offered drinks". Never a \
record standing in for the result: never "Cathy is recorded as drinking ...".
2. Be checkable at the next review: a number, a frequency, a yes/no or a clear \
change (e.g. "no further falls", "redness resolved").
3. Link to the plan: follow from its actions and the need or risk in their \
quotes. Every risk listed after "Flagged risks:" needs an outcome.
4. Use ONLY targets recorded in its quotes (assessment, chart target, SLT, GP, \
dietitian or nurse advice, or a K care standard: "tell the nurse if she eats \
less than half of each meal" sets "Cathy eats at least half of each main \
meal"; "rescreen MUST monthly" sets "Cathy's MUST score is the same or lower \
at the monthly rescreen"). If no target is recorded, use general wording \
("Cathy's weight stays stable or goes up by the next review") - never invent a \
number, frequency, time, amount, equipment, person or diagnosis.
5. Stay in the pathway: nothing from another pathway, no placeholder text, no \
two outcomes saying the same thing. Set each goal ONCE: a "no further ..." \
goal is in one outcome only, and every other outcome measures a different \
result.
6. Be one short plain-English sentence with one result (not "uses her frame \
and has no falls"), using the resident's name given after "Resident:". No \
diagnosis, medicine name or dose unless it is in the quotes.
7. Cite the ids of the quotes it relies on.

""" + _MEASURES_HEADING + """, by pathway (a pattern - the facts and targets \
still come only from the quotes):
- MUST: weight trend, share of each meal eaten, fluid against target, MUST \
score at rescreen.
- Diet and Fluid Notification: the right diet and fluid type every time; \
intake against target.
- Choking Risk: eats and drinks only in the advised IDDSI textures; no \
coughing, choking or chest infection.
- Falls: number of falls by review; safe use of the walking aid; dizziness on \
standing reduced; safe toileting at night.
- Oral Health: mouth clean and moist; no pain; denture worn and fits; dentist \
seen.
- Waterlow: skin intact; Waterlow score the same or lower; no new pressure \
damage.
- Skin Integrity: skin intact; redness or dryness resolved; no new skin tears.
- Wound Care: wound size reducing; no signs of infection; healed by the review \
date where recorded.
- Bladder and Bowel: dry and comfortable; bowels open as often as the records \
say; no UTI; uses the toilet with support.
- Stoma Care: stoma and the skin around it healthy; no leaks; bag changed as \
planned.
- Stress and Distress: fewer distressed episodes; settles with the agreed \
approach; takes part in chosen activities.
- Assessment of Dependency: takes part in daily care as independently as \
possible; needs met at the assessed level.
- PRN Medication: PRN given only when the recorded criteria are met; effect \
recorded; no unmet pain or distress.
- End of Life Wishes: wishes known, recorded and followed; comfortable and \
pain-free; family involved as wished.
- Bedroom Fire Risk Assessment (BFRA): room kept safe; evacuation plan known; \
risks removed.
- Admission Care Plan: settles into the home; preferences known and followed; \
key assessments completed.
- Any other pathway: comfort, safety, independence, wellbeing or a completed \
referral, linked to the flagged risks.

Use only the refs given.
"""


def _names_pathway(name: str, pathway: str) -> bool:
    """A measures line ("Choking Risk", "Bedroom Fire Risk Assessment (BFRA)") is for this
    pathway when its name, or the short name in brackets, is in the pathway's name."""
    full, _, short = name.partition("(")
    names = [pathway_key(full), pathway_key(short.rstrip(")"))]
    return any(n and re.search(rf"\b{re.escape(n)}\b", pathway) for n in names)


def outcome_wording(pathway: str | None) -> str:
    """The wording the outcome rules themselves give for this pathway: the general rules and
    this pathway's measures line (or the "any other pathway" line), read from
    OUTCOME_SYSTEM_PROMPT - so an outcome using it was not copied from a style example."""
    rules, _, measures = OUTCOME_SYSTEM_PROMPT.partition(_MEASURES_HEADING)
    lines = dict(
        line[2:].split(": ", 1) for line in measures.splitlines()
        if line.startswith("- ") and ": " in line
    )
    key = pathway_key(pathway or "")
    chosen = [text for name, text in lines.items() if key and _names_pathway(name, key)]
    if not chosen:  # the last line is the one for any other pathway
        chosen = list(lines.values())[-1:]
    return " ".join([rules, *chosen])

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
- fine when it follows from the action and adds no new fact. A measurable goal \
is fine when its target is in the quotes (e.g. a fluid target of 1500 ml) or \
it uses general wording ("weight stays stable"); a target no quote gives is \
overstated.

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

READ_SYSTEM_PROMPT = """\
You read part of a UK care home resident's records for ONE care plan pathway, \
so that nothing relevant to it is missed. You do not write the care plan.

PATHWAY CONTEXT shows what the pathway's assessment and previous care plan \
already say. RECORDS are numbered lines (V = chart readings, T = chart \
summaries); ids in one bracket are the same record written on several dates - \
judge them together and answer with the FIRST id of the line.

For EVERY record line decide one of:
  relevant, with a kind and a reason of at most 12 words in the record's words:
    new_requirement - raises a care need, risk or action for this pathway that \
the context doesn't already show
    update - changes or confirms something the context shows
    status - useful information for this pathway, no action needed
  not_relevant - nothing to do with this pathway.
When unsure, mark it relevant. Every line appears exactly once, in relevant \
or in not_relevant. Use only ids shown.
"""

_SOURCE_NAMES = {
    SourceType.ASSESSMENT: "Assessment",
    SourceType.PREVIOUS_CARE_PLAN: "Previous care plan",
    SourceType.REVIEW_NOTE: "Review note",
}


def _source_name(item: EvidenceItem) -> str:
    if item.is_note:
        return _SOURCE_NAMES.get(item.source_type, item.label)
    if item.is_observation:
        return f"Observation - {item.label}"
    if item.is_chart_summary:
        return f"Observation summary - {item.label}"
    if item.source_type in (SourceType.RELATED_CARE_PLAN, SourceType.CARE_STANDARD):
        return item.label
    return _SOURCE_NAMES[item.source_type] + (" outcome" if item.id.startswith("O") else "")


def format_evidence_line(item: EvidenceItem) -> str:
    details = [item.date] if item.date else []
    if item.source_type == SourceType.ASSESSMENT:
        when = f" ({details[0]})" if details else ""
        flag = " [FLAGGED RISK]" if item.risk_flagged else ""
        return f"[{item.id}] Assessment{when}{flag} - Q: {item.label} | A: {item.text}"
    name = _source_name(item)
    on_record = item.is_note or item.is_observation or item.is_chart_summary
    if item.supporting:
        details.append("supporting")
    elif on_record and not item.linked_to_pathway:
        details.append("resident record")
    if item.historical:
        details.append("history")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"[{item.id}] {name}{suffix}: {item.text}"


def format_evidence(items: list[EvidenceItem]) -> str:
    return "\n".join(format_evidence_line(item) for item in items)


def group_repeats(items: list[EvidenceItem]) -> list[list[EvidenceItem]]:
    """Items in order, with the same note (or chart entry) written on several
    dates gathered into one group - one line of format_grouped each."""
    groups: dict[tuple[str, str, bool, bool, str], list[EvidenceItem]] = {}
    for item in items:
        if not (item.is_note or item.is_observation):
            groups[(item.id, "", False, False, "")] = [item]
            continue
        key = (item.source_type.value, item.label, item.linked_to_pathway, item.historical,
               " ".join(item.text.split()).casefold())
        groups.setdefault(key, []).append(item)
    return list(groups.values())


def format_grouped(items: list[EvidenceItem]) -> str:
    """Like format_evidence, but the same note (or chart entry) written on
    several dates is shown once with all its ids and dates - lossless and much
    shorter."""
    lines = []
    for group in group_repeats(items):
        first = group[0]
        if len(group) == 1:
            lines.append(format_evidence_line(first))
            continue
        ids = ", ".join(e.id for e in group)
        dates = ", ".join(e.date for e in group if e.date)
        details = [f"x{len(group)}" + (f": {dates}" if dates else "")]
        if first.supporting:
            details.append("supporting")
        elif not first.linked_to_pathway:
            details.append("resident record")
        if first.historical:
            details.append("history")
        name = _source_name(first)
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
    EvidenceItem(id="V1", source_type=SourceType.OBSERVATION, label="Oral Health",
                 text="Oral care: Prompted; Gums: Sore", date="13/10/2026"),
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
            evidence_ids=["A2", "P2", "R1", "R2", "N1", "V1"],
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
    pathway: str | None = None,
    risks: list[RiskFlag] | None = None,
) -> list[BaseMessage]:
    """actions: (ref, action text, the quotes behind it). feedback: ref -> a
    rejected earlier outcome and why, so the rewrite avoids it. risks: the
    assessment's flagged risks, each of which needs an outcome."""
    feedback = feedback or {}
    flagged = "\n".join(f"- [{r.evidence_id}] {r.reason}" for r in risks or []) or "none"
    blocks = []
    for ref, text, quotes in actions:
        block = f"[{ref}] ACTION: {text}\nQUOTES:\n{format_evidence(quotes)}"
        if ref in feedback:
            block += (f"\nREJECTED EARLIER OUTCOME (not supported by the quotes): "
                      f"{feedback[ref]}\nWrite one that says only what the quotes support.")
        blocks.append(block)
    if feedback:
        blocks.append("Write only the outcomes that replace the rejected ones, each with "
                      "the ref it was rejected for; the other actions are there for context.")
    return [
        SystemMessage(content=OUTCOME_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Resident: {resident_name or 'the resident'}\n"
            f"Pathway: {pathway or 'Not specified'}\n"
            f"Flagged risks:\n{flagged}\n\n" + "\n\n".join(blocks)
            + format_style_examples(examples or [])
        ),
    ]


def build_read_messages(
    *,
    resident_name: str | None,
    pathway: str | None,
    context: list[EvidenceItem],
    records: list[EvidenceItem],
) -> list[BaseMessage]:
    """A reader agent's messages: the fixed prompt first (cacheable), then this
    pathway's context and its chunk of records."""
    return [
        SystemMessage(content=READ_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Resident: {resident_name or 'the resident'}\n"
            f"Pathway: {pathway or 'Not specified'}\n\n"
            f"PATHWAY CONTEXT\n{format_evidence(context) or 'None'}\n\n"
            f"RECORDS\n{format_grouped(records)}"
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
