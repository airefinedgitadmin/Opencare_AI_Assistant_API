"""Code-level checks on the model's draft - the "no hallucination, nothing
missed" rules from the deck's Guardrails slide, enforced deterministically
rather than by asking another model.

1. References: every cited id must exist and be the right kind of evidence;
   anything else is discarded and counted. An action left with no valid
   evidence is dropped - it can't be traced to a source, so it can't be shown.
2. Completeness: every flagged risk, every previous care plan action that no
   note supersedes, and every note requirement must be covered by at least
   one action. An uncovered earlier care plan line or care standard step is
   added back as its own card using the source's own words - no model
   involved. An uncovered flagged answer, note or chart line is never pasted
   in as an action (OPENCARE-114): the answer stays a risk staff see as "Not
   covered", and the record is listed in a notice.
3. Nothing removed when combining: words of a required source that the action
   citing it doesn't carry are listed on the card (possibly_missing).
4. Unsupported details: a figure, or a name / proper noun, in a suggestion
   that isn't in its cited evidence is flagged and lowers confidence.
5. Style examples (OPENCARE-51): a suggestion that borrows a detail from an
   example that this resident's records don't contain is dropped (outcome:
   just the outcome); its sources are then added back by rule 2.
6. Confidence is computed from the evidence, never self-reported by the model.

Every problem is also recorded in plain words (ValidationResult.problems) -
that is the feedback the care plan agent revises its draft from.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from careplan_ai.agents.care_plan.schemas import (
    ActionCard,
    CardOrigin,
    Confidence,
    DraftPlanOut,
    EvidenceItem,
    MissingDetail,
    NotUsedItem,
    Pattern,
    PatternOut,
    RiskFlag,
    SourceType,
    Suggestion,
)
from careplan_ai.agents.care_plan.standards import thickened_level
from careplan_ai.agents.care_plan.superseded import describes_state, rule_not_used, state_lines

_NUMBER = re.compile(r"\d+(?:[.:/]\d+)*")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+")
_PLACEHOLDER = re.compile(
    r"\[(?:Resident|Person|Room|Date|Date of birth|ID|Phone|Email|Postcode)\]"
)

UNSUPPORTED_NUMBER_WARNING = "Contains a number that isn't in its sources - check before adding."
UNSUPPORTED_NAME_WARNING = (
    "Names someone or something that isn't in its sources - check before adding."
)
UNADDRESSED_RISK_WARNING = "Risk flagged in the assessment - write an action for it."
# An action that points at the records instead of describing the care.
_RECORD_REFERENCE = re.compile(
    r"\b(?:previous|earlier|last|old|existing)\s+care\s*plans?\b"
    r"|\bcare\s*plan\s+(?:item|entry|point|action)s?\b"
    r"|\bas\s+per\s+(?:the\s+)?(?:assessment|records?|notes?)\b",
    re.IGNORECASE,
)
# OPENCARE-78: an action that gives the reason ("Following review, ...") or is only paperwork about
# the records ("document that a dentist appointment has been requested") instead of the care.
# "Following review, ..." as a lead-in (with its comma) - "following review by the TVN" is fine.
# Paperwork also covers writing down that something happened ("document that she was weighed",
# "note that a referral has been sent") or filing it ("file this with the chart"). A fact about
# the resident ("note that she has an upper denture") and "record every drink on the Fluid
# chart" are not paperwork and are not matched.
_PAPERWORK_PATTERN = (
    r"\b(?:document|note|record|log)\s+that\b.*\b(?:has|have)\s+been\s+"
    r"(?:requested|made|booked|arranged|sent|referred)\b"
    r"|\b(?:document|note|record|log)\s+that\s+(?:\S+\s+){1,3}?"
    r"(?:was|were|had|(?:has|have)\s+been)\b"
    r"|\bfile\s+(?:this|it|these|them)\b"
)
_PAPERWORK = re.compile(_PAPERWORK_PATTERN, re.IGNORECASE)
_NOT_CARE = re.compile(
    r"\bfollowing\s+(?:the\s+|a\s+|his\s+|her\s+|their\s+|this\s+)?(?:recent\s+)?reviews?\s*,"
    r"|" + _PAPERWORK_PATTERN,
    re.IGNORECASE,
)
# A sentence whose own instruction is paperwork ("Staff should document that ..."), as opposed to
# care that also mentions recording ("Staff should help Cathy eat and document that ...").
_PAPERWORK_LEAD = re.compile(
    r"^\s*(?:staff\s+(?:should|must|will|to)\s+)?(?:document|note|record|log|file)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A reason clause in an action ("... because she was coughing", "due to ..."). The action is still
# care, so it is kept and only the reason is taken out. "Medication due to be given" is a time,
# not a reason.
# OPENCARE-98 adds the reasons the Falls retest found - "... at night as she is no longer safe to
# walk alone", "... noting she has needed help twice", "... since she fell", "given that ...".
# Only those forms: "noting any redness" is a check, "as far as she is able" / "as soon as she
# wakes" / "as she prefers" are care, and "since admission" is a time.
_PERSON = r"(?:she|he|they)"
_REASON_WORDS = (
    r"\b(?:because|due\s+to(?!\s+be\b)|as\s+a\s+result\s+of|owing\s+to"
    r"|(?<!far\s)(?<!soon\s)(?<!much\s)(?<!long\s)(?<!well\s)(?<!often\s)(?<!such\s)"
    r"as\s+" + _PERSON + r"\s+(?:is|was|has|had|no\s+longer|cannot|can't|does\s+not|doesn't|"
    r"did\s+not|didn't|keeps|tends)"
    # OPENCARE-99: "... small portions with sweet options as she prefers these" gives the reason;
    # "as she prefers it" / "as she prefers" (the way she likes it) is still care.
    r"|as\s+" + _PERSON + r"\s+(?:prefers|likes|enjoys|dislikes)\s+(?:these|them|this|that|those)"
    r"|noting\s+(?:that\s+)?" + _PERSON + r"|since\s+" + _PERSON + r"|given\s+that)\b"
)
# OPENCARE-99: an action that tacks on how things stand ("...; a dietitian referral has been
# requested and she has a poor appetite reported to the GP") - two things in one, and a status is
# not care. Kept (the care is still there); the agent is asked to write one action per need.
# Only a clause that IS a status - "; <a/the/her ...> <up to 4 words> has been <done>" - so care
# after a semicolon ("; if a new medicine has been started, check ...", "; check whether the
# referral has been made") is never matched.
_STATUS_TAIL = re.compile(
    r";\s*(?:a|an|the|her|his|their)\s+[\w'’-]+(?:\s+[\w'’-]+){0,3}?\s+(?:has|have|had)\s+been\s+"
    r"(?:requested|made|started|sent|reported|arranged|booked|referred|informed|told)\b",
    re.IGNORECASE,
)
_REASON = re.compile(_REASON_WORDS, re.IGNORECASE)
# The reason clause runs to the next comma, semicolon or full stop - a dot or comma inside a
# number ("46.8 kg", "1,500 ml") is part of it.
_REASON_CLAUSE = re.compile(
    r"(,\s*)?" + _REASON_WORDS + r"(?:[^,.;]|[.,](?=\d))*(,)?", re.IGNORECASE
)


# OPENCARE-114: a line that is not an instruction for staff but form or chart text - a question
# with its answer ("Is the Resident oriented to their new environment?: Yes", "...; normal bowel
# habit: No; ..."), a form field ("Sex: F", "Age: 81+", "Special Risks - Medicines: 2", "sex (M)
# and age (81+)"), a score or chart field tacked on ("in response to a MUST Score 1 - Medium risk",
# "His overall score is 3.", "a Monthly record type") or a chart line ("24/11/2026 15:33 ·
# Observation: Found Awake", "2 entries on 16/11/2026").
_FORM_LINE = re.compile(
    r"\?\s*:"
    r"|:\s*(?:yes|no)\b\s*(?:[.;,]|$)"
    r"|^\s*(?:sex|age|gender|total\s+score|score|risk\s+level)\s*:"
    r"|\bspecial\s+risks?\b"
    r"|\bsex\s*\(\s*[mf]\s*\)|\bage\s*\(\s*\d+\s*\+?\s*\)|\b(?:is|are)\s+[MF]\s+and\s+\d+\s*\+"
    r"|^\s*\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d+\s+entr(?:y|ies)\s+on\b"
    r"|\bin\s+response\s+to\s+(?:a\s+|an\s+|the\s+|his\s+|her\s+|their\s+)?[\w-]+\s+score\b"
    r"|\boverall\s+score\s+(?:is|was|of)\b"
    r"|\brecord\s+type\b",
    re.IGNORECASE,
)
# OPENCARE-115: an action about drinks ("Offer Rab a drink every 1-2 hours"), which in a Choking
# plan with thickened fluids must name the level.
_DRINK = re.compile(r"\b(?:drinks?|fluids?|cups?\s+of|sips?)\b", re.IGNORECASE)
# OPENCARE-114: an action that is a note or chart entry copied (nearly) word for word ("Nell eats
# quickly and sometimes needs a reminder to slow down ...") is a record, not an instruction. Only a
# record that is not itself written as an instruction counts: a review note that says "Staff
# should reposition Willie every 2 hours" is care, and an action using it is right.
COPIED_RECORD_OVERLAP = 0.8
_INSTRUCTION_WORDS = re.compile(
    r"\b(?:staff|should|must|please|ensure|encourage|offer|assist|support|remind|check)\b",
    re.IGNORECASE,
)


def not_an_instruction(text: str, records: list[EvidenceItem]) -> str | None:
    """Why `text` is form or record text rather than an instruction for staff, or None.
    `records` are the notes and chart lines the model was shown."""
    if _FORM_LINE.search(text):
        return (
            "is an assessment question, form field, score or chart line, not an instruction"
        )
    plain = _plain(text)
    for record in records:
        if _INSTRUCTION_WORDS.search(record.text):
            continue
        # The same text, not the same subject: "Staff should give full help with denture
        # cleaning." is care written from "Denture cleaning now needs full help.", not a copy.
        if SequenceMatcher(None, plain, _plain(record.text)).ratio() > COPIED_RECORD_OVERLAP:
            return f"copies [{record.id}] word for word instead of saying what staff do"
    return None


def _plain(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _shares_phrase(action: str, line: str) -> bool:
    """OPENCARE-113: the action carries three words in a row of `line` with two content words
    among them ("walks independently with" her walker), word endings aside."""
    action_stems = [_stem(w) for w in _WORD.findall(action)]
    line_stems = [_stem(w) for w in _WORD.findall(line)]
    for i in range(len(line_stems) - 2):
        window = line_stems[i:i + 3]
        if sum(is_content_stem(s) for s in window) < 2:
            continue
        for j in range(len(action_stems) - 2):
            if action_stems[j:j + 3] == window:
                return True
    return False


def _repeats_not_used(
    text: str,
    not_used: list[NotUsedItem],
    by_id: dict[str, EvidenceItem],
    forbidden: dict[str, "re.Pattern[str]"],
    resident_name: str | None = None,
) -> NotUsedItem | None:
    """OPENCARE-113: the not-used line an action repeats - its words mostly (over 60%), a phrase
    of it, or (for a code rule) the very thing the rule rules out ("cutting up his meat")."""
    names = stems([resident_name or ""])
    action_words = content_stems([text]) - names
    for item in not_used:
        line = by_id.get(item.evidence_id)
        if line is None:
            continue
        pattern = forbidden.get(item.evidence_id)
        if pattern is not None and pattern.search(text):
            return item
        # An action that keeps the line's words but changes its figure ("change position every
        # 2 hours" for "... every 4 hours") is the new care, not the old line repeated.
        figures = set(_NUMBER.findall(line.text))
        if figures and not figures <= set(_NUMBER.findall(text)):
            continue
        if _shares_phrase(text, line.text):
            return item
        # Overlap only means something for a line with a few content words of its own: "Cathy is
        # independent with her oral care" shares "oral" with any oral care action.
        line_words = content_stems([line.text]) - names
        union = line_words | action_words
        if (len(line_words) >= _MIN_CONTAINED_WORDS - 1 and union
                and len(line_words & action_words) / len(union) > SAME_NEED_OVERLAP):
            return item
    return None


# OPENCARE-98: an action that waits for the problem ("If Agnes says she feels light-headed when
# standing, staff should offer support") instead of preventing it.
_REACTIVE = re.compile(
    r"\b(?:[Ii]f|[Ww]hen)\s+(?:she|he|they|[A-Z][a-z]+)\s+(?:says|said|feels|reports|complains"
    r"|tells|becomes|is\s+found)\b",
)

# Two actions are the same need when this share of their content words is shared (ticket: more
# than 60% token overlap).
SAME_NEED_OVERLAP = 0.6


def word_overlap(a: str, b: str) -> float:
    """Shared content words over all content words of the two texts (0..1). The resident's name
    and "staff should" style words don't count."""
    first, second = content_stems([a]), content_stems([b])
    union = first | second
    return len(first & second) / len(union) if union else 0.0


# OPENCARE-99: or when nearly all of the shorter action's content words (4 or more) are in the
# other - "offer small portions and sweet foods" inside a longer preferences action.
SAME_NEED_CONTAINED = 0.75
_MIN_CONTAINED_WORDS = 4


def _contained(a: str, b: str) -> bool:
    first, second = content_stems([a]), content_stems([b])
    shorter = min(len(first), len(second))
    return shorter >= _MIN_CONTAINED_WORDS and len(first & second) / shorter > SAME_NEED_CONTAINED


def overlapping_actions(texts: list[str]) -> list[tuple[int, int]]:
    """Index pairs of texts that say mostly the same thing."""
    return [
        (i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if word_overlap(texts[i], texts[j]) > SAME_NEED_OVERLAP or _contained(texts[i], texts[j])
    ]


# OPENCARE-98: an outcome where a record stands in for the result ("Agnes is recorded as using her
# walker") is not a result for the resident. Only "recorded AS ...": "each PRN dose is recorded
# with its effect" and "preferences are recorded and followed" are results the rules ask for.
# OPENCARE-99 adds the record as the subject: "Agnes's recorded daily fluid intake reaches ...".
_RECORDING_OUTCOME = re.compile(
    r"\b(?:is|are|was|were|be|being|been)\s+(?:recorded|documented|noted|logged|charted)\s+as\b"
    # Only "recorded" describing the subject ("Agnes's / her / the recorded ... intake"), not a
    # record being kept of a result ("each drink is recorded with the amount she drinks").
    r"|(?:['’]s|\bher|\bhis|\btheir|\bthe)\s+(?:recorded|documented|logged|charted)\s+"
    r"(?:[a-z]+\s+){0,2}(?:intake|amounts?|weights?|totals?|readings?)\b",
    re.IGNORECASE,
)
# The "no <thing>" goal of an outcome ("has no further falls", "no new skin tears"), so the same
# goal is set by one outcome only.
# Not "no longer", "no more than", "no less than", "no later than" - those aren't goals.
_NO_GOAL = re.compile(
    r"\bno\s+(?!longer\b|more\s+than\b|less\b|later\b|sooner\b|fewer\b|one\b|other\b)"
    r"(?:further\s+|more\s+|new\s+)?([a-z]+)",
    re.IGNORECASE,
)


def describes_recording(text: str) -> bool:
    """The outcome describes a record being kept, not a result for the resident."""
    return bool(_RECORDING_OUTCOME.search(text))


def no_goals(text: str) -> set[str]:
    """The things an outcome says there will be none of ("falls" in "has no further falls")."""
    return {_stem(m.group(1).casefold()) for m in _NO_GOAL.finditer(text)}


def repeats_outcome(text: str, kept: list[str]) -> bool:
    """Says the same as an outcome already kept: more than 60% of its words, or the same
    "no <thing>" goal ("no falls" twice)."""
    goals = no_goals(text)
    return any(
        word_overlap(text, other) > SAME_NEED_OVERLAP or (goals and goals & no_goals(other))
        for other in kept
    )


def paperwork_only(text: str) -> bool:
    """Every sentence is an instruction to write down or file something that happened ("Staff
    should document that she was weighed and file this with the chart") - no care in it."""
    sentences = [s for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    return bool(sentences) and all(
        _PAPERWORK_LEAD.search(s) and _PAPERWORK.search(s) for s in sentences
    )


def strip_reason(text: str) -> str:
    """The action without its reason clause: "offer the diet as advised because she was
    coughing, and sit her upright" -> "offer the diet as advised, and sit her upright"."""

    def cut(m: re.Match[str]) -> str:
        return "," if m.group(2) and not m.group(1) else " "

    out = _REASON_CLAUSE.sub(cut, text)
    out = re.sub(r"\s+([,.;])", r"\1", " ".join(out.split()))
    out = re.sub(r",(?=[.;])", "", out).strip(" ,")
    return out[:1].upper() + out[1:] if out else text
# OPENCARE-79: equipment as a status ("Staff should arrange a sensor mat as requested"), not how it
# is used. Only when the action itself is arranging or ordering - "help Cathy get dressed as
# agreed" is care.
_EQUIPMENT_STATUS = re.compile(
    r"^\s*(?:staff\s+should\s+)?(?:arrange|request|order|obtain)\b[^.]*"
    r"\bas\s+(?:requested|discussed|agreed)\s*[.!]?\s*$",
    re.IGNORECASE,
)
MAX_MISSING_WORDS = 6

# Function words and words every care plan uses - never "details".
_STOPWORDS = {
    "about", "above", "after", "again", "also", "always", "another", "any", "anything", "around",
    "because", "been", "before", "being", "both", "come", "could", "does", "doing", "done",
    "down", "during", "each", "either", "else", "enough", "even", "ever", "every", "from",
    "further", "give", "given", "going", "have", "having", "help", "helped", "helping", "here",
    "herself", "himself", "into", "itself", "just", "keep", "kept", "like", "make", "many",
    "more", "most", "much", "must", "need", "needed", "needs", "never", "none", "only", "other",
    "over", "own", "same", "should", "since", "some", "something", "still", "such", "than",
    "that", "their", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "upon", "very", "want", "well", "were", "what", "when",
    "where", "which", "while", "will", "with", "within", "without", "would", "your", "yes",
    "currently", "resident", "residents", "staff", "member", "members", "carer", "carers",
    "require", "required", "requires", "requiring", "may", "might", "able", "unable", "time",
    "times", "note", "notes", "plan", "care", "cared", "caring", "person",
}

# Generic care-plan vocabulary a style example may legitimately lend to the
# wording (tone, not facts). Anything outside this list that only an example
# contains is treated as a borrowed detail.
_STYLE_VOCAB = {
    "support", "supported", "supporting", "supportive", "encourage", "encouraged",
    "encouraging", "encouragement", "assist", "assisted", "assisting", "assistance", "promote",
    "promoted", "promoting", "ensure", "ensured", "ensuring", "maintain", "maintained",
    "maintaining", "monitor", "monitored", "monitoring", "check", "checked", "checking",
    "checks", "offer", "offered", "offering", "prompt", "prompted", "prompting", "prompts",
    "remind", "reminded", "reminding", "independence", "independent", "independently",
    "dignity", "dignified", "respect", "respected", "respectful", "privacy", "private",
    "choice", "choices", "choose", "preference", "preferences", "preferred", "prefer",
    "comfortable", "comfort", "comfortably", "safe", "safely", "safety", "regular",
    "regularly", "daily", "routine", "routines", "wellbeing", "appropriate", "appropriately",
    "discreet", "discreetly", "gently", "gentle", "patient", "patiently", "reassure",
    "reassurance", "reassured", "explain", "explained", "explaining", "involve", "involved",
    "involving", "continue", "continued", "continues", "continuing", "ongoing", "aware",
    "awareness", "observe", "observed", "observing", "record", "recorded", "recording",
    "report", "reported", "reporting", "concern", "concerns", "change", "changes", "changed",
    "needs", "ability", "abilities", "level", "levels", "manner", "approach", "consistent",
    "consistently", "possible", "necessary", "where", "whenever", "usual", "usually", "good",
    "best", "relevant", "any", "updated", "update", "review", "reviewed", "reviewing",
    "plan", "outcome", "outcomes", "action", "actions", "achieve", "achieved", "goal",
    "goals", "improve", "improved", "improving", "reduce", "reduced", "reducing", "risk",
    "risks", "minimise", "minimised", "promptly", "timely", "senior", "team",
    "shift", "shifts", "throughout", "fully", "experience", "experiences", "feel", "feels",
    "feeling", "remain", "remains", "remaining", "happy", "content", "settled", "calm",
    "confident", "confidence", "listen", "listened", "listening", "talk", "talking",
    "communicate", "communication", "manage", "managed", "managing", "able", "allow",
    "allowed", "allowing", "provide", "provided", "providing", "make", "sure", "full",
}


def _stem(word: str) -> str:
    w = word.casefold().replace("’", "'")
    if w.endswith("'s"):
        w = w[:-2]
    w = w.strip("'-")
    for suffix in ("ing", "ed", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            w = w[: -len(suffix)]
            break
    if w.endswith("e") and len(w) >= 5:
        w = w[:-1]
    return w


_STOP_STEMS = {_stem(w) for w in _STOPWORDS}
_STYLE_STEMS = {_stem(w) for w in _STYLE_VOCAB}


def stems(texts: list[str]) -> set[str]:
    return {_stem(w) for text in texts for w in _WORD.findall(text)}


def stem(word: str) -> str:
    return _stem(word)


def is_content_stem(s: str) -> bool:
    """Worth searching on: not a function word or a word every note uses."""
    return len(s) >= 3 and s not in _STOP_STEMS


def content_stems(texts: list[str]) -> set[str]:
    """Stems that say what something is about - no function words or generic
    care-plan vocabulary ("support", "staff", "ensure")."""
    return {
        s for s in stems(texts) if len(s) >= 4 and s not in _STOP_STEMS and s not in _STYLE_STEMS
    }


def _key_terms(text: str, ignore: set[str]) -> dict[str, str]:
    """Content words of a source: stem -> the word as written."""
    terms: dict[str, str] = {}
    for word in _WORD.findall(text):
        stem = _stem(word)
        if len(stem) >= 4 and stem not in _STOP_STEMS and stem not in ignore:
            terms.setdefault(stem, word.strip("'’-").lower())
    return terms


@dataclass
class ValidationResult:
    risks: list[RiskFlag]
    not_used: list[NotUsedItem]
    cards: list[ActionCard]
    invalid_reference_count: int
    dropped_action_count: int
    backfilled_count: int
    unsupported_count: int
    copied_count: int = 0
    missing_detail_count: int = 0
    patterns: list[Pattern] = field(default_factory=list)
    # Every problem in plain words, for the care plan agent to fix on revision.
    problems: list[str] = field(default_factory=list)
    # OPENCARE-114: records (notes, chart lines) no action covers - shown to staff as notices,
    # never pasted in as actions. Uncovered flagged answers stay in `risks` ("Not covered").
    not_covered: list[str] = field(default_factory=list)
    # OPENCARE-113: earlier care plan lines that describe the resident, not care - listed for staff.
    not_actions: list[str] = field(default_factory=list)
    # Earlier care plan lines that are only paperwork: not carried into the plan - staff are told.
    not_carried: list[str] = field(default_factory=list)

    @property
    def problem_score(self) -> int:
        """Lower is better - used to keep the best of the draft and its revisions."""
        return (
            3 * self.invalid_reference_count
            + 3 * self.dropped_action_count
            + 3 * self.unsupported_count
            + 3 * self.copied_count
            + 2 * self.backfilled_count
            + self.missing_detail_count
        )


def _number_key(number: str) -> str:
    return number.lstrip("0") or "0"


def _known_numbers(sources: list[str]) -> set[str]:
    """Whole numbers in the sources, plus the parts of dates/times/fractions
    ("02/03/2025" -> 2, 3, 2025) - never digits inside a longer number."""
    known: set[str] = set()
    for number in _NUMBER.findall(" ".join(sources)):
        known.add(_number_key(number))
        known.update(_number_key(part) for part in re.split(r"[.:/]", number))
    return known


def unsupported_numbers(text: str, sources: list[str]) -> list[str]:
    known = _known_numbers(sources)

    def supported(number: str) -> bool:
        parts = [_number_key(p) for p in re.split(r"[.:/]", number)]
        return _number_key(number) in known or all(p in known for p in parts)

    return [n for n in _NUMBER.findall(text) if not supported(n)]


def unsupported_names(text: str, sources: list[str]) -> list[str]:
    """Capitalised words mid-sentence (names, places, services, days) that
    none of the sources contain."""
    known = {
        w.casefold().replace("’", "'").removesuffix("'s")
        for w in _WORD.findall(" ".join(sources))
    }
    found: list[str] = []
    for sentence in _SENTENCE.split(text):
        for word in _WORD.findall(sentence)[1:]:  # the first word is capitalised anyway
            base = word.replace("’", "'").removesuffix("'s").strip("'-")
            if base[:1].isupper() and base != "I" and base.casefold() not in known:
                found.append(base)
    return found


def borrowed_terms(text: str, *, source_stems: set[str], example_stems: set[str]) -> list[str]:
    """Details in `text` that only a style example could have supplied."""
    found = [m.group(0) for m in _PLACEHOLDER.finditer(text)]
    if not example_stems:
        return found
    for word in _WORD.findall(text):
        stem = _stem(word)
        if (
            len(stem) >= 4
            and stem not in _STOP_STEMS
            and stem not in _STYLE_STEMS
            and stem in example_stems
            and stem not in source_stems
        ):
            found.append(word)
    return found


def missing_details(
    text: str, cited: list[EvidenceItem], *, check_ids: set[str], ignore: set[str]
) -> list[MissingDetail]:
    """For each cited source in check_ids: its content words the text lacks."""
    have = stems([text])
    missing: list[MissingDetail] = []
    for item in cited:
        if item.id not in check_ids:
            continue
        terms = _key_terms(item.text, ignore) or _key_terms(item.label, ignore)
        words = [word for stem, word in terms.items() if stem not in have]
        if words:
            missing.append(MissingDetail(evidence_id=item.id, words=words[:MAX_MISSING_WORDS]))
    return missing


def confidence_for(
    evidence_ids: list[str], by_id: dict[str, EvidenceItem], warnings: list[str]
) -> Confidence:
    if warnings:
        return Confidence.LOW
    source_types = {by_id[i].source_type for i in evidence_ids if i in by_id}
    # Backed by two different sources (e.g. assessment + review note) = high.
    return Confidence.HIGH if len(source_types) >= 2 else Confidence.MEDIUM


def detail_warnings(text: str, sources: list[str]) -> list[str]:
    warnings: list[str] = []
    if unsupported_numbers(text, sources):
        warnings.append(UNSUPPORTED_NUMBER_WARNING)
    if unsupported_names(text, sources):
        warnings.append(UNSUPPORTED_NAME_WARNING)
    return warnings


def linked_charts(evidence: list[EvidenceItem]) -> list[list[str]]:
    """OPENCARE-99: one id group per current chart linked to this pathway (primary, not history):
    its summary line first, then its entries newest first - an action must cite one of them."""
    charts: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        if ((item.is_observation or item.is_chart_summary) and item.linked_to_pathway
                and not item.supporting and not item.historical):
            charts.setdefault(item.label, []).append(item)
    groups = []
    for items in charts.values():
        summaries = [i.id for i in items if i.is_chart_summary]
        entries = [i.id for i in items if i.is_observation]
        groups.append(summaries + entries[::-1])
    return groups


def _dedupe(ids: list[str]) -> list[str]:
    return list(dict.fromkeys(i.strip() for i in ids if i and i.strip()))


def validate_patterns(
    patterns: list[PatternOut], evidence: list[EvidenceItem], *, resident_name: str | None
) -> list[Pattern]:
    """Keeps a pattern only if it cites at least two real items and names no
    number or name those items don't contain - otherwise it is dropped."""
    by_id = {e.id: e for e in evidence}
    kept: list[Pattern] = []
    for pattern in patterns:
        ids = [i for i in _dedupe(pattern.evidence_ids) if i in by_id]
        summary = " ".join(pattern.summary.split())
        if len(ids) < 2 or not summary:
            continue
        sources = [by_id[i].text for i in ids] + [by_id[i].label for i in ids]
        sources += [resident_name] if resident_name else []
        if detail_warnings(summary, sources):
            continue
        kept.append(Pattern(summary=summary, evidence_ids=ids))
    return kept


def _required_note(item: EvidenceItem, draft_kind: str | None) -> bool:
    if item.historical:
        # History explains and personalises; a need from the past only becomes
        # a requirement when current evidence raises it again.
        return False
    if draft_kind == "new_requirement":
        return True
    # A note written against this pathway that the model didn't classify still
    # counts - when unsure, show it. A resident-level note only counts when
    # the model found a requirement in it.
    return item.linked_to_pathway and draft_kind is None


def validate_draft(
    evidence: list[EvidenceItem],
    draft: DraftPlanOut,
    *,
    resident_name: str | None,
    pathway: str | None = None,
    examples: list[str] | None = None,
    resident_sources: list[str] | None = None,
    with_outcomes: bool = True,
    required_groups: list[list[str]] | None = None,
) -> ValidationResult:
    """`evidence` is what the model was shown. `resident_sources` is all of the
    resident's text (including notes it wasn't shown), used only to decide
    whether a detail could have come from somewhere other than an example.
    `required_groups`: records a reader agent found a need in (read.py)."""
    by_id = {e.id: e for e in evidence}
    invalid = 0
    problems: list[str] = []

    def bad_ref(what: str, ref: str) -> None:
        nonlocal invalid
        invalid += 1
        problems.append(f"{what} '{ref}' is not a valid evidence id for that use.")

    def is_kind(evidence_id: str, source_type: SourceType, prefix: str) -> bool:
        item = by_id.get(evidence_id)
        return item is not None and item.source_type == source_type and item.id.startswith(prefix)

    def is_note(evidence_id: str) -> bool:
        item = by_id.get(evidence_id)
        return item is not None and item.is_note

    def is_current_note(evidence_id: str) -> bool:
        item = by_id.get(evidence_id)
        return item is not None and item.is_note and not item.historical

    # --- 1. risks: only real assessment answers ---
    risks: list[RiskFlag] = []
    risk_seen: set[str] = set()
    for risk in draft.risks:
        rule = by_id[risk.evidence_id].risk_rule if risk.evidence_id in by_id else None
        if rule in ("no_risk", "info"):
            # OPENCARE-118: the question's rule says this answer is not a risk ("Oriented: Yes",
            # "Hand Rails: Yes", a comment box) - not counted, so no filler action is asked for.
            risk_seen.add(risk.evidence_id)
            continue
        if is_kind(risk.evidence_id, SourceType.ASSESSMENT, "A") and risk.evidence_id not in {
            r.evidence_id for r in risks
        }:
            risks.append(RiskFlag(evidence_id=risk.evidence_id, reason=risk.reason.strip()))
        elif risk.evidence_id not in risk_seen:
            bad_ref("Risk", risk.evidence_id)
        risk_seen.add(risk.evidence_id)
    # OPENCARE-98: an answer the template flags as a risk is a risk whether or not the model listed
    # it - so it must be covered by an action, or it is shown as not covered.
    for item in evidence:
        if (item.risk_flagged and item.source_type == SourceType.ASSESSMENT
                and item.id.startswith("A") and item.id not in {r.evidence_id for r in risks}):
            risks.append(
                RiskFlag(evidence_id=item.id, reason=f"Flagged in the assessment: {item.label}")
            )
    risk_ids = {r.evidence_id for r in risks}

    # --- note classification ---
    note_kinds: dict[str, str] = {}
    for cls in draft.note_classes:
        if is_note(cls.evidence_id):
            note_kinds[cls.evidence_id] = cls.kind
        else:
            bad_ref("Note class", cls.evidence_id)

    # --- superseded previous actions: must name a real, current note ---
    not_used: list[NotUsedItem] = []
    for item in draft.not_used:
        if is_kind(item.evidence_id, SourceType.PREVIOUS_CARE_PLAN, "P") and is_current_note(
            item.superseded_by
        ):
            if item.evidence_id not in {n.evidence_id for n in not_used}:
                not_used.append(
                    NotUsedItem(
                        evidence_id=item.evidence_id,
                        superseded_by=item.superseded_by,
                        reason=item.reason.strip(),
                    )
                )
        else:
            bad_ref("not_used", f"{item.evidence_id} superseded by {item.superseded_by}")
    # OPENCARE-113: lines the current score or diet level makes wrong are not used, whatever the
    # model said - decided in code from the records (superseded.py).
    rules = rule_not_used(evidence, pathway=pathway, resident_name=resident_name)
    forbidden = {r.evidence_id: r.forbidden for r in rules if r.forbidden is not None}
    for rule in rules:
        if rule.evidence_id not in {n.evidence_id for n in not_used}:
            not_used.append(NotUsedItem(evidence_id=rule.evidence_id,
                                        superseded_by=rule.superseded_by, reason=rule.reason))
    superseded = {n.evidence_id for n in not_used}
    # OPENCARE-113: an earlier line that only describes the resident is not care - never required
    # as an action, and listed for staff.
    states = state_lines(evidence, resident_name) - superseded

    # --- carried over as written: current previous care plan items only ---
    carried: set[str] = set()
    for ref in _dedupe(draft.carry_over):
        if ref in states:
            continue  # OPENCARE-113: listed for staff, never shown as an action
        if is_kind(ref, SourceType.PREVIOUS_CARE_PLAN, "P") and ref not in superseded:
            carried.add(ref)
        else:
            bad_ref("carry_over", ref)

    # --- what must be covered ---
    required: set[str] = set(risk_ids)
    not_carried: list[str] = []
    not_actions: list[str] = []
    for source in evidence:
        if source.source_type == SourceType.PREVIOUS_CARE_PLAN and source.id.startswith("P"):
            if source.id in superseded:
                continue
            if source.id in states:
                if source.text not in not_actions:
                    not_actions.append(source.text)
                continue
            if paperwork_only(source.text):
                # Paperwork is not care (OPENCARE-78): not carried forward word for word.
                if source.text not in not_carried:
                    not_carried.append(source.text)
                continue
            required.add(source.id)
        elif source.is_note and _required_note(source, note_kinds.get(source.id)):
            required.add(source.id)
        elif source.source_type == SourceType.CARE_STANDARD:
            # OPENCARE-99: each step of the care standard the recorded score calls for.
            required.add(source.id)

    # --- actions ---
    context_sources = [s for s in (resident_name, pathway) if s]
    ignore_stems = stems(context_sources)
    all_sources = (resident_sources or [e.text for e in evidence]) + [
        e.label for e in evidence
    ] + context_sources
    source_stems = stems(all_sources)
    example_stems = stems(examples or [])

    # OPENCARE-115: in a Choking plan with thickened fluids, the words every drink action carries.
    thick = thickened_level(evidence, pathway)
    # OPENCARE-114: notes and chart lines - an action must not be one of them copied.
    records = [
        e for e in evidence
        if e.is_note or e.source_type in (SourceType.OBSERVATION, SourceType.OBSERVATION_SUMMARY)
    ]
    cards: list[ActionCard] = []
    dropped = copied = unsupported_count = missing_count = 0
    for index, action in enumerate(draft.actions):
        cited = _dedupe(action.evidence_ids)
        ids = [i for i in cited if i in by_id and i not in superseded]
        # OPENCARE-115/116: a care standard step brings the score, level or reading it answers,
        # so staff see it under Sources ("TPR & BP 17/11/2026: standing 108/64").
        for ref in list(ids):
            backing = by_id[ref].backed_by
            if backing and backing in by_id and backing not in ids:
                ids.append(backing)
        text = action.text.strip()
        for ref in cited:
            if ref not in ids:
                bad_ref(f"Action \"{text[:60]}\" cites", ref)
        if not ids or not text:
            dropped += 1
            problems.append(f"Action \"{text[:80]}\" has no valid evidence and was dropped.")
            continue
        if all(by_id[i].supporting for i in ids):
            # Supporting charts are context for this pathway (care team rules):
            # they back an action, never drive one on their own.
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" cites only supporting readings ({', '.join(ids)}) and "
                "was dropped. Cite the assessment, care plan, note or primary chart it responds "
                "to, or remove it."
            )
            continue
        if _RECORD_REFERENCE.search(text):
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" refers to the records instead of describing the care "
                "and was dropped. Write the care itself, or put a P item that states no care "
                "in carry_over."
            )
            continue
        if _NOT_CARE.search(text):
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" gives a reason or only records paperwork, and was "
                "dropped. Write only what staff do for the resident (and how often, if the "
                "evidence says); the reason stays in the cited evidence."
            )
            continue
        if _EQUIPMENT_STATUS.search(text):
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" says only that equipment is arranged, and was dropped. "
                "Say how staff use the equipment for the resident, using only details the "
                "evidence gives."
            )
            continue
        if thick and _DRINK.search(text) and thick not in text.casefold():
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" is about drinks but does not say they are {thick} "
                f"(the level in the records), and was dropped. Every drink action must name the "
                "level."
            )
            continue
        repeated = _repeats_not_used(text, not_used, by_id, forbidden, resident_name)
        if repeated is not None:
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" repeats [{repeated.evidence_id}], which no longer applies "
                f"({repeated.reason}), and was dropped. Write only the care that applies now, "
                "without that earlier wording."
            )
            continue
        not_instruction = not_an_instruction(text, records)
        if not_instruction is None and describes_state(text, resident_name):
            # OPENCARE-113: "Willie's skin is intact." says how he is, not what staff do.
            not_instruction = "describes the resident, not what staff do"
        if not_instruction:
            dropped += 1
            problems.append(
                f"Action \"{text[:80]}\" {not_instruction}, and was dropped. Rewrite it as one "
                "instruction for staff (what to do, and how often if the evidence says), in plain "
                "English. No questions, answers, scores, reasons or chart readings; they stay in "
                "the cited evidence."
            )
            continue
        borrowed = borrowed_terms(text, source_stems=source_stems, example_stems=example_stems)
        if borrowed:
            copied += 1
            problems.append(
                f"Action \"{text[:80]}\" uses details that only a STYLE EXAMPLE contains "
                f"({', '.join(borrowed)}) and was dropped."
            )
            continue

        if _REASON.search(text):
            # Kept: dropping it could lose the care (e.g. a texture-modified diet). The reason is
            # taken out here, and the agent is told so it writes it that way.
            problems.append(
                f"Action \"{text[:80]}\" explains why. Keep the care and take the reason out; "
                "the reason stays in the cited evidence."
            )
            text = strip_reason(text)

        status = _STATUS_TAIL.search(text)
        if status:
            # Kept without the status, as a reason is: the care before it is still care.
            problems.append(
                f"Action \"{text[:80]}\" adds how things stand after a semicolon (something "
                "that has been requested, started or reported). Write only what staff do, one "
                "action per need; the status stays in the cited evidence."
            )
            rest = _SENTENCE_END.split(text[status.end():], maxsplit=1)
            text = (text[:status.start()].rstrip(" ,;") + "."
                    + (" " + rest[1] if len(rest) > 1 else "")).strip()

        if _REACTIVE.search(text):
            # OPENCARE-98: "If Agnes says she feels light-headed, staff should..." waits for the
            # problem. Kept (it is still care), and the agent is asked for the preventive step.
            problems.append(
                f"Action \"{text[:80]}\" waits for the problem to happen. Write what staff do "
                "routinely to prevent it, using only what the evidence gives."
            )

        sources = [by_id[i].text for i in ids] + [by_id[i].label for i in ids] + context_sources
        warnings = detail_warnings(text, sources)
        unsupported_count += len(warnings)
        problems += [f"Action \"{text[:80]}\": {w}" for w in warnings]

        outcome: Suggestion | None = None
        # Session drafts are actions only; outcomes come from the outcome step.
        outcome_text = action.outcome_text.strip() if with_outcomes else ""
        borrowed = borrowed_terms(
            outcome_text, source_stems=source_stems, example_stems=example_stems
        ) if outcome_text else []
        if borrowed:
            copied += 1
            problems.append(
                f"Outcome \"{outcome_text[:80]}\" uses details only a STYLE EXAMPLE contains "
                f"({', '.join(borrowed)}) and was dropped."
            )
            outcome_text = ""
        if outcome_text:
            cited_outcome = _dedupe(action.outcome_evidence_ids)
            outcome_ids = [i for i in cited_outcome if i in by_id and i not in superseded]
            for ref in cited_outcome:
                if ref not in outcome_ids:
                    bad_ref(f"Outcome \"{outcome_text[:60]}\" cites", ref)
            # An outcome follows from its action, so the action's evidence
            # backs it when the model cited nothing usable for it.
            outcome_ids = outcome_ids or ids
            outcome_sources = (
                [by_id[i].text for i in set(outcome_ids) | set(ids)]
                + [by_id[i].label for i in set(outcome_ids) | set(ids)]
                + [text]
                + context_sources
            )
            outcome_warnings = detail_warnings(outcome_text, outcome_sources)
            unsupported_count += len(outcome_warnings)
            problems += [f"Outcome \"{outcome_text[:80]}\": {w}" for w in outcome_warnings]
            outcome = Suggestion(
                text=outcome_text,
                original_text=outcome_text,
                evidence_ids=outcome_ids,
                warnings=outcome_warnings,
                confidence=confidence_for(outcome_ids, by_id, outcome_warnings),
            )

        missing = missing_details(
            text, [by_id[i] for i in ids], check_ids=required | states, ignore=ignore_stems
        )
        missing_count += len(missing)
        problems += [
            f"Action \"{text[:80]}\" leaves out details of [{m.evidence_id}]: {', '.join(m.words)}"
            for m in missing
        ]
        cards.append(
            ActionCard(
                id="",  # numbered below
                text=text,
                original_text=text,
                evidence_ids=ids,
                risk_evidence_ids=[i for i in ids if i in risk_ids],
                warnings=warnings,
                possibly_missing=missing,
                confidence=confidence_for(ids, by_id, warnings),
                outcome=outcome,
                draft_index=index,
            )
        )

    # OPENCARE-98: one action per need - two actions saying mostly the same thing ("take her to
    # the bathroom at night" twice) are the same need written twice.
    for first, second in overlapping_actions([card.text for card in cards]):
        problems.append(
            f"Actions \"{cards[first].text[:60]}\" and \"{cards[second].text[:60]}\" cover the "
            "same need. Write one action for it, citing the evidence of both."
        )

    # --- 2. completeness: nothing missed, nothing skipped ---
    covered = {i for card in cards for i in card.evidence_ids}
    backfilled = 0
    not_covered: list[str] = []
    # The same line from more than one earlier care plan is shown once, citing each of them.
    # Only earlier plan lines: other records with the same words ("Yes" to two questions) are
    # different needs.
    shown: dict[str, ActionCard] = {}

    def plan_line(source: EvidenceItem) -> str | None:
        if source.source_type != SourceType.PREVIOUS_CARE_PLAN:
            return None
        return " ".join(source.text.casefold().split())

    for source in evidence:  # evidence order keeps backfilled cards in a sensible order
        if source.id not in required or source.id in covered:
            continue
        line = plan_line(source)
        same = shown.get(line) if line else None
        if same is not None:
            same.evidence_ids.append(source.id)
            continue
        if source.id in carried:
            # The model judged it states no care it could write without
            # guessing: shown to staff exactly as written, never reworded.
            cards.append(
                ActionCard(
                    id="",
                    origin=CardOrigin.SOURCE_TEXT,
                    text=source.text,
                    original_text=source.text,
                    evidence_ids=[source.id],
                    confidence=Confidence.MEDIUM,
                    verification="source_text",
                )
            )
            if line:
                shown[line] = cards[-1]
            continue
        backfilled += 1
        problems.append(
            f"[{source.id}] must be covered by an action but no action cites it: {source.text}"
        )
        if source.id in risk_ids:
            # OPENCARE-114: a flagged answer no action covers is never pasted in as an action
            # ("Sex: F", "...?: Yes"); it stays in the risks, so staff see it under "Not covered".
            continue
        if source.is_note:
            # OPENCARE-114: nor is a note - it is listed for staff to check instead.
            not_covered.append(_not_covered_line(source))
            continue
        text = source.text
        cards.append(
            ActionCard(
                id="",
                origin=CardOrigin.SOURCE_TEXT,
                text=text,
                original_text=text,
                evidence_ids=[source.id],
                confidence=Confidence.MEDIUM,
                verification="source_text",
            )
        )
        if line:
            shown[line] = cards[-1]

    # Needs a reader agent found in records the care plan agent didn't start
    # with: an action must cite one of each record's ids (the same record on
    # several dates counts once), or it is shown word for word.
    # OPENCARE-99: so must each current chart linked to this pathway (its summary line, else its
    # newest entry, is shown) - the Food chart of a MUST plan was read but never acted on.
    for group in [*(required_groups or []), *linked_charts(evidence)]:
        head = by_id.get(group[0])
        if head is None or group[0] in required or any(i in covered for i in group):
            continue
        backfilled += 1
        problems.append(
            f"[{head.id}] raises a need for this pathway but no action cites it: {head.text}"
        )
        # OPENCARE-114: a chart line or record ("2 entries on 16/11/2026 ...") is never pasted in
        # as an action; it is listed for staff to check instead.
        not_covered.append(_not_covered_line(head))

    for index, card in enumerate(cards, start=1):
        card.id = f"S{index}"

    return ValidationResult(
        risks=risks,
        not_used=not_used,
        cards=cards,
        invalid_reference_count=invalid,
        dropped_action_count=dropped,
        backfilled_count=backfilled,
        unsupported_count=unsupported_count,
        copied_count=copied,
        missing_detail_count=missing_count,
        patterns=validate_patterns(draft.patterns, evidence, resident_name=resident_name),
        problems=problems,
        not_carried=not_carried,
        not_covered=not_covered,
        not_actions=not_actions,
    )


def _not_covered_line(source: EvidenceItem) -> str:
    """OPENCARE-114: a record no action covers, for staff to check - never an action itself."""
    label = f"{source.label}: " if source.label and source.label != source.text else ""
    return (
        "Open Care Buddy did not write an action for this record. Check it and add one if "
        f"it is needed: \"{label}{source.text}\""
    )
