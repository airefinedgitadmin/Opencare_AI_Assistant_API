"""Code-level checks on the model's draft - the "no hallucination, nothing
missed" rules from the deck's Guardrails slide, enforced deterministically
rather than by asking another model.

1. References: every cited id must exist and be the right kind of evidence;
   anything else is discarded and counted. An action left with no valid
   evidence is dropped - it can't be traced to a source, so it can't be shown.
2. Completeness: every flagged risk, every previous care plan action that no
   note supersedes, and every note requirement must be covered by at least
   one action. Anything left uncovered is added back as its own card using the
   source's own words - no model involved.
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
        if is_kind(risk.evidence_id, SourceType.ASSESSMENT, "A") and risk.evidence_id not in {
            r.evidence_id for r in risks
        }:
            risks.append(RiskFlag(evidence_id=risk.evidence_id, reason=risk.reason.strip()))
        elif risk.evidence_id not in risk_seen:
            bad_ref("Risk", risk.evidence_id)
        risk_seen.add(risk.evidence_id)
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
    superseded = {n.evidence_id for n in not_used}

    # --- carried over as written: current previous care plan items only ---
    carried: set[str] = set()
    for ref in _dedupe(draft.carry_over):
        if is_kind(ref, SourceType.PREVIOUS_CARE_PLAN, "P") and ref not in superseded:
            carried.add(ref)
        else:
            bad_ref("carry_over", ref)

    # --- what must be covered ---
    required: set[str] = set(risk_ids)
    for source in evidence:
        if source.source_type == SourceType.PREVIOUS_CARE_PLAN and source.id.startswith("P"):
            if source.id not in superseded:
                required.add(source.id)
        elif source.is_note and _required_note(source, note_kinds.get(source.id)):
            required.add(source.id)

    # --- actions ---
    context_sources = [s for s in (resident_name, pathway) if s]
    ignore_stems = stems(context_sources)
    all_sources = (resident_sources or [e.text for e in evidence]) + [
        e.label for e in evidence
    ] + context_sources
    source_stems = stems(all_sources)
    example_stems = stems(examples or [])

    cards: list[ActionCard] = []
    dropped = copied = unsupported_count = missing_count = 0
    for index, action in enumerate(draft.actions):
        cited = _dedupe(action.evidence_ids)
        ids = [i for i in cited if i in by_id and i not in superseded]
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
        borrowed = borrowed_terms(text, source_stems=source_stems, example_stems=example_stems)
        if borrowed:
            copied += 1
            problems.append(
                f"Action \"{text[:80]}\" uses details that only a STYLE EXAMPLE contains "
                f"({', '.join(borrowed)}) and was dropped."
            )
            continue

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
            text, [by_id[i] for i in ids], check_ids=required, ignore=ignore_stems
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

    # --- 2. completeness: nothing missed, nothing skipped ---
    covered = {i for card in cards for i in card.evidence_ids}
    backfilled = 0
    for source in evidence:  # evidence order keeps backfilled cards in a sensible order
        if source.id not in required or source.id in covered:
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
            continue
        backfilled += 1
        problems.append(
            f"[{source.id}] must be covered by an action but no action cites it: {source.text}"
        )
        is_risk = source.id in risk_ids
        text = f"{source.label}: {source.text}" if is_risk else source.text
        cards.append(
            ActionCard(
                id="",
                origin=CardOrigin.SOURCE_TEXT,
                text=text,
                original_text=text,
                evidence_ids=[source.id],
                risk_evidence_ids=[source.id] if is_risk else [],
                warnings=[UNADDRESSED_RISK_WARNING] if is_risk else [],
                confidence=Confidence.MEDIUM,
                verification="source_text",
            )
        )

    # Needs a reader agent found in records the care plan agent didn't start
    # with: an action must cite one of each record's ids (the same record on
    # several dates counts once), or it is shown word for word.
    for group in required_groups or []:
        first = by_id.get(group[0])
        if first is None or group[0] in required or any(i in covered for i in group):
            continue
        backfilled += 1
        problems.append(
            f"[{first.id}] raises a need for this pathway but no action cites it: {first.text}"
        )
        cards.append(
            ActionCard(
                id="",
                origin=CardOrigin.SOURCE_TEXT,
                text=first.text,
                original_text=first.text,
                evidence_ids=[first.id],
                confidence=Confidence.MEDIUM,
                verification="source_text",
            )
        )

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
    )
