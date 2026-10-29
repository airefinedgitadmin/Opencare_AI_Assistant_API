"""What agent memory keeps: references to the resident's records, never their text.

A drafting session (and a saved plan's trace) needs the evidence on every call,
but the records belong to the backend's database. So agent memory keeps, per
evidence item, only its id, kind, backend record id, date, flags and a keyed
fingerprint of its content. The backend sends the evidence pack again with each
call; `rehydrate` matches its records to the stored ids by fingerprint, and the
text lives in this process's memory only for that request. A record that has
changed or gone since is shown as changed, never with stale text.

What can't be a reference is the AI's own wording and staff's edits
(suggestions, statements, reasons, search terms): that is kept encrypted
(memory/sealing.py). Nothing else about the resident is stored - not even the
name, which comes back with the evidence pack.
"""

from datetime import datetime
from typing import Any

from careplan_ai.agents.care_plan.evidence import build_evidence
from careplan_ai.agents.care_plan.schemas import (
    CarePlanSession,
    CarePlanTrace,
    EvidenceItem,
    SourceType,
)
from careplan_ai.api.schemas.agent import StartSessionRequest
from careplan_ai.api.schemas.base import CamelModel
from careplan_ai.memory.sealing import fingerprint, seal, unseal

CHANGED_LABEL = "Record changed or no longer available"
CHANGED_TEXT = "This record has changed or is no longer available since the AI read it."

# Session fields holding the AI's wording or reasons - kept encrypted.
_SEALED_SESSION_FIELDS = ("cards", "risks", "patterns", "not_used", "screened", "style_examples")


class EvidenceRef(CamelModel):
    id: str
    source_type: SourceType
    source_ref: str | None = None
    date: str | None = None
    linked_to_pathway: bool = True
    historical: bool = False
    supporting: bool = False
    fingerprint: str


def evidence_fingerprint(item: EvidenceItem) -> str:
    return fingerprint(item.source_type.value, item.source_ref, item.label, item.text)


def to_refs(items: list[EvidenceItem]) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            id=e.id,
            source_type=e.source_type,
            source_ref=e.source_ref,
            date=e.date,
            linked_to_pathway=e.linked_to_pathway,
            historical=e.historical,
            supporting=e.supporting,
            fingerprint=e.fingerprint or evidence_fingerprint(e),
        )
        for e in items
    ]


def rehydrate(refs: list[EvidenceRef], request: StartSessionRequest | None) -> list[EvidenceItem]:
    """The stored ids, with their text from the evidence the backend just sent."""
    pool: dict[str, list[EvidenceItem]] = {}
    for item in build_evidence(request) if request is not None else []:
        pool.setdefault(evidence_fingerprint(item), []).append(item)
    items: list[EvidenceItem] = []
    for ref in refs:
        matches = pool.get(ref.fingerprint)
        if matches:
            items.append(matches.pop(0).model_copy(update={
                "id": ref.id,
                "linked_to_pathway": ref.linked_to_pathway,
                "historical": ref.historical,
                "supporting": ref.supporting,
                "fingerprint": ref.fingerprint,
            }))
        else:
            items.append(EvidenceItem(
                id=ref.id, source_type=ref.source_type, source_ref=ref.source_ref,
                label=CHANGED_LABEL, text=CHANGED_TEXT, date=ref.date,
                linked_to_pathway=ref.linked_to_pathway, historical=ref.historical,
                supporting=ref.supporting, changed=True, fingerprint=ref.fingerprint,
            ))
    return items


# --- sessions ---------------------------------------------------------------


def session_to_stored(session: CarePlanSession) -> dict[str, Any]:
    data = session.model_dump(mode="json")
    secret = {field: data.pop(field) for field in _SEALED_SESSION_FIELDS}
    secret["searches"] = data["generation"].pop("searches")
    data.pop("evidence")
    data["resident_name"] = None
    return {
        "session": data,
        "evidence": [r.model_dump(mode="json") for r in to_refs(session.evidence)],
        "sealed": seal(secret),
    }


def session_from_stored(
    raw: dict[str, Any], request: StartSessionRequest | None
) -> CarePlanSession:
    if "sealed" not in raw:  # written before references (legacy)
        return CarePlanSession.model_validate(raw)
    data = dict(raw["session"])
    secret = unseal(raw["sealed"])
    assert isinstance(secret, dict)
    data["generation"] = {**data["generation"], "searches": secret.pop("searches", [])}
    data.update(secret)
    data["evidence"] = rehydrate(
        [EvidenceRef.model_validate(r) for r in raw["evidence"]], request
    )
    data["resident_name"] = request.resident_name if request is not None else None
    return CarePlanSession.model_validate(data)


# --- traces of saved care plans ------------------------------------------------


def trace_to_stored(trace: CarePlanTrace) -> dict[str, Any]:
    cited: dict[str, EvidenceItem] = {}
    for statement in trace.statements:
        cited.update({e.id: e for e in statement.evidence})
    cited.update({e.id: e for e in trace.not_used_evidence})
    data = trace.model_dump(mode="json")
    statements = [
        {**s, "evidence": [e["id"] for e in s["evidence"]]} for s in data.pop("statements")
    ]
    secret = {
        "statements": statements,
        "not_used": data.pop("not_used"),
        "not_used_ids": [e["id"] for e in data.pop("not_used_evidence")],
    }
    return {
        "trace": data,
        "evidence": [r.model_dump(mode="json") for r in to_refs(list(cited.values()))],
        "sealed": seal(secret),
    }


def trace_from_stored(raw: dict[str, Any], request: StartSessionRequest | None) -> CarePlanTrace:
    if "sealed" not in raw:  # legacy
        return CarePlanTrace.model_validate(raw)
    secret = unseal(raw["sealed"])
    assert isinstance(secret, dict)
    by_id = {
        e.id: e
        for e in rehydrate([EvidenceRef.model_validate(r) for r in raw["evidence"]], request)
    }
    data = dict(raw["trace"])
    data["statements"] = [
        {**s, "evidence": [by_id[i] for i in s["evidence"] if i in by_id]}
        for s in secret["statements"]
    ]
    data["not_used"] = secret["not_used"]
    data["not_used_evidence"] = [by_id[i] for i in secret["not_used_ids"] if i in by_id]
    return CarePlanTrace.model_validate(data)


def subject_of(raw: dict[str, Any]) -> tuple[int, datetime | None]:
    """Whose evidence a stored session/trace needs, and as of when - readable
    without the key, so the backend knows what to send."""
    body = raw.get("session") or raw.get("trace") or raw
    as_of = body.get("evidence_as_of") or body.get("created_at")
    if isinstance(as_of, str):
        as_of = datetime.fromisoformat(as_of)
    return int(body["assessment_template_user_id"]), as_of
