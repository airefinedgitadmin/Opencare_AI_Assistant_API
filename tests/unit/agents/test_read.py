import pytest

from careplan_ai.agents.care_plan.read import read_rest
from careplan_ai.agents.care_plan.schemas import (
    EvidenceItem,
    ReadOut,
    RelevantRecordOut,
    SourceType,
)


def _a(i: int) -> EvidenceItem:
    return EvidenceItem(id=f"A{i}", source_type=SourceType.ASSESSMENT, label="Falls",
                        text="Fell twice")


def _c(i: int, text: str, historical: bool = False) -> EvidenceItem:
    return EvidenceItem(id=f"C{i}", source_type=SourceType.CARER_NOTE, label="Carer note",
                        text=text, linked_to_pathway=False, historical=historical,
                        date=f"{i:02d}/10/2026")


@pytest.fixture
def reader() -> list[list[str]]:
    """The record ids each reader call was given, in call order."""
    return []


def _script(monkeypatch: pytest.MonkeyPatch, calls: list[list[str]], *answers: ReadOut) -> None:
    replies = iter(answers)

    async def _read(**kwargs: object) -> ReadOut:
        records: list[EvidenceItem] = kwargs["records"]  # type: ignore[assignment]
        calls.append([e.id for e in records])
        return next(replies)

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.read_records", _read)


async def test_nothing_is_read_when_everything_fits(
    base_env: None, monkeypatch: pytest.MonkeyPatch, reader: list[list[str]]
) -> None:
    _script(monkeypatch, reader)
    evidence = [_a(1), _c(1, "Fell in the lounge.")]

    reading = await read_rest(evidence, evidence, resident_name="Cathy", pathway="Falls")

    assert reader == [] and reading.calls == 0 and reading.unread == []


async def test_every_record_is_accounted_for_and_skipped_ones_are_asked_again(
    base_env: None, monkeypatch: pytest.MonkeyPatch, reader: list[list[str]]
) -> None:
    evidence = [_a(1), _c(1, "Fell in the lounge."), _c(2, "Enjoyed bingo."),
                _c(3, "Found on the floor at night."), _c(4, "Fell last year.", historical=True)]
    _script(
        monkeypatch, reader,
        # First pass skips C3 and answers an id it was never shown (ignored).
        ReadOut(relevant=[RelevantRecordOut(evidence_id="C1", kind="new_requirement",
                                            reason="fell in the lounge"),
                          RelevantRecordOut(evidence_id="C4", kind="new_requirement",
                                            reason="fell last year"),
                          RelevantRecordOut(evidence_id="Z9", kind="status", reason="x")],
                not_relevant=["C2"]),
        ReadOut(relevant=[RelevantRecordOut(evidence_id="C3", kind="update",
                                            reason="found on the floor")],
                not_relevant=[]),
    )

    reading = await read_rest(evidence, [evidence[0]], resident_name="Cathy", pathway="Falls")

    assert reader == [["C1", "C2", "C3", "C4"], ["C3"]]
    assert {s.evidence_id: s.kind for s in reading.screened} == {
        # History never becomes a requirement on its own.
        "C1": "new_requirement", "C2": "not_relevant", "C3": "update", "C4": "status",
    }
    assert reading.required == [["C1"]]
    assert [e.id for e in reading.added] == ["C1", "C3", "C4"]
    assert reading.unread == []


async def test_a_need_in_a_supporting_reading_is_never_required(
    base_env: None, monkeypatch: pytest.MonkeyPatch, reader: list[list[str]]
) -> None:
    supporting = EvidenceItem(id="V1", source_type=SourceType.OBSERVATION, label="TPR",
                              text="Temperature: 38.2", supporting=True)
    evidence = [_a(1), supporting]
    _script(monkeypatch, reader,
            ReadOut(relevant=[RelevantRecordOut(evidence_id="V1", kind="new_requirement",
                                                reason="temperature 38.2")],
                    not_relevant=[]))

    reading = await read_rest(evidence, [evidence[0]], resident_name="Cathy", pathway="Falls")

    assert [(s.evidence_id, s.kind) for s in reading.screened] == [("V1", "status")]
    assert reading.required == []
    assert [e.id for e in reading.added] == ["V1"]  # still given to the care plan agent


async def test_records_still_unread_after_the_second_ask_are_reported(
    base_env: None, monkeypatch: pytest.MonkeyPatch, reader: list[list[str]]
) -> None:
    evidence = [_a(1), _c(1, "Fell in the lounge."), _c(2, "Enjoyed bingo.")]
    _script(monkeypatch, reader,
            ReadOut(relevant=[], not_relevant=["C2"]), ReadOut(relevant=[], not_relevant=[]))

    reading = await read_rest(evidence, [evidence[0]], resident_name="Cathy", pathway="Falls")

    assert reading.unread == ["C1"]
    assert [s.evidence_id for s in reading.screened] == ["C2"]


async def test_large_records_are_split_into_parallel_chunks(
    base_env: None, monkeypatch: pytest.MonkeyPatch, reader: list[list[str]]
) -> None:
    monkeypatch.setenv("CAREPLAN_AI_AGENT_READER_CHUNK_TOKENS", "200")
    from careplan_ai.core.config import get_settings
    get_settings.cache_clear()
    evidence = [_a(1)] + [_c(i, f"Note number {i} about the garden and the weather.")
                          for i in range(1, 41)]

    async def _read(**kwargs: object) -> ReadOut:
        records: list[EvidenceItem] = kwargs["records"]  # type: ignore[assignment]
        reader.append([e.id for e in records])
        return ReadOut(relevant=[], not_relevant=[e.id for e in records])

    monkeypatch.setattr("careplan_ai.agents.care_plan.llm.read_records", _read)
    try:
        reading = await read_rest(evidence, [evidence[0]], resident_name="Cathy",
                                  pathway="Falls")
    finally:
        get_settings.cache_clear()

    assert len(reader) > 1  # several readers ...
    assert sorted(i for chunk in reader for i in chunk) == sorted(e.id for e in evidence[1:])
    assert len(reading.screened) == 40 and reading.unread == []  # ... nothing missed
