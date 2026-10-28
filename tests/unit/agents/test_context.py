import pytest

from careplan_ai.agents.care_plan.context import count_tokens, rank_for_pathway
from careplan_ai.agents.care_plan.packs import build_pack
from careplan_ai.agents.care_plan.prompts import format_grouped
from careplan_ai.agents.care_plan.schemas import EvidenceItem, SourceType
from careplan_ai.agents.care_plan.search import RecordIndex
from careplan_ai.core.exceptions import EvidenceTooLargeError


def _a(i: int, label: str = "Q", text: str = "Yes") -> EvidenceItem:
    return EvidenceItem(id=f"A{i}", source_type=SourceType.ASSESSMENT, label=label, text=text)


def _c(i: int, date: str | None, text: str = "Had a settled night.",
       historical: bool = False) -> EvidenceItem:
    return EvidenceItem(id=f"C{i}", source_type=SourceType.CARER_NOTE, label="Carer note",
                        date=date, text=text, linked_to_pathway=False, historical=historical)


def test_search_finds_the_matching_records_best_first_and_keeps_figures_whole() -> None:
    items = [
        _c(1, None, "Enjoyed the music session."),
        _c(2, None, "Refused to brush her teeth."),
        _c(3, None, "Refused breakfast."),
        _c(4, None, "Drank 250ml of water on 02/03/2025."),
    ]
    index = RecordIndex(items)

    assert [e.id for e, _ in index.search("refused teeth brushing", k=5)] == ["C2", "C3"]
    assert [e.id for e, _ in index.search("250ml", k=5)] == ["C4"]
    assert [e.id for e, _ in index.search("02/03/2025", k=5)] == ["C4"]
    assert index.search("refused", k=5, exclude={"C2"})[0][0].id == "C3"


def test_pack_gives_every_need_its_best_match_before_anything_else() -> None:
    evidence = [
        _a(1, "Dentures", "Yes"),
        _a(2, "Falls", "Fell twice"),
        *(_c(i, f"{i:02d}/10/2026", "Enjoyed the music session.") for i in range(1, 6)),
        _c(6, "01/01/2025", "Her dentures were lost.", historical=True),
        _c(7, "02/01/2025", "She fell in the lounge.", historical=True),
    ]
    must = count_tokens(evidence[0]) + count_tokens(evidence[1])
    room = count_tokens(evidence[7]) + count_tokens(evidence[8])

    pack = build_pack(evidence, pathway=None, max_must_read_tokens=10_000,
                      pack_tokens=must + room, matches_per_need=3)

    assert [e.id for e in pack.shown] == ["A1", "A2", "C6", "C7"]
    # The index covers every record, for the agent's search tool.
    assert len(pack.index.items) == len(evidence)


def test_must_read_evidence_is_never_cut() -> None:
    evidence = [_a(i, "Q", "word " * 200) for i in range(1, 6)]

    with pytest.raises(EvidenceTooLargeError):
        build_pack(evidence, pathway=None, max_must_read_tokens=100, pack_tokens=100,
                   matches_per_need=3)


def test_repeated_notes_are_shown_once_with_every_id_and_date() -> None:
    notes = [_c(1, "01/10/2026", "Refused to brush her teeth."),
             _c(2, "05/10/2026", "Refused to brush  her teeth."),
             _c(3, "06/10/2026", "Slept well.")]

    assert format_grouped(notes).splitlines() == [
        "[C1, C2] Carer note (x2: 01/10/2026, 05/10/2026; resident record): "
        "Refused to brush her teeth.",
        "[C3] Carer note (06/10/2026, resident record): Slept well.",
    ]


def test_pathway_related_notes_are_ranked_first_newest_first() -> None:
    must = [_a(1, "Dentures", "Yes"), _a(2, "Oral cleanliness", "Some plaque")]
    old_teeth = _c(1, "01/09/2026", "Plaque noticed on her dentures again.")
    new_music = _c(2, "10/10/2026", "Enjoyed the music session.")
    new_teeth = _c(3, "09/10/2026", "Refused to clean her dentures tonight.")

    ranked = rank_for_pathway(must, [old_teeth, new_music, new_teeth], "Oral Health Assessment")

    assert [e.id for e in ranked] == ["C3", "C1", "C2"]
