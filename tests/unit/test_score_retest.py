"""The retest scorer (evals/score_retest.py) reads a draft as copied from the screen and finds
what the retest document expects - checked on Agnes's real Falls draft of 26/11/2026."""

import sys
from pathlib import Path

import pytest

EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(EVALS))

import score_retest as sr  # noqa: E402

DATA = sr.load_expected()


def _expected(resident: str, pathway: str) -> dict:
    expected = sr.expected_for(DATA, resident, pathway)
    assert expected is not None
    return expected


def test_agnes_falls_draft_scores_the_gaps_a_reviewer_found() -> None:
    draft = sr.parse_draft((EVALS / "samples" / "81362_falls.txt").read_text(encoding="utf-8"))
    assert len(draft.actions) == 9 and len(draft.outcomes) == 4
    s = sr.score(_expected("81362", "falls"), draft)
    assert s.actions_missed == [
        "Toilet before bed",
        "Toilet after the morning diuretic",
        "Closed footwear with grip",
        "Night light",
        "GP/pharmacist medicines review",
    ]
    assert s.outcomes_missed == ["Wears safe footwear"]
    assert s.must_not_hits == [] and s.sources_missed == []
    assert (s.found, s.total) == (10, 16)


AGNES_EDITED = """Actions:
Keep Agnes's walker beside her chair and bed, and remind her to use it when she stands.
Place a sensor mat beside her bed at night and go to her straight away when it alarms.
Offer the toilet before bed, at night checks and after her morning diuretic.
Walk with her to the bathroom at night.
Help her sit on the edge of the bed for a moment before standing.
Check her lying and standing blood pressure weekly.
Ask the GP/pharmacist to review her medicines.
Make sure she wears closed, well-fitting footwear with grip.
Leave a night light on and keep her room clear of clutter.
Outcomes:
Agnes has no further falls by the next review.
Agnes uses her walker every time she stands or walks.
Agnes gets to the toilet safely at night with staff.
Agnes reports less dizziness on standing.
Agnes wears safe footwear every day.
Sources:
Observation - Falls Diary
Observation - TPR & BP
"""


def test_the_plan_a_senior_carer_would_write_passes() -> None:
    s = sr.score(_expected("81362", "falls"), sr.parse_draft(AGNES_EDITED))
    assert s.passed, sr.report(s, "edited")


@pytest.mark.parametrize("line", [
    "Agnes walks independently with her walker.",
    "Encourage Agnes as she walks independently to the lounge.",
    "Do not rush her; she walks independently.",
])
def test_out_of_date_care_is_a_must_not_hit(line: str) -> None:
    draft = sr.parse_draft(f"Actions:\n{line}\n")
    assert sr.score(_expected("81362", "falls"), draft).must_not_hits


@pytest.mark.parametrize("line", [
    "Agnes no longer walks independently; walk with her at night.",
    "Do not rush her, as she can no longer walk independently.",
    "Do not give Rab a normal diet or thin fluids.",
])
def test_saying_it_no_longer_applies_is_not_a_hit(line: str) -> None:
    for rid, key in (("81362", "falls"), ("81363", "choking")):
        draft = sr.parse_draft(f"Actions:\n{line}\n")
        assert sr.score(_expected(rid, key), draft).must_not_hits == []


def test_weigh_monthly_is_a_hit_only_without_weekly() -> None:
    expected = _expected("81363", "must")
    only_monthly = sr.parse_draft("Actions:\nWeigh Rab monthly.\n")
    both = sr.parse_draft("Actions:\nWeigh Rab weekly.\nRescreen MUST and weigh monthly.\n")
    assert sr.score(expected, only_monthly).must_not_hits
    assert sr.score(expected, both).must_not_hits == []


def test_a_low_risk_pathway_with_many_actions_is_reported() -> None:
    lines = "\n".join(f"Remind Nell to slow down at meal {i}." for i in range(6))
    s = sr.score(_expected("81366", "choking"), sr.parse_draft(f"Actions:\n{lines}\n"))
    assert s.too_many_actions is not None


def test_every_pathway_of_the_document_is_there_and_every_pattern_compiles() -> None:
    keys = {(rid, key) for rid, r in DATA["residents"].items() for key in r["pathways"]}
    assert len(keys) == 20  # 5 residents x 4 pathways
    for rid, key in keys:
        expected = _expected(rid, key)
        rules = expected["actions"] + expected["outcomes"] + expected.get("must_not", [])
        for rule in rules:
            for pattern in rule["any"]:
                sr.find(pattern, "anything")


def test_the_file_name_picks_the_resident_and_pathway(tmp_path: Path) -> None:
    (tmp_path / "81362_falls.txt").write_text(AGNES_EDITED, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert sr.main([str(tmp_path)]) == 0
