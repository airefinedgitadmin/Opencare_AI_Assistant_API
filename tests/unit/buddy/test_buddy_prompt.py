"""OPENCARE-102: the Buddy prompt is filled from the request, and the answer is read back."""

import re
from pathlib import Path

import pytest

from careplan_ai.buddy import prompt_files
from careplan_ai.buddy.prompt import build_prompt, recent_history, split_answer
from careplan_ai.buddy.schemas import BuddyChatRequest, BuddySource, BuddyTurn
from careplan_ai.core.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _env(base_env: None) -> None:
    """Settings (prompt folder, limits) come from the environment, as in the service."""


def _request(**overrides: object) -> BuddyChatRequest:
    data: dict[str, object] = {
        "request_id": "r1",
        "user": {"first_name": "Laura", "role": "Carer", "home": "Stobhill", "shift": "Day shift"},
        "now": "24/11/2026 10:15",
        "screen": "On the resident's record: Mr Ewan Gilchrist",
        "question": "How is he eating?",
        "context": {
            "residents": [
                {"id": "r1", "name": "Mr Ewan Gilchrist"},
                {"id": "r2", "name": "Mrs Agnes Duncan"},
            ],
            "tasks": [{"who": "Ewan Gilchrist", "task": "Fluid Chart", "status": "Overdue"}],
            "traffic_lights": [{"name": "Mr Ewan Gilchrist", "traffic_light": "Red"}],
            "guidance": [{"title": "Falls", "text": "Check for injury."}],
            "my_development": {"activity": {"period": "November 2026"}},
            "home_documents": [{"title": "Visiting Policy", "summary": "Visiting hours."}],
        },
    }
    data.update(overrides)
    return BuddyChatRequest.model_validate(data)


class TestBuildPrompt:
    def test_every_placeholder_is_filled(self) -> None:
        prompt = build_prompt(_request())
        assert not re.search(r"\$\{[a-z_]+\}", prompt)

    def test_user_screen_and_time(self) -> None:
        prompt = build_prompt(_request())
        assert (
            "WHO YOU ARE TALKING TO: Laura, Carer, Stobhill care home, Day shift. "
            "Today is Tuesday 24/11/2026; the time now is 10:15." in prompt
        )
        assert "SCREEN: On the resident's record: Mr Ewan Gilchrist" in prompt
        assert "(the only people you may discuss): Mr Ewan Gilchrist, Mrs Agnes Duncan." in prompt

    def test_the_home_is_never_hard_coded(self) -> None:
        prompt = build_prompt(_request())
        assert "Larchfield" not in prompt
        assert "dummy" not in prompt.lower()
        assert "I can only help with the residents at Stobhill and with using Open Care" in prompt

    def test_each_block_goes_under_its_heading(self) -> None:
        prompt = build_prompt(_request())
        blocks = [
            ("\nTRAFFIC LIGHTS (exactly", '[{"name":"Mr Ewan Gilchrist","traffic_light":"Red"}]'),
            ("\nMY DEVELOPMENT (the user", '"activity":{"period":"November 2026"}'),
            ("\nRECORDS:\n", '[{"id":"r1","name":"Mr Ewan Gilchrist"}'),
            ("\nDISCHARGED (no longer", "\n[]\n"),
            ("\nTODAY (live", '"task":"Fluid Chart","status":"Overdue"'),
            ("\nHOME DOCUMENTS (Buddy", '"title":"Visiting Policy"'),
        ]
        headings = [prompt.index(heading) for heading, _ in blocks]
        assert headings == sorted(headings)
        ends = headings[1:] + [len(prompt)]
        for (heading, data), start, end in zip(blocks, headings, ends, strict=True):
            assert data in prompt[start:end], heading

    def test_guidance_is_read_from_my_development(self) -> None:
        prompt = build_prompt(_request())
        assert '"best_practice_guidance":[{"title":"Falls","text":"Check for injury."}]' in prompt

    def test_text_in_the_data_that_looks_like_a_placeholder_is_left_alone(self) -> None:
        request = _request()
        request.context.residents[0]["notes"] = [{"text": "Wrote ${home} on the form"}]
        assert "Wrote ${home} on the form" in build_prompt(request)

    def test_rule_17_no_follow_up_questions(self) -> None:
        assert '17. Do not add follow-up questions or a "Next:" line.' in build_prompt(_request())


class TestSplitAnswer:
    def test_sources_line_is_parsed_and_removed(self) -> None:
        text, sources = split_answer(
            "Ewan ate well.\nSources: [Source: Notes · Carer Note · 23/11/2026 14:10] "
            "[Source: Care Plan · Eating and drinking · 01/11/2026]"
        )
        assert text == "Ewan ate well."
        assert sources == [
            BuddySource(area="Notes", item="Carer Note", date="23/11/2026 14:10"),
            BuddySource(area="Care Plan", item="Eating and drinking", date="01/11/2026"),
        ]

    def test_buddy_knowledge_source_has_no_date(self) -> None:
        _, sources = split_answer(
            "Visiting is open.\nSources: [Source: Buddy knowledge · Visiting Policy]"
        )
        assert sources == [BuddySource(area="Buddy knowledge", item="Visiting Policy", date="")]

    @pytest.mark.parametrize(
        "line",
        [
            "**Sources:** [Source: Notes · Carer Note · 23/11/2026 14:10]",
            "sources: [Source: Notes · Carer Note · 23/11/2026 14:10]",
        ],
    )
    def test_sources_line_variants(self, line: str) -> None:
        text, sources = split_answer("Answer.\n" + line)
        assert text == "Answer."
        assert len(sources) == 1

    def test_duplicates_dropped_and_at_most_five(self) -> None:
        items = " ".join(f"[Source: Notes · Carer Note · 2{i}/11/2026 10:00]" for i in range(7))
        _, sources = split_answer(
            "A.\nSources: [Source: Notes · Carer Note · 20/11/2026 10:00] " + items
        )
        assert len(sources) == 5
        assert len({s.date for s in sources}) == 5

    def test_next_line_removed(self) -> None:
        text, _ = split_answer("Answer.\nNext: What else? | Anything?")
        assert text == "Answer."

    def test_tokens_and_tables_kept(self) -> None:
        body = (
            "Here is today.\n[[today]]\n"
            "| Resident | Why |\n|---|---|\n| Ewan | Overdue: Fluid Chart |"
        )
        text, sources = split_answer(body)
        assert text == body
        assert sources == []


class TestNothingHardCoded:
    """The wording and limits come from files and settings, not from code."""

    def test_the_open_care_guide_comes_from_its_own_file(self) -> None:
        guide = prompt_files.load(prompt_files.OPEN_CARE_GUIDE)
        assert guide.startswith("OPEN CARE GUIDE (how the app works).")
        prompt = build_prompt(_request())
        assert guide in prompt
        assert prompt.index(guide) < prompt.index("\nTRAFFIC LIGHTS (exactly")

    def test_another_prompt_folder_changes_the_wording_without_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "chat_prompt.md").write_text(
            "Hello ${user_name} at ${home}.\n${open_care_guide}\nRECORDS:\n${residents}\n",
            encoding="utf-8",
        )
        (tmp_path / "open_care_guide.md").write_text("GUIDE v2", encoding="utf-8")
        monkeypatch.setenv("CAREPLAN_AI_BUDDY_PROMPT_DIR", str(tmp_path))
        assert build_prompt(_request()) == (
            "Hello Laura at Stobhill.\nGUIDE v2\nRECORDS:\n"
            '[{"id":"r1","name":"Mr Ewan Gilchrist"},{"id":"r2","name":"Mrs Agnes Duncan"}]'
        )

    def test_a_missing_prompt_file_is_a_clear_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("CAREPLAN_AI_BUDDY_PROMPT_DIR", str(tmp_path))
        with pytest.raises(ConfigurationError, match="prompt file not found"):
            build_prompt(_request())

    def test_an_unknown_placeholder_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "chat_prompt.md").write_text("Hi ${user_name} ${nhs_number}", encoding="utf-8")
        (tmp_path / "open_care_guide.md").write_text("", encoding="utf-8")
        monkeypatch.setenv("CAREPLAN_AI_BUDDY_PROMPT_DIR", str(tmp_path))
        with pytest.raises(ConfigurationError, match="nhs_number"):
            build_prompt(_request())

    def test_the_shipped_prompt_uses_only_known_placeholders(self) -> None:
        build_prompt(_request())  # raises on an unknown one

    def test_history_turns_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CAREPLAN_AI_BUDDY_HISTORY_TURNS", "2")
        turns = [
            BuddyTurn(role="user" if i % 2 == 0 else "assistant", text=f"t{i}") for i in range(6)
        ]
        assert [t.text for t in recent_history(turns)] == ["t4", "t5"]

    def test_max_sources_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CAREPLAN_AI_BUDDY_MAX_SOURCES", "2")
        items = " ".join(f"[Source: Notes · Carer Note · 2{i}/11/2026 10:00]" for i in range(4))
        _, sources = split_answer("A.\nSources: " + items)
        assert len(sources) == 2
