"""OPENCARE-109: POST /buddy/kb/index - read a home document and suggest its details.

The model is faked."""

import io

import pytest
from docx import Document
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage

from careplan_ai.api.app import create_app
from careplan_ai.buddy import kb, service

pytestmark = pytest.mark.integration

HEADERS = {"X-Api-Key": "test-api-key"}

FIRE = (
    "Fire Procedure\nRaise the alarm at the nearest call point.\n"
    "Phone 999 and ask for the Fire Service.\n"
    "Close doors and windows.\nMove residents to the next fire zone.\nDo not use the lift."
)


class FakeModel:
    def __init__(self, reply: str = "", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.sent: list[BaseMessage] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.sent = messages
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    monkeypatch.setattr(kb, "get_buddy_model", lambda: model)
    monkeypatch.setattr(service, "get_buddy_model", lambda: model)
    return model


def _post(client: TestClient, name: str, data: bytes, ids: str = "") -> object:
    return client.post(
        "/buddy/kb/index",
        files={"file": (name, data, "application/octet-stream")},
        data={"home_name": "Larchfield", "emergency_ids": ids},
        headers=HEADERS,
    )


FIRE_REPLY = (
    '```json\n{"title": "Fire Procedure", "summary": "What staff do when the fire alarm sounds.",'
    ' "tags": ["Fire", "smoke", "fire alarm", "evacuation", "assembly point",'
    ' "fire", "zones", "extra"],'
    ' "kind": "Procedure", "emergency": "fire",'
    ' "text": ["Raise the alarm at the nearest call point.",'
    ' "Phone 999 and ask for the Fire Service."]}\n```'
)


def test_a_fire_procedure_is_suggested_as_an_emergency_procedure(
    base_env: None, fake: FakeModel
) -> None:
    fake.reply = FIRE_REPLY
    with TestClient(create_app()) as client:
        response = _post(client, "Fire_Procedure.txt", FIRE.encode())

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Fire Procedure"
    assert body["kind"] == "Emergency procedure"
    assert body["emergency"] == "fire"
    assert body["emergency_label"] is None
    # Lower case, no repeats, at most 6.
    assert body["tags"] == ["fire", "smoke", "fire alarm", "evacuation", "assembly point", "zones"]
    assert body["text"][0] == "Raise the alarm at the nearest call point."
    prompt = str(fake.sent[0].content)
    assert '"Policy", "Procedure", "Guidance", "Emergency procedure"' in prompt
    assert '"collapse", "fire", "missing", "power", "flood", "gas"' in prompt
    assert "DOCUMENT (Fire_Procedure.txt):\nFire Procedure\nRaise the alarm" in prompt


def test_the_file_is_read_off_the_event_loop(
    base_env: None, fake: FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code review: reading a large PDF must not hold up other requests on this worker."""
    import threading

    seen: list[str] = []
    real = kb.extract_text

    def spy(file_name: str, data: bytes) -> str:
        seen.append(threading.current_thread().name)
        return real(file_name, data)

    monkeypatch.setattr(kb, "extract_text", spy)
    fake.reply = FIRE_REPLY
    with TestClient(create_app()) as client:
        loop_thread: list[str] = []

        async def where() -> None:
            loop_thread.append(threading.current_thread().name)

        client.portal.call(where)
        assert _post(client, "Fire_Procedure.txt", FIRE.encode()).status_code == 200

    assert seen and seen[0] != loop_thread[0]


def test_a_policy(base_env: None, fake: FakeModel) -> None:
    fake.reply = (
        '{"title": "Visiting Policy", "summary": "When and how family can visit.",'
        ' "tags": ["visiting", "visitors", "family"], "kind": "Policy", "emergency": null,'
        ' "text": ["Visiting is open from 10:00 to 20:00."]}'
    )
    with TestClient(create_app()) as client:
        body = _post(
            client, "Visiting.txt", b"Visiting Policy. Visiting is open from 10:00 to 20:00."
        ).json()
    assert body["kind"] == "Policy"
    assert body["emergency"] is None
    assert len(body["tags"]) == 3


def test_the_homes_own_other_types_are_offered(base_env: None, fake: FakeModel) -> None:
    fake.reply = (
        '{"title": "Lift", "summary": "If the lift breaks down.",'
        ' "tags": ["lift", "breakdown", "stuck"], "kind": "Emergency procedure",'
        ' "emergency": "x-lift-breakdown", "text": ["Call the engineer."]}'
    )
    with TestClient(create_app()) as client:
        body = _post(
            client, "Lift.txt", b"Lift breakdown procedure", ids="collapse,fire,x-lift-breakdown"
        ).json()
    assert body["emergency"] == "x-lift-breakdown"
    assert '"x-lift-breakdown"' in str(fake.sent[0].content)


def test_an_emergency_id_not_offered_is_dropped(base_env: None, fake: FakeModel) -> None:
    fake.reply = (
        '{"title": "T", "summary": "S", "tags": ["a", "b", "c"], "kind": "Procedure",'
        ' "emergency": "earthquake", "text": []}'
    )
    with TestClient(create_app()) as client:
        body = _post(client, "T.txt", b"text").json()
    assert body["emergency"] is None
    assert body["kind"] == "Procedure"


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        '{"title": "T", "summary": "S", "tags": ["a", "b"], "kind": "Policy", "text": []}',
        '{"title": "T", "summary": "S", "tags": ["a", "b", "c"], "kind": "Memo", "text": []}',
        '{"title": "", "summary": "S", "tags": ["a", "b", "c"], "kind": "Policy", "text": []}',
    ],
    ids=["no json", "too few tags", "unknown kind", "no title"],
)
def test_an_invalid_suggestion_is_ai_unavailable(
    base_env: None, fake: FakeModel, reply: str
) -> None:
    fake.reply = reply
    with TestClient(create_app()) as client:
        response = _post(client, "T.txt", b"some text")
    assert response.status_code == 502
    assert response.json() == {"error": "ai_unavailable"}


def test_model_down_is_ai_unavailable(base_env: None, fake: FakeModel) -> None:
    fake.error = RuntimeError("Azure 500")
    with TestClient(create_app()) as client:
        assert _post(client, "T.txt", b"some text").status_code == 502


def test_no_readable_text(base_env: None, fake: FakeModel) -> None:
    with TestClient(create_app()) as client:
        response = _post(client, "Scan.txt", b"   \n  ")
        broken = _post(client, "Broken.pdf", b"%PDF-1.4 not really a pdf")
    assert response.status_code == 422
    assert response.json() == {"error": "no_text"}
    assert broken.status_code == 422
    assert fake.sent == []


def test_over_the_size_limit(
    base_env: None, monkeypatch: pytest.MonkeyPatch, fake: FakeModel
) -> None:
    monkeypatch.setenv("CAREPLAN_AI_BUDDY_KB_MAX_BYTES", "10")
    with TestClient(create_app()) as client:
        response = _post(client, "Big.txt", b"x" * 11)
    assert response.status_code == 413
    assert fake.sent == []


def test_requires_api_key(base_env: None, fake: FakeModel) -> None:
    with TestClient(create_app()) as client:
        response = client.post("/buddy/kb/index", files={"file": ("T.txt", b"x")})
    assert response.status_code == 401


def _docx(lines: list[str]) -> bytes:
    doc = Document()
    for line in lines:
        doc.add_paragraph(line)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Assembly point"
    table.rows[0].cells[1].text = "Front car park"
    out = io.BytesIO()
    doc.save(out)
    return out.getvalue()


def _pdf(text: str) -> bytes:
    """A minimal one-page PDF with the text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


class TestExtractText:
    def test_word_paragraphs_and_tables(self) -> None:
        text = kb.extract_text("Fire.docx", _docx(["Fire Procedure", "Raise the alarm."]))
        assert "Fire Procedure\nRaise the alarm." in text
        assert "Assembly point | Front car park" in text

    def test_pdf(self) -> None:
        assert "Raise the alarm" in kb.extract_text("Fire.pdf", _pdf("Raise the alarm"))

    def test_plain_text_utf8_and_latin1(self) -> None:
        assert kb.extract_text("a.txt", "Café hours".encode()) == "Café hours"
        assert kb.extract_text("a.txt", "Café hours".encode("latin-1")) == "Café hours"

    def test_only_the_first_characters_go_to_the_model(self, base_env: None) -> None:
        prompt = kb.build_kb_prompt("Long.txt", "y" * 20_000, ["fire"])
        document = prompt.split("DOCUMENT (Long.txt):\n", 1)[1].rstrip("\n")
        assert len(document) == 12_000
