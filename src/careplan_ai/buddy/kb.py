"""Buddy knowledge indexing (OPENCARE-109): read a home's policy or procedure and suggest its
title, summary, tags, type and main points. The manager checks and edits every field before
saving; nothing here is stored or used by Buddy until they do.
"""

import io
import json
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from careplan_ai.buddy import prompt_files
from careplan_ai.buddy.service import get_buddy_model
from careplan_ai.core.config import get_settings
from careplan_ai.core.logging import get_logger

logger = get_logger(__name__)

KB_INDEX_PROMPT = "kb_index_prompt.md"


class NoReadableText(Exception):
    """The file has no text that can be read (a scan, an empty or broken file)."""


class KbIndexFailed(Exception):
    """The model failed or returned something that isn't a valid suggestion."""


class KbIndexResult(BaseModel):
    title: str = Field(min_length=1)
    summary: str = ""
    tags: list[str]
    kind: str
    emergency: str | None = None
    emergency_label: str | None = None
    text: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def _tags(cls, value: Any) -> list[str]:
        tags: list[str] = []
        for tag in value or []:
            t = re.sub(r"\s+", " ", str(tag)).strip().lower()
            if t and t not in tags:
                tags.append(t)
        return tags

    @field_validator("text", mode="before")
    @classmethod
    def _text(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        return [str(v).strip() for v in (value or []) if str(v).strip()]


def extract_text(file_name: str, data: bytes) -> str:
    """The document's text: PDF, Word (.docx) or plain text."""
    name = file_name.lower()
    try:
        if name.endswith(".pdf"):
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif name.endswith(".docx"):
            from docx import Document

            doc = Document(io.BytesIO(data))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(parts)
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
    except Exception as exc:  # a broken or encrypted file reads as no text
        raise NoReadableText(str(exc)) from exc
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        raise NoReadableText("no text in the file")
    return text


def build_kb_prompt(file_name: str, text: str, emergency_ids: list[str]) -> str:
    settings = get_settings()
    values = {
        "kinds": ", ".join(f'"{k}"' for k in settings.buddy_kb_kinds),
        "emergency_ids": ", ".join(f'"{e}"' for e in emergency_ids),
        "file_name": file_name,
        "document": text[: settings.buddy_kb_max_chars],
    }
    template = prompt_files.load(KB_INDEX_PROMPT)
    return prompt_files.PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def _json_object(reply: str) -> dict[str, Any]:
    """The JSON object in the reply (models sometimes wrap it in a code fence)."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise KbIndexFailed("no JSON in the reply")
    try:
        value = json.loads(reply[start : end + 1])
    except json.JSONDecodeError as exc:
        raise KbIndexFailed("invalid JSON") from exc
    if not isinstance(value, dict):
        raise KbIndexFailed("not a JSON object")
    return value


def validate(raw: dict[str, Any], emergency_ids: list[str]) -> KbIndexResult:
    """Tags 3-6 (more are cut to 6), kind in the allowed list, emergency an allowed id or null."""
    settings = get_settings()
    try:
        result = KbIndexResult.model_validate(raw)
    except Exception as exc:
        raise KbIndexFailed(f"invalid suggestion: {exc}") from exc
    result.tags = result.tags[: settings.buddy_kb_max_tags]
    if len(result.tags) < settings.buddy_kb_min_tags:
        raise KbIndexFailed("too few tags")
    if result.emergency is not None and result.emergency not in emergency_ids:
        result.emergency = None
    if result.emergency is not None:
        result.kind = settings.buddy_kb_emergency_kind
    if result.kind not in settings.buddy_kb_kinds:
        raise KbIndexFailed(f"unknown kind {result.kind!r}")
    # "Other" names are the manager's to type; the model never sets one.
    result.emergency_label = None
    return result


async def index(file_name: str, data: bytes, emergency_ids: list[str]) -> KbIndexResult:
    text = extract_text(file_name, data)
    started = time.monotonic()
    try:
        reply = await get_buddy_model().ainvoke(
            [HumanMessage(content=build_kb_prompt(file_name, text, emergency_ids))]
        )
    except Exception as exc:
        logger.error("buddy_kb_index_failed", error=type(exc).__name__)
        raise KbIndexFailed("model call failed") from exc
    content = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
    result = validate(_json_object(content), emergency_ids)
    logger.info(
        "buddy_kb_indexed",
        chars=len(text),
        tags=len(result.tags),
        kind=result.kind,
        latency_ms=int((time.monotonic() - started) * 1000),
    )
    return result
