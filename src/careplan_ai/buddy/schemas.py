"""The /buddy/chat contract (OPENCARE-102).

Java (the only caller) sends the question and the data blocks it built for the signed-in
user's home; this service fills the prompt, calls the model and returns the answer with its
sources split out. The data blocks are passed through to the prompt as Java built them, so
their inner shape is owned by the Java context builder (OPENCARE-103), not by this service.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class BuddyUser(BaseModel):
    first_name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    home: str = Field(min_length=1)
    shift: str = ""


class BuddyTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class BuddyContext(BaseModel):
    """The data blocks Open Care built for this question. Nothing else reaches the AI."""

    # Extra blocks are refused rather than silently ignored, so a block Java adds later is
    # never sent to the model without the prompt knowing where it goes.
    model_config = ConfigDict(extra="forbid")

    residents: list[dict[str, Any]] = Field(default_factory=list)
    discharged: list[dict[str, Any]] = Field(default_factory=list)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    traffic_lights: list[dict[str, Any]] = Field(default_factory=list)
    my_development: dict[str, Any] = Field(default_factory=dict)
    guidance: list[dict[str, Any]] = Field(default_factory=list)
    home_documents: list[dict[str, Any]] = Field(default_factory=list)


class BuddyChatRequest(BaseModel):
    request_id: str = Field(min_length=1)
    user: BuddyUser
    # "dd/mm/yyyy HH:MM", the home's local time.
    now: str = Field(pattern=r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$")
    screen: str = ""
    question: str = Field(min_length=1)
    history: list[BuddyTurn] = Field(default_factory=list)
    context: BuddyContext = Field(default_factory=BuddyContext)


class BuddySource(BaseModel):
    area: str
    item: str
    date: str = ""


class BuddyChatResponse(BaseModel):
    answer_markdown: str
    sources: list[BuddySource]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
