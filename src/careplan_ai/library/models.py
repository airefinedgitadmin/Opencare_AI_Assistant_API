"""Wire and storage models for the style library."""

from typing import Literal

from pydantic import Field

from careplan_ai.api.schemas.base import CamelModel

LibrarySource = Literal["staff_written", "ai_assisted"]
LibraryStatus = Literal["active", "archived"]


class LibraryCarePlanIn(CamelModel):
    """One saved care plan, as the backend sends it for indexing.

    Names are sent only so they can be replaced with placeholders before
    anything is stored; they are never stored themselves."""

    care_plan_id: int
    pathway: str
    assessment_template_id: int | None = None
    title: str | None = None
    action: str
    outcome: str | None = None
    care_plan_date: str | None = None
    author_role: str | None = None
    status: LibraryStatus = "active"
    source: LibrarySource = "staff_written"
    # AI-assisted plans only: did staff change any AI wording before saving?
    edited: bool = False
    resident_names: list[str] = Field(default_factory=list)
    # Staff, relatives, professionals - anyone else named in the text.
    other_names: list[str] = Field(default_factory=list)


class LibraryBatchIn(CamelModel):
    # The care home's name, e.g. "Stobhill" - a readable label on each record.
    home_name: str | None = None
    items: list[LibraryCarePlanIn] = Field(min_length=1, max_length=100)


class LibraryExclusionIn(CamelModel):
    excluded: bool = True


class LibraryEntry(CamelModel):
    """What is stored in the vector store: anonymised text plus tags."""

    id: str
    care_plan_id: int
    text: str
    pathway: str
    pathway_key: str
    assessment_template_id: int | None = None
    care_plan_date: str | None = None
    author_role: str | None = None
    status: LibraryStatus = "active"
    source: LibrarySource = "staff_written"
    edited: bool = False
    excluded: bool = False
    indexed_at: str
    home_name: str | None = None


class StyleExample(CamelModel):
    """A retrieved library entry, as handed to the drafting model."""

    id: str
    care_plan_id: int
    text: str
    source: LibrarySource
    edited: bool = False
    score: float = 0.0


class LibraryIndexResult(CamelModel):
    enabled: bool
    indexed: list[str] = Field(default_factory=list)
    # Excluded entries keep their exclusion when re-indexed.
    excluded: list[str] = Field(default_factory=list)


def pathway_key(pathway: str) -> str:
    return " ".join(pathway.split()).casefold()


def entry_id(care_plan_id: int) -> str:
    return f"cp-{care_plan_id}"
