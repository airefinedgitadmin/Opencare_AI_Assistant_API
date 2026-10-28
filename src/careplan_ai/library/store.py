"""Vector stores for the style library, behind one interface.

* PineconeVectorStore - an integrated-embedding index (Pinecone embeds the
  text itself, so there is no separate embedding API). One NAMESPACE PER CARE
  HOME, so one home's care plans can never be retrieved for another.
* InMemoryVectorStore - tests and local dev; word-overlap scoring.

Swapping provider (e.g. to an Azure-hosted store) means adding a class here
and a value for CAREPLAN_AI_LIBRARY_PROVIDER - nothing else changes.
"""

import asyncio
import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any, Protocol

from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import ConfigurationError, LibraryStoreError
from careplan_ai.core.logging import get_logger
from careplan_ai.library.models import LibraryEntry

logger = get_logger(__name__)

# Tags stored next to the text (Pinecone record fields / filterable metadata).
_FIELDS = [
    "care_plan_id", "text", "pathway", "pathway_key", "assessment_template_id",
    "care_plan_date", "author_role", "status", "source", "edited", "excluded", "indexed_at",
    "home_name",
]
# Pinecone's limit on records per upsert_records call (integrated embedding).
_UPSERT_BATCH = 96


def namespace_for(home_id: int) -> str:
    # The id, never the name: a renamed home must keep its library. The name is
    # stored on every record (home_name) so the console stays readable.
    return f"home-{home_id}"


class VectorStore(Protocol):
    provider: str

    async def get(self, home_id: int, ids: list[str]) -> dict[str, LibraryEntry]: ...

    async def upsert(self, home_id: int, entries: list[LibraryEntry]) -> None: ...

    async def delete(self, home_id: int, ids: list[str]) -> None: ...

    async def set_excluded(self, home_id: int, entry_id: str, excluded: bool) -> bool: ...

    async def search(
        self, home_id: int, *, pathway_key: str, query: str, top_k: int
    ) -> list[tuple[LibraryEntry, float]]:
        """Active, non-excluded entries of this home and pathway, best first."""
        ...


_WORD = re.compile(r"[a-z]{3,}")


def _vector(text: str) -> Counter[str]:
    return Counter(_WORD.findall(text.casefold()))


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    dot = sum(count * b[word] for word, count in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class InMemoryVectorStore:
    provider = "memory"

    def __init__(self) -> None:
        self._homes: dict[str, dict[str, LibraryEntry]] = {}

    def _ns(self, home_id: int) -> dict[str, LibraryEntry]:
        return self._homes.setdefault(namespace_for(home_id), {})

    async def get(self, home_id: int, ids: list[str]) -> dict[str, LibraryEntry]:
        ns = self._ns(home_id)
        return {i: ns[i] for i in ids if i in ns}

    async def upsert(self, home_id: int, entries: list[LibraryEntry]) -> None:
        ns = self._ns(home_id)
        for entry in entries:
            ns[entry.id] = entry

    async def delete(self, home_id: int, ids: list[str]) -> None:
        ns = self._ns(home_id)
        for i in ids:
            ns.pop(i, None)

    async def set_excluded(self, home_id: int, entry_id: str, excluded: bool) -> bool:
        entry = self._ns(home_id).get(entry_id)
        if entry is None:
            return False
        entry.excluded = excluded
        return True

    async def search(
        self, home_id: int, *, pathway_key: str, query: str, top_k: int
    ) -> list[tuple[LibraryEntry, float]]:
        q = _vector(query)
        hits = [
            (entry, _cosine(q, _vector(entry.text)))
            for entry in self._ns(home_id).values()
            if entry.pathway_key == pathway_key and entry.status == "active" and not entry.excluded
        ]
        hits.sort(key=lambda hit: hit[1], reverse=True)
        return hits[:top_k]


def _entry_from_fields(entry_id: str, fields: dict[str, Any]) -> LibraryEntry:
    # Pinecone hands numbers back as floats; pydantic coerces them to int.
    data: dict[str, Any] = {k: fields[k] for k in _FIELDS if fields.get(k) is not None}
    for key in ("care_plan_id", "assessment_template_id"):
        if key in data:
            data[key] = int(data[key])
    return LibraryEntry.model_validate({"id": entry_id, **data})


class PineconeVectorStore:
    """Sync Pinecone SDK run in a worker thread, like the Postgres memory store."""

    provider = "pinecone"

    def __init__(
        self, *, api_key: str, index_name: str, cloud: str, region: str, embed_model: str
    ) -> None:
        self._api_key = api_key
        self._index_name = index_name
        self._cloud = cloud
        self._region = region
        self._embed_model = embed_model
        self._index_handle: Any = None

    def _index(self) -> Any:
        if self._index_handle is None:
            from pinecone import Pinecone

            pc = Pinecone(api_key=self._api_key)
            if not pc.has_index(self._index_name):
                # First use: create the integrated-embedding index ("text" is embedded).
                logger.info("library_index_creating", index=self._index_name,
                            cloud=self._cloud, region=self._region, model=self._embed_model)
                pc.create_index_for_model(
                    name=self._index_name,
                    cloud=self._cloud,
                    region=self._region,
                    embed={  # type: ignore[arg-type]  # TypedDict; plain dict is accepted
                        "model": self._embed_model,
                        "field_map": {"text": "text"},
                    },
                )
            self._index_handle = pc.Index(self._index_name)
        return self._index_handle

    async def _run(self, fn):  # type: ignore[no-untyped-def]
        def wrapper():  # type: ignore[no-untyped-def]
            try:
                return fn(self._index())
            except LibraryStoreError:
                raise
            except Exception as exc:  # the SDK raises many unrelated types
                logger.error("library_store_failed", error=type(exc).__name__, detail=str(exc))
                raise LibraryStoreError("Care plan library is unavailable") from exc

        return await asyncio.to_thread(wrapper)

    async def get(self, home_id: int, ids: list[str]) -> dict[str, LibraryEntry]:
        if not ids:
            return {}

        def _get(index: Any) -> dict[str, LibraryEntry]:
            response = index.fetch(ids=ids, namespace=namespace_for(home_id))
            return {
                vid: _entry_from_fields(vid, dict(vector.metadata or {}))
                for vid, vector in response.vectors.items()
            }

        return await self._run(_get)

    async def upsert(self, home_id: int, entries: list[LibraryEntry]) -> None:
        records = []
        for entry in entries:
            fields = entry.model_dump(exclude={"id"})
            # Pinecone rejects null field values.
            records.append({"_id": entry.id, **{k: v for k, v in fields.items() if v is not None}})

        def _upsert(index: Any) -> None:
            # Integrated-embedding indexes take at most 96 records per upsert.
            for start in range(0, len(records), _UPSERT_BATCH):
                index.upsert_records(namespace_for(home_id), records[start : start + _UPSERT_BATCH])

        await self._run(_upsert)

    async def delete(self, home_id: int, ids: list[str]) -> None:
        await self._run(lambda index: index.delete(ids=ids, namespace=namespace_for(home_id)))

    async def set_excluded(self, home_id: int, entry_id: str, excluded: bool) -> bool:
        if not await self.get(home_id, [entry_id]):
            return False
        await self._run(
            lambda index: index.update(
                id=entry_id, set_metadata={"excluded": excluded}, namespace=namespace_for(home_id)
            )
        )
        return True

    async def search(
        self, home_id: int, *, pathway_key: str, query: str, top_k: int
    ) -> list[tuple[LibraryEntry, float]]:
        def _search(index: Any) -> list[tuple[LibraryEntry, float]]:
            response = index.search(
                namespace=namespace_for(home_id),
                query={
                    "inputs": {"text": query},
                    "top_k": top_k,
                    "filter": {
                        "pathway_key": {"$eq": pathway_key},
                        "status": {"$eq": "active"},
                        "excluded": {"$eq": False},
                    },
                },
                fields=_FIELDS,
            )
            return [
                (_entry_from_fields(hit["_id"], dict(hit["fields"])), float(hit["_score"]))
                for hit in response.result.hits
            ]

        return await self._run(_search)


@lru_cache
def get_vector_store() -> VectorStore | None:
    """None when the library is switched off (CAREPLAN_AI_LIBRARY_PROVIDER=none)."""
    settings = get_settings()
    if settings.library_provider == "none":
        return None
    if settings.library_provider == "memory":
        return InMemoryVectorStore()
    if settings.pinecone_api_key is None:
        raise ConfigurationError(
            "CAREPLAN_AI_PINECONE_API_KEY is required for the Pinecone library"
        )
    return PineconeVectorStore(
        api_key=settings.pinecone_api_key.get_secret_value(),
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        embed_model=settings.pinecone_embed_model,
    )
