"""Encryption for the little patient text agent memory keeps.

Agent memory never stores record text (memory/references.py keeps references
to the backend's records instead). What it must keep between calls is the
AI's own wording - suggestions, staff edits, statements of saved plans - and
that is sealed here with a Fernet key (AES-128-CBC + HMAC-SHA256) held in the
app's settings, so the database alone can't read it.

The same key keys the record fingerprints (HMAC-SHA256), so a fingerprint of
a short answer like "Yes" can't be reversed by guessing.
"""

import hashlib
import hmac
import json
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

from careplan_ai.core.config import get_settings
from careplan_ai.core.exceptions import ConfigurationError, MemoryStoreError


@lru_cache
def _key() -> bytes:
    settings = get_settings()
    key = settings.agent_state_key
    if key is None:
        if settings.memory_db_dsn is not None:
            raise ConfigurationError(
                "CAREPLAN_AI_AGENT_STATE_KEY must be set when agent memory uses a database"
            )
        # In-memory storage dies with the process; so may its key.
        return Fernet.generate_key()
    return key.get_secret_value().strip().encode()


@lru_cache
def _fernet() -> Fernet:
    try:
        return Fernet(_key())
    except ValueError as exc:
        raise ConfigurationError("CAREPLAN_AI_AGENT_STATE_KEY is not a valid Fernet key") from exc


def seal(data: object) -> str:
    return _fernet().encrypt(json.dumps(data, default=str).encode()).decode()


def unseal(token: str) -> object:
    try:
        return json.loads(_fernet().decrypt(token.encode()))
    except InvalidToken as exc:
        raise MemoryStoreError("Agent memory could not be decrypted (wrong state key?)") from exc


def fingerprint(*parts: str | None) -> str:
    """A keyed fingerprint of a record's content - equal content, equal print."""
    message = "\x1f".join(p or "" for p in parts).encode()
    return hmac.new(_fingerprint_key(), message, hashlib.sha256).hexdigest()[:24]


@lru_cache
def _fingerprint_key() -> bytes:
    _fernet()  # validates the key
    return hashlib.sha256(b"careplan-ai-fingerprint\x00" + _key()).digest()
