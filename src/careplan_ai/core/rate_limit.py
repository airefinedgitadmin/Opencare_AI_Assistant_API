"""Per-caller rate limiting for the LLM-calling endpoint.

Keyed by the calling end user's fingerprint when the Java backend sends one
(see `CALLER_FINGERPRINT_HEADER`), falling back to the shared `X-Api-Key` for
any caller that doesn't send it. The API key alone is NOT a good key for this:
every real request today comes from the same backend service using the same
key, so keying on it bucket-shares the whole platform's traffic under one
limit - a handful of staff using the feature in the same minute could exhaust
it. Keying per end user instead means one busy person can't starve everyone
else.

The fingerprint is a salted HMAC computed by Java, not a real user id - this
service only ever sees an opaque token it can compare for equality (same
person -> same token), never something it can decode back to a real identity.
That's intentional: this service doesn't need to know WHO is calling, only
whether two requests came from the same caller, and identity is already
tracked properly, once, in Java's own audit log.

In-memory storage is enough for a single-instance deployment; if this ever runs
as more than one replica, point slowapi at a shared backend (e.g. Redis)
instead - see slowapi's `storage_uri` option.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from careplan_ai.api.dependencies import API_KEY_HEADER, CALLER_FINGERPRINT_HEADER
from careplan_ai.core.config import get_settings


def _rate_limit_key(request: Request) -> str:
    fingerprint = request.headers.get(CALLER_FINGERPRINT_HEADER)
    if fingerprint:
        return f"caller:{fingerprint}"
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        return f"apikey:{api_key}"
    return get_remote_address(request)


def generation_limit_value() -> str:
    """Read the limit from settings at request time, not import time.

    A plain string passed to `@limiter.limit(...)` would be fixed at module
    import - before `Settings` can be constructed in tests. slowapi accepts a
    callable instead specifically for this kind of runtime-configurable limit.
    """
    settings = get_settings()
    return f"{settings.rate_limit_requests}/{settings.rate_limit_window_seconds}seconds"


limiter = Limiter(key_func=_rate_limit_key)
